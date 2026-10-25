
"""Implementation of note onset and frame detection models

this is rightMagentaofOnsets and Framesrough translation of implementation [1].
This model is used in music transcription and is able to detect the onset of a note、Finish、Sustained frame and velocity.

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
from torch import nn

from .losses import DEFAULT_LOSS_WEIGHTS, compute_multi_task_losses
from .multi_task_heads import MLPBranchFactory, MultiTaskHeads, PostDecoderProcessor
from .music_encoder import MusicEncoder
from .sequence_decoder import build_sequence_decoder, count_trainable_parameters


def _prepare_batch_tensors(batch, device):
    """
    Move the batch tensors needed by the assembly path onto the target device.
    """
    audio = batch['audio'].to(device)
    onset_label = batch['onset'].to(device)
    frame_label = batch['frame'].to(device)
    velocity_label = batch['velocity'].to(device)
    return audio, onset_label, frame_label, velocity_label


def _slice_time_dimension(tensor, target_time_dim, has_batch_dim):
    """
    Slice sequence outputs to match the label time dimension.
    """
    if has_batch_dim:
        return tensor[:, :target_time_dim, :]
    return tensor[:target_time_dim, :]


def _format_batch_outputs(
    onset_logits,
    frame_logits,
    velocity_logits,
    onset_label,
    frame_label,
    velocity_label,
):
    """
    Align decoder head outputs with the label layout used by training and evaluation.
    """
    onset_pred = torch.sigmoid(onset_logits)
    frame_pred = torch.sigmoid(frame_logits)
    velocity_pred = torch.sigmoid(velocity_logits)

    has_batch_dim = onset_label.dim() == 3
    if not has_batch_dim:
        onset_logits = onset_logits.squeeze(0)
        frame_logits = frame_logits.squeeze(0)
        onset_pred = onset_pred.squeeze(0)
        frame_pred = frame_pred.squeeze(0)
        velocity_pred = velocity_pred.squeeze(0)
        target_time_dim = onset_label.shape[0]

        # Why: evaluation path may feed a single example without batch dimension,
        # so the batch runner normalizes shapes here instead of leaking this detail outward.
        onset_logits = _slice_time_dimension(onset_logits, target_time_dim, has_batch_dim=False)
        frame_logits = _slice_time_dimension(frame_logits, target_time_dim, has_batch_dim=False)
        onset_pred = _slice_time_dimension(onset_pred, target_time_dim, has_batch_dim=False)
        frame_pred = _slice_time_dimension(frame_pred, target_time_dim, has_batch_dim=False)
        velocity_pred = _slice_time_dimension(velocity_pred, target_time_dim, has_batch_dim=False)

        predictions = {
            'onset': onset_pred,
            'frame': frame_pred,
            'velocity': velocity_pred,
        }
        return predictions, onset_logits, frame_logits

    target_time_dim = onset_label.shape[1]
    onset_logits = _slice_time_dimension(onset_logits, target_time_dim, has_batch_dim=True)
    frame_logits = _slice_time_dimension(frame_logits, target_time_dim, has_batch_dim=True)
    onset_pred = _slice_time_dimension(onset_pred, target_time_dim, has_batch_dim=True)
    frame_pred = _slice_time_dimension(frame_pred, target_time_dim, has_batch_dim=True)
    velocity_pred = _slice_time_dimension(velocity_pred, target_time_dim, has_batch_dim=True)

    predictions = {
        'onset': onset_pred.reshape(onset_label.shape),
        'frame': frame_pred.reshape(frame_label.shape),
        'velocity': velocity_pred.reshape(velocity_label.shape),
    }
    return predictions, onset_logits.reshape(onset_label.shape), frame_logits.reshape(frame_label.shape)


def run_transcription_batch(model, batch, loss_weights=None):
    """
    Run one training/evaluation batch against the assembled transcription model.
    """
    if loss_weights is None:
        loss_weights = DEFAULT_LOSS_WEIGHTS
    device = next(model.parameters()).device
    audio, onset_label, frame_label, velocity_label = _prepare_batch_tensors(batch, device)

    # Why: batch execution is not part of the encoder/adapter/decoder/heads decomposition itself,
    # so it lives at module scope and the model class only keeps a thin compatibility wrapper.
    onset_logits, frame_logits, velocity_logits = model(audio)
    predictions, onset_logits, frame_logits = _format_batch_outputs(
        onset_logits,
        frame_logits,
        velocity_logits,
        onset_label,
        frame_label,
        velocity_label,
    )
    losses = compute_multi_task_losses(
        onset_logits,
        frame_logits,
        predictions['velocity'],
        onset_label,
        frame_label,
        velocity_label,
        onset_weight=loss_weights[0],
        frame_weight=loss_weights[1],
        velocity_weight=loss_weights[2],
    )
    return predictions, losses


class OnsetsAndFrames(nn.Module):
    """
    Main models for note onset and frame detection
    
    This is a multi-task learning model that simultaneously predicts:
    1. onset: note onset time
    2. offset: note end time  
    3. frame: note duration frame
    4. velocity: note velocity
    
    supportQwenMode: useMERT->QwenPipe, go through firstMERTExtract features and then passQwenfor further processing
    
    Data flow:audio → feature_extractor → qwen_decoder → MLP branches → predictions
    
    Args:
        input_features: Input feature dimensions (retain compatibility, not actually used)
        output_features: Output feature dimensions (usuallyMIDInumber of notes)
        model_complexity: Model complexity parameters (compatibility retained, not actually used)
        encoder_model: Front-end model types, supported "MERT_95m", "MERT_330m", "MusicFM", "MuQ"
        qwen_config: QwenFront-end configuration dictionary
    """
    def __init__(self, output_features, model_complexity=48, 
                 encoder_model="MERT_95m", device="cpu",  # supportGPUAccelerate, defaultcpu
                 encoder_layer_selection="1,5,10,15,20,23",
                 decoder_layer_selection="all",
                 decoder_model="Qwen2.5-0.5B",  # support:Qwen2.5-0.5B, GLM-0.5B, Qwen3-0.6B, Hunyuan-0.5B-Instruct, MiniCPM4-0.5B, Falcon-H1-0.5B-Base
                 decoder_freeze_model=True,
                 decoder_backend="llm",
                 encoder_freeze=False,
                 enable_branch1=True, # Parallel convolution branch
                 enable_branch2=True, # dilated convolution branch
                 enable_branch3=True, # Transformerbranch
                 head_mode=PostDecoderProcessor.HEAD_MODE_INDEPENDENT):
        super().__init__()
        
        self.output_features = output_features
        self.device = device
        self.head_mode = head_mode
        self.decoder_backend = decoder_backend
        self.encoder_freeze = encoder_freeze
        
        # Initialize feature extractor
        self._init_feature_extractor(
            encoder_model, encoder_layer_selection, freeze_all=encoder_freeze
        )
        
        # Save encoder model name
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
            
        # Why: B1 ablations need a real LLM→scratch/none swap; keep attribute name
        # `qwen_decoder` for checkpoint / logging compatibility.
        print(
            f"Sequence decoder backend={decoder_backend}, "
            f"ref_model={decoder_model}, freeze_llm={decoder_freeze_model}, "
            f"encoder_freeze={encoder_freeze}"
        )
        self.qwen_decoder = build_sequence_decoder(
            decoder_backend=decoder_backend,
            decoder_model=decoder_model,
            encoder_model=encoder_model,
            feature_dim=self._get_encoder_feature_dim(encoder_model),
            # Why: encoder/decoder are separate; decoder layer selection must use its own config.
            layer_selection=decoder_layer_selection,
            freeze_model=decoder_freeze_model,
            device=device,
            enable_branch1=enable_branch1,
            enable_branch2=enable_branch2,
            enable_branch3=enable_branch3,
        )
        # GetdecoderThe output dimensions of
        feature_dim = self.qwen_decoder.hidden_size
        
        self._init_multi_task_heads(feature_dim)
        print(
            f"OnsetsAndFrames trainable params: "
            f"{count_trainable_parameters(self):,}"
        )
    
    def _init_feature_extractor(
        self,
        encoder_model="mert",
        layer_selection="1,5,10,15,20,23",
        freeze_all=False,
    ):
        """
        Initialize feature extractor
        
        Args:
            encoder_model: Feature source type, supported "MERT", "MERT_95m", "MERT_330m", "MuQ", "MusicFM"
            layer_selection: Layer selection parameters
            freeze_all: If True, freeze the entire encoder backbone (ICASSP encoder FT axis)
        """
        self.feature_extractor = MusicEncoder(
            encoder_model=encoder_model,
            use_last=False,
            freeze_all=freeze_all,
            device=self.device,
            layer_selection=layer_selection,
        )
    
    def _get_encoder_feature_dim(self, encoder_model):
        """Get encoder feature dimensions"""
        print(f"DEBUG: _get_encoder_feature_dim called with encoder_model={encoder_model}")
        if encoder_model in MusicEncoder.MODEL_ALIASES:
            encoder_model = MusicEncoder.MODEL_ALIASES[encoder_model]
        
        if encoder_model in MusicEncoder.MODEL_CONFIGS:
            return MusicEncoder.MODEL_CONFIGS[encoder_model]["feature_dim"]
        else:
            print(f"DEBUG: Unknown encoder_model {encoder_model}, defaulting to 768")
            return 768  # default768dimension
    
    def _init_multi_task_heads(self, feature_dim):
        """
        Initialize multi-task heads after the decoder stage.
        
        Args:
            feature_dim: decoderOutput feature dimensions
        """
        # Why: heads live in their own module; transcriber only wires them together.
        self.multi_task_heads: MultiTaskHeads = MLPBranchFactory.create_post_decoder_processor(
            decoder_output_dim=feature_dim, 
            output_dim=self.output_features,
            # Why: only hierarchical vs independent head modes remain for clean ablations.
            head_mode=self.head_mode,
        )

    def forward(self, audio):
        """
        Model forward propagation
        
        Args:
            audio: Raw audio input (required parameter) with shape [batch, time] or [time]
            
        Returns:
            tuple: (onset_pred, frame_pred, velocity_pred)
                - onset_pred: note onset prediction
                - frame_pred: Note frame prediction (independent branch output)
                - velocity_pred: Note velocity prediction
        """
        if audio is None:
            raise ValueError("Need to provideaudioparameter")
        
        # Step 1: Extract audio features
        if len(audio.shape) == 1:
            # if notbatchdimensions, addbatchDimensions
            audio = audio.unsqueeze(0)
        
        # Extract features using a feature extractor
        audio_features = self.feature_extractor(audio)
        
        # Qwenmodel:MERT->Qwenpipeline
        # Pass the extracted features as input todecoderfor further processing
        decoder_output = self.qwen_decoder(audio_features)
        
        # Extract fusion features (LLMDecoderreturn dictionary format)
        input_features = decoder_output['fused_feature']
        
        # Prediction using new post-processor (one line of code call)
        onset_logits, frame_logits, velocity_logits = self.multi_task_heads(input_features)
        
        # returnlogits, and then used when calculating lossessigmoidto improve numerical stability
        # Or apply it manually at inference timesigmoid
        return onset_logits, frame_logits, velocity_logits
