import os
import sys
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, Wav2Vec2FeatureExtractor

try:
    from .constants import DEFAULT_LAYER_SELECTION, resolve_project_path
except ImportError:
    DEFAULT_LAYER_SELECTION = "7"

    def resolve_project_path(relative_path):
        return relative_path

class MusicEncoder(nn.Module):
    """
    Unified Audio Encoder Frontend, supporting MERT-95M, MERT-330M, MuQ, and MusicFM.
    """

    MODEL_CONFIGS = {
        "MERT_95m": {
            "url": "m-a-p/MERT-v1-95M",
            "local_path": "model/MERT-v1-95M",
            "num_layers": 13,
            "feature_dim": 768,
            "type": "mert"
        },
        "MERT_330m": {
            "url": "m-a-p/MERT-v1-330M",
            "local_path": "model/MERT-v1-330M",
            "num_layers": 25,
            "feature_dim": 1024,
            "type": "mert"
        },
        # 24 blocks. hidden_states[0] is block 1, [23] is block 24. 25 Hz, 24 kHz, dim 1024.
        "MERT2": {
            "url": "m-a-p/MERT-v2-30s",
            "local_path": "model/MERT-v2-30s",
            "num_layers": 24,
            "feature_dim": 1024,
            "type": "mert"
        },
        "MuQ": {
            "url": "model/MuQ-large-msd-iter",
            "local_path": "model/MuQ-large-msd-iter",
            "num_layers": -1,  # Dynamic, will be detected
            "feature_dim": 1024,
            "type": "muq"
        },
        "MusicFM": {
            "home_path": "model/MusicFM_MSD",
            "stat_path": "model/MusicFM_MSD/musicfm/data/msd_stats.json",
            "local_path": "model/MusicFM_MSD/musicfm/data/pretrained_msd.pt",
            "num_layers": 13,
            "feature_dim": 1024,
            "type": "musicfm"
        }
    }

    # Alias mappings for compatibility
    MODEL_ALIASES = {
        "MERT": "MERT_95m",
        "MERT-95M": "MERT_95m",
        "MERT-330M": "MERT_330m",
        "MERT-v2-30s": "MERT2",
        "MERT_v2": "MERT2",
        "mert2": "MERT2",
    }

    def __init__(self, encoder_model="MERT_95m", use_last=False, freeze_all=False, 
                 device="cpu", layer_selection=DEFAULT_LAYER_SELECTION, 
                 freeze_feature_extractor=False, freeze_backbone=False, is_flash=False):
        super().__init__()
        
        # Resolve aliases
        if encoder_model in self.MODEL_ALIASES:
            encoder_model = self.MODEL_ALIASES[encoder_model]
            
        self.encoder_model = encoder_model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_last = use_last
        self.layer_selection_param = layer_selection

        if encoder_model not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported encoder model: {encoder_model}. Supported models: {list(self.MODEL_CONFIGS.keys())}")

        self.config = dict(self.MODEL_CONFIGS[encoder_model])
        for path_key in ("local_path", "home_path", "stat_path"):
            if path_key in self.config:
                self.config[path_key] = resolve_project_path(self.config[path_key])
        self.model_type = self.config["type"]
        self.num_layers = self.config["num_layers"]
        self.feature_dim = self.config["feature_dim"]

        # Load specific model
        self._load_model(is_flash)

        # Parse layer selection
        self.selected_layers = self._parse_layer_selection(layer_selection)

        # Apply freeze policy
        self._apply_freeze(freeze_all, freeze_feature_extractor, freeze_backbone)

        # Initialize multi-layer fusion weights
        if not use_last and len(self.selected_layers) > 1:
            self.layer_weights = nn.Parameter(
                data=torch.ones(len(self.selected_layers), device=self.device), requires_grad=True
            )
            print(f"Initialized {len(self.selected_layers)} layer weights for {encoder_model}. Layers: {self.selected_layers}")
        else:
            self.layer_weights = None
            if use_last:
                print(f"{encoder_model}: Using the last layer features without fusion.")
            elif len(self.selected_layers) == 1:
                print(f"{encoder_model}: Using single layer ({self.selected_layers[0]}) features without fusion.")

        # GPU optimizations (for MERT models)
        if self.model_type == "mert":
            self._init_gpu_optimizations()

    def _load_model(self, is_flash=False):
        if self.model_type == "mert":
            try:
                self.model = AutoModel.from_pretrained(self.config["local_path"], trust_remote_code=True)
                print(f"Loaded {self.encoder_model} from local path: {self.config['local_path']}")
            except Exception:
                self.model = AutoModel.from_pretrained(self.config["url"], trust_remote_code=True)
                print(f"Loaded {self.encoder_model} from HuggingFace Hub: {self.config['url']}")
            self.model = self.model.to(self.device)

        elif self.model_type == "muq":
            from muq import MuQ
            if not os.path.exists(self.config["local_path"]):
                raise FileNotFoundError(f"Local model path not found: {self.config['local_path']}")
            self.model = MuQ.from_pretrained(self.config["local_path"])
            self.model.to(self.device)
            # Dynamically detect num_layers for MuQ
            test_audio = torch.randn(1, 16000).to(self.device)
            with torch.no_grad():
                outputs = self.model(test_audio, output_hidden_states=True)
                self.num_layers = len(outputs.hidden_states)
            print(f"MuQ model actual num_layers: {self.num_layers}")

        elif self.model_type == "musicfm":
            if self.config["home_path"] not in sys.path:
                sys.path.append(self.config["home_path"])
            from musicfm.model.musicfm_25hz import MusicFM25Hz
            self.model = MusicFM25Hz(
                is_flash=is_flash,
                stat_path=self.config["stat_path"],
                model_path=self.config["local_path"],
            )
            self.model = self.model.to(self.device)

    def _apply_freeze(self, freeze_all, freeze_feature_extractor, freeze_backbone):
        if freeze_all:
            for p in self.model.parameters():
                p.requires_grad = False
            print(f"All {self.encoder_model} parameters frozen.")
            if self.model_type == "muq":
                self.model.eval()
            return

        if self.model_type == "mert":
            feature_extractor = getattr(self.model, "feature_extractor", None)
            if feature_extractor is None:
                print(f"Warning: feature_extractor not found for {self.encoder_model}.")
            elif hasattr(feature_extractor, "_freeze_parameters"):
                feature_extractor._freeze_parameters()
                print(f"{self.encoder_model} feature extractor frozen, Transformer layers trainable.")
            else:
                for param in feature_extractor.parameters():
                    param.requires_grad = False
                print(f"{self.encoder_model} feature extractor frozen, Transformer layers trainable.")
        
        elif self.model_type == "musicfm":
            if freeze_feature_extractor:
                if hasattr(self.model, "conv"):
                    for p in self.model.conv.parameters(): p.requires_grad = False
                if hasattr(self.model, "preprocessor_melspec_2048"):
                    for p in getattr(self.model, "preprocessor_melspec_2048").parameters(): p.requires_grad = False
                try:
                    for feature in getattr(self.model, "features", []):
                        for i in range(getattr(self.model, "num_codebooks", 0)):
                            name = f"quantizer_{feature}_{i}"
                            if hasattr(self.model, name):
                                for p in getattr(self.model, name).parameters(): p.requires_grad = False
                except Exception: pass
                print("MusicFM feature extractor parameters frozen.")

            if freeze_backbone:
                if hasattr(self.model, "conformer"):
                    for p in self.model.conformer.parameters(): p.requires_grad = False
                if hasattr(self.model, "linear"):
                    for p in self.model.linear.parameters(): p.requires_grad = False
                if hasattr(self.model, "cls_token") and isinstance(self.model.cls_token, nn.Parameter):
                    self.model.cls_token.requires_grad = False
                print("MusicFM backbone parameters frozen.")

    def _init_gpu_optimizations(self):
        if self.device.type == 'cuda':
            self.register_buffer('_zero_tensor', torch.tensor(0.0, device=self.device))
            self.register_buffer('_one_tensor', torch.tensor(1.0, device=self.device))
            if self.encoder_model == "MERT2":
                min_samples = int(getattr(self.model.config, "minimum_input_samples", 24000))
                dummy_len = max(24000, min_samples)
            else:
                dummy_len = 1000
            dummy_input = torch.randn(1, dummy_len, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input, output_hidden_states=True)
            print(f"GPU optimizations initialized for {self.encoder_model} on {self.device}")

    def _parse_layer_selection(self, layer_selection):
        if isinstance(layer_selection, str):
            if layer_selection.lower() == "all":
                return list(range(self.num_layers))
            try:
                layer_nums = [int(x.strip()) for x in layer_selection.split(',') if x.strip()]
                for layer_num in layer_nums:
                    if layer_num < 0 or layer_num >= self.num_layers:
                        raise ValueError(f"Layer index {layer_num} out of range (0-{self.num_layers-1})")
                return sorted(list(set(layer_nums)))
            except ValueError as e:
                raise ValueError(f"Invalid layer selection format: {layer_selection}. Error: {e}")
        elif isinstance(layer_selection, (int, list)):
            layer_nums = [layer_selection] if isinstance(layer_selection, int) else layer_selection
            for layer_num in layer_nums:
                if not isinstance(layer_num, int) or layer_num < 0 or layer_num >= self.num_layers:
                    raise ValueError(f"Layer index {layer_num} out of range (0-{self.num_layers-1})")
            return sorted(list(set(layer_nums)))
        else:
            raise ValueError(f"Unsupported layer selection parameter type: {type(layer_selection)}")

    @staticmethod
    def create_processor(encoder_model, sampling_rate=24000):
        # Resolve aliases
        if encoder_model in MusicEncoder.MODEL_ALIASES:
            encoder_model = MusicEncoder.MODEL_ALIASES[encoder_model]
            
        if encoder_model in ["MERT_95m", "MERT_330m", "MERT", "MERT2"]:
            return Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=sampling_rate,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=False,
            )
        elif encoder_model == "MuQ":
            from muq import MuQ
            return MuQ.create_muq_processor(sampling_rate=sampling_rate)
        elif encoder_model == "MusicFM":
            return None
        else:
            raise ValueError(f"Unknown encoder_model: {encoder_model}")

    def preprocess_audio_gpu(self, audio_waveform, sampling_rate=24000):
        if self.model_type == "musicfm":
            return audio_waveform

        with torch.no_grad():
            if isinstance(audio_waveform, np.ndarray):
                audio_tensor = torch.from_numpy(audio_waveform).to(self.device, dtype=torch.float32)
                if audio_waveform.dtype == np.int16 and hasattr(self, 'sequence_length'):
                    audio_tensor.div_(self.sequence_length / 10)
            else:
                audio_tensor = audio_waveform.to(self.device) if audio_waveform.device != self.device else audio_waveform
                if audio_tensor.dtype == torch.int16:
                    audio_tensor = audio_tensor.float()
                    if hasattr(self, 'sequence_length'): audio_tensor.div_(self.sequence_length / 10)
                elif audio_tensor.dtype != torch.float32:
                    audio_tensor = audio_tensor.float()

            audio_tensor.clamp_(-1.0, 1.0)

            if audio_tensor.numel() > 1:
                if audio_tensor.dim() > 1:
                    mean = audio_tensor.mean(dim=-1, keepdim=True)
                    std = audio_tensor.std(dim=-1, keepdim=True, unbiased=False)
                else:
                    mean = audio_tensor.mean()
                    std = audio_tensor.std(unbiased=False)

                std_threshold = 1e-8
                if isinstance(std, torch.Tensor):
                    valid_std = std > std_threshold
                    if valid_std.any():
                        if audio_tensor.dim() > 1:
                            audio_tensor = torch.where(
                                valid_std,
                                (audio_tensor - mean) / torch.clamp(std, min=std_threshold),
                                audio_tensor
                            )
                        else:
                            audio_tensor.sub_(mean).div_(std)
                elif std > std_threshold:
                    audio_tensor.sub_(mean).div_(std)

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            return audio_tensor

    def preprocess_audio_batch_gpu(self, audio_batch, sampling_rate=24000):
        if self.model_type == "musicfm":
            return audio_batch
            
        if not isinstance(audio_batch, (list, tuple)):
            return self.preprocess_audio_gpu(audio_batch, sampling_rate)
            
        with torch.no_grad():
            processed_audios = []
            max_length = 0
            for audio in audio_batch:
                processed = self.preprocess_audio_gpu(audio, sampling_rate)
                if processed.dim() > 1:
                    processed = processed.squeeze(0)
                processed_audios.append(processed)
                max_length = max(max_length, processed.size(0))
                
            padded_audios = []
            for audio in processed_audios:
                if audio.size(0) < max_length:
                    padding = torch.zeros(max_length - audio.size(0), device=self.device, dtype=audio.dtype)
                    audio = torch.cat([audio, padding], dim=0)
                padded_audios.append(audio)
            return torch.stack(padded_audios, dim=0)

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)

        if self.encoder_model == "MERT2":
            # Dataset stores int16 / (clip_length / 10). MERT2 wants raw PCM in [-1, 1].
            x = x * (x.shape[-1] / 10.0) / 32768.0

        model_trainable = any(p.requires_grad for p in self.model.parameters())
        # Why: if the encoder is frozen, stop grads on the backbone forward but keep layer_weights trainable.
        model_forward_context = nullcontext() if model_trainable else torch.no_grad()

        if self.model_type == "musicfm":
            with model_forward_context:
                if self.use_last:
                    return self.model.get_latent(x, layer_ix=self.num_layers - 1)

                if len(self.selected_layers) == 1:
                    selected_h = self.model.get_latent(x, layer_ix=self.selected_layers[0])
                else:
                    selected_h = [self.model.get_latent(x, layer_ix=li) for li in self.selected_layers]

            if len(self.selected_layers) == 1:
                return F.pad(selected_h, (0, 0, 0, 1), mode="reflect")

            h = torch.stack(selected_h, dim=3)
            h = F.pad(h, (0, 0, 0, 0, 0, 1), mode="reflect")
            weights = torch.softmax(self.layer_weights, dim=0)
            return torch.matmul(h, weights)

        with model_forward_context:
            outputs = self.model(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs["hidden_states"]

            if self.use_last:
                return hidden_states[-1]

            if len(self.selected_layers) == 1:
                selected_h = hidden_states[self.selected_layers[0]]
            else:
                selected_h = [hidden_states[i] for i in self.selected_layers]

        if len(self.selected_layers) == 1:
            if self.model_type == "mert":
                selected_h = F.pad(selected_h, (0, 0, 0, 1), mode='reflect')
            return selected_h

        h_stack = torch.stack(selected_h, dim=3)
        if self.model_type == "mert":
            h_stack = F.pad(h_stack, (0, 0, 0, 0, 0, 1), mode='reflect')
        weights = torch.softmax(self.layer_weights, dim=0)
        return torch.matmul(h_stack, weights)

    def get_feature_extractor_output(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        with torch.no_grad():
            if self.model_type == "musicfm":
                return self.model.get_latent(x, layer_ix=0)
            else:
                outputs = self.model(x, output_hidden_states=True)
                h = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs["hidden_states"]
                return h[0]

    def get_layer_weights(self):
        if self.layer_weights is not None:
            if self.model_type == "muq":
                return torch.softmax(self.layer_weights, dim=0).detach()
            return torch.softmax(self.layer_weights, dim=0)
        return None
