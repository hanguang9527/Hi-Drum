import json
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .constants import DEFAULT_LAYER_SELECTION, resolve_project_path
from .adapter import Adapter


def _debug_report(hypothesis_id, location, msg, data, run_id='pre-fix'):
    # Why: on GPU OOM during forward, dump runtime state to logs instead of guessing configs.
    env_path = '.dbg/qwen-train-oom.env'
    server_url = 'http://127.0.0.1:7777/event'
    session_id = 'qwen-train-oom'
    try:
        with open(env_path, 'r', encoding='utf-8') as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if line.startswith('DEBUG_SERVER_URL='):
                    server_url = line.split('=', 1)[1]
                elif line.startswith('DEBUG_SESSION_ID='):
                    session_id = line.split('=', 1)[1]
    except OSError:
        pass

    payload = {
        'sessionId': session_id,
        'runId': os.environ.get('TRAE_DEBUG_RUN_ID', run_id),
        'hypothesisId': hypothesis_id,
        'location': location,
        'msg': msg,
        'data': data,
        'ts': int(time.time() * 1000),
    }
    try:
        request = urllib.request.Request(
            server_url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
        )
        urllib.request.urlopen(request, timeout=0.2).read()
    except Exception:
        pass


class LLMDecoder(nn.Module):
    """
    UniversalLLMDecoder class, supports multipleLLMmodel asdecoderbranch
    include:GLM-0.5B, Qwen3-0.6B, Hunyuan-0.5B-Instruct, MiniCPM4-0.5B, Falcon-H1-0.5B-Base
    """
    
    # Model configuration mapping table
    MODEL_CONFIGS = {
        "GLM-0.5B": {
            "model_name": "THUDM/GLM-4-9B-Chat",  # What is actually used isGLM-0.5BVariants
            "local_path": "model/GLM-0.5B",
            "num_layers": 24,
            "hidden_size": 1024,
            "model_type": "glm"
        },
        "Qwen3-0.6B": {
            "model_name": "Qwen/Qwen3-0.6B",
            "local_path": "model/Qwen3-0.6B",
            "num_layers": 28,
            "hidden_size": 1024,
            "model_type": "qwen"
        },
        "Hunyuan-0.5B-Instruct": {
            "model_name": "Tencent-Hunyuan/HunyuanDenseV1-0.5B-Instruct",
            "local_path": "model/Hunyuan-0.5B-Instruct",
            "num_layers": 24,
            "hidden_size": 1024,
            "model_type": "hunyuan"
        },
        "MiniCPM4-0.5B": {
            "model_name": "openbmb/MiniCPM4-0.5B",
            "local_path": "model/MiniCPM4-0.5B",
            "num_layers": 24,
            "hidden_size": 1024,
            "model_type": "minicpm"
        },
        "Falcon-H1-0.5B-Base": {
            "model_name": "tiiuae/Falcon-H1-0.5B-Base",
            "local_path": "model/Falcon-H1-0.5B-Base",
            "num_layers": 36,
            "hidden_size": 1024,
            "model_type": "falcon"
        },
        "Qwen2.5-0.5B": {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "local_path": "model/Qwen2.5-0.5B-Instruct",
            "num_layers": 24,
            "hidden_size": 896,
            "model_type": "qwen"
        }
    }
    
    def __init__(self, 
                 decoder_model="Qwen2.5-0.5B",
                 encoder_model="95M",
                 feature_dim=768,
                 layer_selection="all",
                 freeze_model=True,
                 device='cpu',
                 enable_branch1=True,
                 enable_branch2=True,
                 enable_branch3=True):
        """
        initializationLLMdecoder
        
        Args:
            decoder_model: Decoder model name, supports:GLM-0.5B, Qwen3-0.6B, Hunyuan-0.5B-Instruct, 
                          MiniCPM4-0.5B, Falcon-H1-0.5B-Base, Qwen2.5-0.5B
            encoder_model: Encoder model name, supports:95M, 330M, MuQ, MusicFM
            feature_dim: Encoder feature dimension (768or1024)
            layer_selection: Layer selection parameters
            freeze_model: Whether to freeze model parameters
            device: equipment
            enable_branch1/2/3: Ablation experiment control switch
        """
        super().__init__()
        
        self.decoder_model = decoder_model
        self.encoder_model = encoder_model
        self.feature_dim = feature_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        self.freeze_model = freeze_model
        
        # Verify and get model configuration
        if decoder_model not in self.MODEL_CONFIGS:
            raise ValueError(f"Not supporteddecoderModel: {decoder_model}")
        
        self.model_config = self.MODEL_CONFIGS[decoder_model]
        self.model_name = self.model_config["model_name"]
        self.local_model_path = resolve_project_path(self.model_config["local_path"])
        self.num_layers = self.model_config["num_layers"]
        self.hidden_size = self.model_config["hidden_size"]
        
        print(f"LLMDecoder configuration: {decoder_model}")
        
        # initializationAdapter
        self.adapter = Adapter(
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            decoder_hidden_dim=self.hidden_size,
            enable_branch1=enable_branch1,
            enable_branch2=enable_branch2,
            enable_branch3=enable_branch3
        )
        print(f"  - Model name: {self.model_name}")
        print(f"  - local path: {self.local_model_path}")
        print(f"  - Number of layers: {self.num_layers}")
        print(f"  - Hidden layer dimensions: {self.hidden_size}")
        print(f"  - Encoder model: {encoder_model}")
        print(f"  - Encoder feature dimensions: {feature_dim}")
        
        # Analysis layer selection parameters
        self.selected_layers = self._parse_layer_selection(layer_selection)
        # Why: decoder layer_selection must stay decoupled from the encoder layer choice.
        
        # Load model
        self._load_model()
        
        # Make sure the adapter is on the correct device
        self.adapter = self.adapter.to(self.device)
        
        print(f"Universal adapter created: {self.adapter.get_info()}")
        
        # Freeze model parameters
        if self.freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"all{decoder_model}Parameters frozen")
        
        # Initialize layer weights (for multi-layer feature fusion)
        if len(self.selected_layers) > 1:
            self.layer_weights = nn.Parameter(torch.ones(len(self.selected_layers)))
            print(f"initialization{len(self.selected_layers)}Layer weight parameters are used for multi-layer feature fusion")
        else:
            self.layer_weights = None
            print(f"Use single layer features (section{self.selected_layers[0]}layer), no multi-layer fusion")
    
    def _load_model(self):
        """loadLLMModel"""
        try:
            if not os.path.exists(self.local_model_path):
                raise FileNotFoundError(f"The local model path does not exist: {self.local_model_path}")
            
            # Load the model andtokenizer
            self.model = AutoModel.from_pretrained(
                self.local_model_path, 
                trust_remote_code=True, 
                local_files_only=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_path, 
                trust_remote_code=True, 
                local_files_only=True
            )
            
            print(f"Successfully loaded model: {self.decoder_model}")
            
            # Make sure the model configuration is enabledhidden statesoutput
            self.model.config.output_hidden_states = True
            
            # Move the model to the specified device (GPUPrioritize, resolveGLM-0.5Bofhistogram_cpuquestion)
            if torch.cuda.is_available():
                print(f"Move the model toGPU: {self.device}")
                self.model = self.model.to(self.device)
            else:
                print(f"useCPUequipment: {self.device}")
                self.model = self.model.to(self.device)
            
        except FileNotFoundError as e:
            print(f"Local model file not found: {e}")
            raise
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise
    
    def _parse_layer_selection(self, layer_selection):
        """Analysis layer selection parameters"""
        if isinstance(layer_selection, str):
            if layer_selection.lower() == "all":
                return list(range(self.num_layers))
            else:
                # Parse comma separated layer numbers
                layers = []
                for layer_str in layer_selection.split(','):
                    layer = int(layer_str.strip())
                    if 0 <= layer < self.num_layers:
                        layers.append(layer)
                    else:
                        print(f"warn: Layer number{layer}out of range[0, {self.num_layers}), ignored")
                return layers if layers else [self.num_layers - 1]
        elif isinstance(layer_selection, (list, tuple)):
            return [int(l) for l in layer_selection if 0 <= int(l) < self.num_layers]
        else:
            return [self.num_layers - 1]  # Use last layer by default
    
    def forward(self, encoder_features, attention_mask=None):
        """
        forward propagation
        
        Args:
            encoder_features: Encoder features [batch_size, seq_len, feature_dim]
            attention_mask: attention mask [batch_size, seq_len]
            
        Returns:
            dict: Dictionary containing multi-layer features and fused features
        """
        batch_size, seq_len, _ = encoder_features.shape

        # #region debug-point A:decoder-input
        _debug_report(
            'A',
            'onsets_and_frames/llm_decoder.py:forward',
            '[DEBUG] Decoder forward input observed',
            {
                'decoder_model': self.decoder_model,
                'freeze_model': self.freeze_model,
                'grad_enabled': torch.is_grad_enabled(),
                'batch_size': int(batch_size),
                'seq_len': int(seq_len),
                'feature_dim': int(encoder_features.shape[-1]),
                'selected_layer_count': int(len(self.selected_layers)),
                'selected_layers_preview': self.selected_layers[:8],
                'cuda_allocated_mb': round(torch.cuda.memory_allocated(self.device) / 1024 / 1024, 2) if self.device.type == 'cuda' else None,
                'cuda_reserved_mb': round(torch.cuda.memory_reserved(self.device) / 1024 / 1024, 2) if self.device.type == 'cuda' else None,
            },
        )
        # #endregion
        
        # Make sure the input data is on the correct device
        encoder_features = encoder_features.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Convert feature dimensions via adapter
        adapted_features = self.adapter(encoder_features)  # [batch_size, seq_len, hidden_size]
        
        # Create virtual inputs (used to extract model internal features)
        # Here we use the adapted features as input and obtain the output of each layer through forward propagation of the model.
        
        # forMiniCPM4-0.5BWaiting for special models to prepare new onespast_key_valuesFormat
        model_kwargs = {
            'inputs_embeds': adapted_features,
            'attention_mask': attention_mask
        }
        
        # Check if you need to use a new oneCacheFormat (forMiniCPM4-0.5Band other models)
        if self.decoder_model == 'MiniCPM4-0.5B':
            try:
                from transformers.cache_utils import DynamicCache
                # use newCacheFormat
                model_kwargs['past_key_values'] = DynamicCache()
                model_kwargs['use_cache'] = True
            except ImportError:
                # ifCacheClass not available, use default format
                pass
        
        try:
            if hasattr(self.model, 'embeddings'):
                # For something likeBERTmodel
                outputs = self.model(**model_kwargs)
            else:
                # For other types of models, use features directly
                outputs = self.model(**model_kwargs)
        except RuntimeError as exc:
            # #region debug-point D:decoder-oom
            _debug_report(
                'D',
                'onsets_and_frames/llm_decoder.py:forward',
                '[DEBUG] Decoder forward raised runtime error',
                {
                    'decoder_model': self.decoder_model,
                    'freeze_model': self.freeze_model,
                    'grad_enabled': torch.is_grad_enabled(),
                    'seq_len': int(seq_len),
                    'selected_layer_count': int(len(self.selected_layers)),
                    'error': str(exc),
                    'cuda_allocated_mb': round(torch.cuda.memory_allocated(self.device) / 1024 / 1024, 2) if self.device.type == 'cuda' else None,
                    'cuda_reserved_mb': round(torch.cuda.memory_reserved(self.device) / 1024 / 1024, 2) if self.device.type == 'cuda' else None,
                    'cuda_max_allocated_mb': round(torch.cuda.max_memory_allocated(self.device) / 1024 / 1024, 2) if self.device.type == 'cuda' else None,
                },
            )
            # #endregion
            raise
        
        # Get all hidden layer states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
        else:
            # if nothidden_states, using the output of the last layer
            hidden_states = [adapted_features] * (self.num_layers + 1)

        # #region debug-point B:hidden-states
        _debug_report(
            'B',
            'onsets_and_frames/llm_decoder.py:forward',
            '[DEBUG] Decoder hidden states collected',
            {
                'hidden_state_count': int(len(hidden_states)),
                'selected_layer_count': int(len(self.selected_layers)),
                'returns_all_layers': bool(getattr(self.model.config, 'output_hidden_states', False)),
                'last_hidden_shape': list(hidden_states[-1].shape) if hidden_states else None,
                'cuda_allocated_mb': round(torch.cuda.memory_allocated(self.device) / 1024 / 1024, 2) if self.device.type == 'cuda' else None,
                'cuda_reserved_mb': round(torch.cuda.memory_reserved(self.device) / 1024 / 1024, 2) if self.device.type == 'cuda' else None,
            },
        )
        # #endregion
        
        # Extract features of selected layer
        selected_features = []
        for layer_idx in self.selected_layers:
            if layer_idx < len(hidden_states):
                selected_features.append(hidden_states[layer_idx])
            else:
                selected_features.append(hidden_states[-1])
        
        # Multi-layer feature fusion
        if len(selected_features) > 1 and self.layer_weights is not None:
            # Perform on layer weightssoftmaxnormalization
            weights = F.softmax(self.layer_weights, dim=0)
            
            # Weighted fusion of multi-layer features
            fused_feature = torch.zeros_like(selected_features[0])
            for i, feature in enumerate(selected_features):
                fused_feature += weights[i] * feature
        else:
            # Single layer features, used directly
            fused_feature = selected_features[0] if selected_features else adapted_features
        
        return {
            'selected_layer_features': selected_features,  # Feature list of each layer
            'fused_feature': fused_feature,                # Features after fusion
            'last_hidden_state': hidden_states[-1] if hidden_states else adapted_features,  # last layer features
            'decoder_type': self.decoder_model,          # Decoder type
            'layer_weights': self.layer_weights          # Layer weights (for visualization or analysis)
        }
    
    def get_model_info(self):
        """Get model information"""
        return {
            'decoder_model': self.decoder_model,
            'model_name': self.model_name,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'selected_layers': self.selected_layers,
            'encoder_model': self.encoder_model,
            'feature_dim': self.feature_dim,
            'device': str(self.device),
            'freeze_model': self.freeze_model
        }
    
    @classmethod
    def get_supported_models(cls):
        """Get a list of supported models"""
        return list(cls.MODEL_CONFIGS.keys())
    
    @classmethod
    def get_model_config(cls, model_name):
        """Get the configuration of the specified model"""
        return cls.MODEL_CONFIGS.get(model_name, None)
