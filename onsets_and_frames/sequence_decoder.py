"""Sequence decoder backends for ICASSP B1 ablations.

decoder_backend:
  - llm: existing frozen/unfrozen pretrained LLM path (LLMDecoder)
  - scratch_transformer: randomly-initialized Transformer matched to Qwen3-0.6B width/depth
  - none: adapter only (encoder + adapter + heads), no sequence decoder body
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import Adapter
from .llm_decoder import LLMDecoder


# Match Qwen3-0.6B body geometry (see model/Qwen3-0.6B/config.json).
QWEN3_06B_HIDDEN = 1024
QWEN3_06B_LAYERS = 28
QWEN3_06B_HEADS = 16
QWEN3_06B_FFN = 3072


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_all_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class ScratchTransformerDecoder(nn.Module):
    """
    From-scratch Transformer decoder body with the same Adapter front-end as LLMDecoder.

    Layer geometry defaults mirror Qwen3-0.6B (28 x 1024, 16 heads, FFN 3072).
    Layer outputs are fused with learnable softmax weights (same interface as LLMDecoder).
    """

    def __init__(
        self,
        encoder_model="MERT_330m",
        decoder_model="Qwen3-0.6B",
        feature_dim=1024,
        layer_selection="all",
        device="cpu",
        enable_branch1=True,
        enable_branch2=True,
        enable_branch3=True,
        num_layers=QWEN3_06B_LAYERS,
        hidden_size=QWEN3_06B_HIDDEN,
        nhead=QWEN3_06B_HEADS,
        dim_feedforward=QWEN3_06B_FFN,
        dropout=0.1,
    ):
        super().__init__()
        self.decoder_backend = "scratch_transformer"
        self.decoder_model = decoder_model
        self.encoder_model = encoder_model
        self.feature_dim = feature_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.freeze_model = False

        self.adapter = Adapter(
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            decoder_hidden_dim=hidden_size,
            enable_branch1=enable_branch1,
            enable_branch2=enable_branch2,
            enable_branch3=enable_branch3,
        )

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

        self.selected_layers = self._parse_layer_selection(layer_selection)
        if len(self.selected_layers) > 1:
            self.layer_weights = nn.Parameter(torch.ones(len(self.selected_layers)))
        else:
            self.layer_weights = None

        self.adapter = self.adapter.to(self.device)
        self.to(self.device)

        print(
            f"ScratchTransformerDecoder: layers={num_layers}, hidden={hidden_size}, "
            f"nhead={nhead}, ffn={dim_feedforward}, n_selected={len(self.selected_layers)}"
        )
        print(f"  Adapter: {self.adapter.get_info()}")
        print(
            f"  Params: total={count_all_parameters(self):,}, "
            f"trainable={count_trainable_parameters(self):,}"
        )

    def _parse_layer_selection(self, layer_selection):
        if isinstance(layer_selection, str):
            if layer_selection.lower() == "all":
                return list(range(self.num_layers))
            layers = []
            for layer_str in layer_selection.split(","):
                layer = int(layer_str.strip())
                if 0 <= layer < self.num_layers:
                    layers.append(layer)
            return layers if layers else [self.num_layers - 1]
        if isinstance(layer_selection, (list, tuple)):
            return [int(l) for l in layer_selection if 0 <= int(l) < self.num_layers]
        return [self.num_layers - 1]

    def forward(self, encoder_features, attention_mask=None):
        encoder_features = encoder_features.to(self.device)
        x = self.adapter(encoder_features)

        key_padding_mask = None
        if attention_mask is not None:
            # HF mask: 1=keep, 0=pad → PyTorch key_padding_mask: True=pad
            key_padding_mask = ~(attention_mask.to(self.device).bool())

        hidden_states = []
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
            hidden_states.append(x)
        x = self.final_norm(x)
        # Align with LLMDecoder indexing: use post-layer states; final norm on fused result below.

        selected_features = []
        for layer_idx in self.selected_layers:
            if layer_idx < len(hidden_states):
                selected_features.append(hidden_states[layer_idx])
            else:
                selected_features.append(hidden_states[-1])

        if len(selected_features) > 1 and self.layer_weights is not None:
            weights = F.softmax(self.layer_weights, dim=0)
            fused = torch.zeros_like(selected_features[0])
            for i, feature in enumerate(selected_features):
                fused = fused + weights[i] * feature
            fused = self.final_norm(fused)
        else:
            fused = self.final_norm(selected_features[0] if selected_features else x)

        return {
            "selected_layer_features": selected_features,
            "fused_feature": fused,
            "last_hidden_state": hidden_states[-1] if hidden_states else fused,
            "decoder_type": "scratch_transformer",
            "layer_weights": self.layer_weights,
        }

    def get_model_info(self):
        return {
            "decoder_backend": self.decoder_backend,
            "decoder_model": self.decoder_model,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "selected_layers": self.selected_layers,
            "encoder_model": self.encoder_model,
            "feature_dim": self.feature_dim,
            "device": str(self.device),
            "freeze_model": self.freeze_model,
            "trainable_params": count_trainable_parameters(self),
            "total_params": count_all_parameters(self),
        }


class PassthroughDecoder(nn.Module):
    """Adapter-only path: no LLM / no scratch Transformer body."""

    def __init__(
        self,
        encoder_model="MERT_330m",
        decoder_model="Qwen3-0.6B",
        feature_dim=1024,
        device="cpu",
        enable_branch1=True,
        enable_branch2=True,
        enable_branch3=True,
        hidden_size=QWEN3_06B_HIDDEN,
    ):
        super().__init__()
        self.decoder_backend = "none"
        self.decoder_model = decoder_model
        self.encoder_model = encoder_model
        self.feature_dim = feature_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        self.hidden_size = hidden_size
        self.freeze_model = True
        self.num_layers = 0
        self.layer_weights = None
        self.selected_layers = []

        self.adapter = Adapter(
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            decoder_hidden_dim=hidden_size,
            enable_branch1=enable_branch1,
            enable_branch2=enable_branch2,
            enable_branch3=enable_branch3,
        ).to(self.device)

        print(f"PassthroughDecoder (none): adapter-only → heads")
        print(f"  Adapter: {self.adapter.get_info()}")
        print(
            f"  Params: total={count_all_parameters(self):,}, "
            f"trainable={count_trainable_parameters(self):,}"
        )

    def forward(self, encoder_features, attention_mask=None):
        encoder_features = encoder_features.to(self.device)
        adapted = self.adapter(encoder_features)
        return {
            "selected_layer_features": [adapted],
            "fused_feature": adapted,
            "last_hidden_state": adapted,
            "decoder_type": "none",
            "layer_weights": None,
        }

    def get_model_info(self):
        return {
            "decoder_backend": self.decoder_backend,
            "decoder_model": self.decoder_model,
            "num_layers": 0,
            "hidden_size": self.hidden_size,
            "selected_layers": [],
            "encoder_model": self.encoder_model,
            "feature_dim": self.feature_dim,
            "device": str(self.device),
            "freeze_model": self.freeze_model,
            "trainable_params": count_trainable_parameters(self),
            "total_params": count_all_parameters(self),
        }


def build_sequence_decoder(
    decoder_backend="llm",
    decoder_model="Qwen3-0.6B",
    encoder_model="MERT_330m",
    feature_dim=1024,
    layer_selection="all",
    freeze_model=True,
    device="cpu",
    enable_branch1=True,
    enable_branch2=True,
    enable_branch3=True,
):
    """
    Factory used by OnsetsAndFrames.

    Returns a module with:
      - .hidden_size
      - .adapter
      - .forward(...) -> dict containing fused_feature
      - .get_model_info()
      - .decoder_backend
    """
    backend = (decoder_backend or "llm").lower().strip()
    if backend in ("llm", "qwen", "pretrained"):
        decoder = LLMDecoder(
            decoder_model=decoder_model,
            encoder_model=encoder_model,
            feature_dim=feature_dim,
            layer_selection=layer_selection,
            freeze_model=freeze_model,
            device=device,
            enable_branch1=enable_branch1,
            enable_branch2=enable_branch2,
            enable_branch3=enable_branch3,
        )
        decoder.decoder_backend = "llm"
        return decoder

    if backend in ("scratch_transformer", "scratch", "transformer"):
        # Geometry tied to decoder_model config when available.
        hidden = QWEN3_06B_HIDDEN
        layers = QWEN3_06B_LAYERS
        if decoder_model in LLMDecoder.MODEL_CONFIGS:
            cfg = LLMDecoder.MODEL_CONFIGS[decoder_model]
            hidden = cfg["hidden_size"]
            layers = cfg["num_layers"]
        return ScratchTransformerDecoder(
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            feature_dim=feature_dim,
            layer_selection=layer_selection,
            device=device,
            enable_branch1=enable_branch1,
            enable_branch2=enable_branch2,
            enable_branch3=enable_branch3,
            num_layers=layers,
            hidden_size=hidden,
            nhead=QWEN3_06B_HEADS if hidden == QWEN3_06B_HIDDEN else max(1, hidden // 64),
            dim_feedforward=QWEN3_06B_FFN if hidden == QWEN3_06B_HIDDEN else hidden * 3,
        )

    if backend in ("none", "passthrough", "adapter_only"):
        hidden = QWEN3_06B_HIDDEN
        if decoder_model in LLMDecoder.MODEL_CONFIGS:
            hidden = LLMDecoder.MODEL_CONFIGS[decoder_model]["hidden_size"]
        return PassthroughDecoder(
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            feature_dim=feature_dim,
            device=device,
            enable_branch1=enable_branch1,
            enable_branch2=enable_branch2,
            enable_branch3=enable_branch3,
            hidden_size=hidden,
        )

    raise ValueError(
        f"Unsupported decoder_backend={decoder_backend!r}. "
        f"Use one of: llm | scratch_transformer | none"
    )
