"""
adapter module - Three parallel branch architectures

Include:
1. Fully connected branch (original functionality)
2. Inflated 1D convolution branch (multi-scale temporal modeling)
3. TransformerBranching (long-distance dependency modeling)
4. Learnable weight fusion mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class DilatedConv1DBlock(nn.Module):
    """Dilated 1D convolution block - branch2core components of"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 51, 
                 dilations: list = None, dropout: float = 0.1):
        super().__init__()
        
        if dilations is None:
            dilations = [2, 3, 4, 5, 6]
        
        self.dilations = dilations
        self.out_channels = out_channels
        
        # Create separate convolutional layers for each dilation rate
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=self._get_padding(kernel_size, dilation)  # Keep time dimension
                ),
                # Why: 空洞卷积分支感受野大、不同 dilation 的统计分布差异明显，先做归一化更利于稳定训练。
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ) for dilation in dilations
        ])
        
        # Output projection layer - Make sure dimensions match
        # Modification: The input dimension becomes out_channels * len(dilations), since we now use concatenation instead of summation
        self.concat_norm = nn.BatchNorm1d(out_channels * len(dilations))
        self.output_projection = nn.Linear(out_channels * len(dilations), out_channels)
        # Why: 分支输出在融合前先做尺度对齐，避免仅因数值幅度不同导致融合权重学偏。
        self.output_norm = nn.LayerNorm(out_channels)
        # Why: 这里加轻量 shortcut 不是因为网络太深，而是为了保留原始时序信息，降低大核空洞卷积对输入分布的破坏。
        self.residual_projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def _get_padding(self, kernel_size: int, dilation: int) -> int:
        """Calculate keeping the output length constantpadding"""
        return (kernel_size - 1) * dilation // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward propagation
        Args:
            x: input tensor [batch, channels, time]
        Returns:
            output tensor [batch, channels, time]
        """
        # Store the results of all dilated convolutions
        conv_outputs = []
        
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)  # [batch, out_channels, time]
            conv_outputs.append(conv_out)
        
        # Modification: use splicing (Concat) summation of substitutions (Sum)
        # [batch, out_channels * num_dilations, time]
        combined = torch.cat(conv_outputs, dim=1)
        
        combined = self.concat_norm(combined)
        
        # Transpose back [batch, time, channels] Perform linear projection
        combined = combined.transpose(1, 2)  # [batch, time, out_channels * num_dilations]
        combined = self.output_projection(combined)  # [batch, time, out_channels]
        
        if self.dropout is not None:
            combined = self.dropout(combined)

        residual = self.residual_projection(x).transpose(1, 2)  # [batch, time, out_channels]
        combined = self.output_norm(combined + residual)
        
        # transfer back [batch, channels, time]
        return combined.transpose(1, 2)


class PreNorm(nn.Module):
    """Pre-normalization module"""
    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MultiHeadAttention(nn.Module):
    """multi-head attention mechanism"""
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    """feedforward network"""
    
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformerpiece - branch3core components of"""
    
    def __init__(self, input_dim, output_dim, depth=2, heads=8, dim_head=64, mlp_dim=512, dropout=0.1):
        super().__init__()
        # input projection layer
        self.input_projection = nn.Linear(input_dim, output_dim)
        
        self.input_norm = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(output_dim, MultiHeadAttention(output_dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(output_dim, FeedForward(output_dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        # Project to the target dimension first
        x = self.input_projection(x)  # [batch, time, output_dim]
        # Why: 先对投影后的特征做标准化，便于与另外两个卷积分支在融合前保持更接近的输出尺度。
        x = self.input_norm(x)
        
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ParallelConvBranch(nn.Module):
    """
    branch1: Parallel one-dimensional convolution branch
    usekernel_size=3and5Two parallel one-dimensional convolutions to simulateonset/frameTime tolerance in detection.
    - kernel_size=3 (receptive field~40ms): Capture features at precise points in time.
    - kernel_size=5 (receptive field~67ms): Captures features within a slightly wider tolerance window.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        # convolution branch1: kernel_size=3
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1  # (3-1)//2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        # convolution branch2: kernel_size=5
        self.conv2 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=5,
            padding=2  # (5-1)//2 is 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        # Result fused projection layer
        self.projection = nn.Linear(out_channels * 2, out_channels)
        # Why: 融合后的输出在进入总融合前先做标准化，减少不同分支数值尺度不一致的问题。
        self.output_norm = nn.LayerNorm(out_channels)
        # Why: 这里保留一条轻量残差通路，让浅层卷积分支在学习局部模式的同时不丢掉原始时间结构。
        self.residual_projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor [batch, channels, time]
        Returns:
            output tensor [batch, channels, time]
        """
        # Pass two convolutional layers in parallel
        out1 = self.activation(self.bn1(self.conv1(x)))
        out2 = self.activation(self.bn2(self.conv2(x)))

        # Splicing along channel dimensions
        combined = torch.cat([out1, out2], dim=1) # [batch, out_channels * 2, time]

        # Transpose to fit linear layer
        combined = combined.transpose(1, 2) # [batch, time, out_channels * 2]

        # Project back to target dimension
        projected = self.projection(combined) # [batch, time, out_channels]

        if self.dropout is not None:
            projected = self.dropout(projected)

        residual = self.residual_projection(x).transpose(1, 2) # [batch, time, out_channels]
        projected = self.output_norm(projected + residual)

        # Transpose back to convolution format
        projected = projected.transpose(1, 2)

        return projected


class Adapter(nn.Module):
    """adapter - Three parallel branch architectures"""
    
    # The dimension mapping table remains unchanged
    ENCODER_OUTPUT_DIMS = {
        "MERT_95m": 768,
        "MERT_330m": 1024,
        "MERT2": 1024,
        "MERT-v2-30s": 1024,
        "MERT": 768,
        "MuQ": 1024,
        "MusicFM": 1024,
    }
    
    DECODER_INPUT_DIMS = {
        "Qwen2.5-0.5B": 896,
        "GLM-0.5B": 1024,
        "Qwen3-0.6B": 1024,
        "Hunyuan-0.5B-Instruct": 1024,
        "MiniCPM4-0.5B": 1024,
        "Falcon-H1-0.5B-Base": 1024,
    }
    
    def __init__(self, 
                 encoder_model: str = "MERT_330m",
                 decoder_model: str = "Qwen/Qwen2.5-0.5B-Instruct", 
                 decoder_hidden_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True,
                 activation: str = "relu",
                 # New parameters
                 branch1_ratio: float = 0.33,  # Parallel convolution branch initial weight
                 branch2_ratio: float = 0.33,  # Dilated convolution branch initial weight  
                 branch3_ratio: float = 0.34,  # TransformerBranch initial weight
                 transformer_depth: int = 2,    # TransformerNumber of layers
                 transformer_heads: int = 8,    # TransformerNumber of heads
                 dilated_conv_kernel: int = 51, # Dilated convolution kernel size
                 dilated_conv_dilations: list = None, # Expansion rate
                 # Ablation experiment control switch
                 enable_branch1: bool = True,  # Whether to enable parallel convolution branches
                 enable_branch2: bool = True,  # Whether to enable dilated convolution branch
                 enable_branch3: bool = True,  # Whether to enableTransformerbranch
                 ):
        super().__init__()
        
        self.input_dim = self.ENCODER_OUTPUT_DIMS.get(encoder_model, 1024)
        self.output_dim = decoder_hidden_dim or self.DECODER_INPUT_DIMS.get(decoder_model, 1024)
        self.dropout = dropout
        
        self.enable_branch1 = enable_branch1
        self.enable_branch2 = enable_branch2
        self.enable_branch3 = enable_branch3
        
        # 1. Build three major branches
        if self.enable_branch1:
            self.branch1_parallel_conv = ParallelConvBranch(
                in_channels=self.input_dim,
                out_channels=self.output_dim,
                dropout=self.dropout
            )
        else:
            self.branch1_parallel_conv = None
            
        if self.enable_branch2:
            self.branch2_dilated_conv = self._build_branch2_dilated_conv(dilated_conv_kernel, dilated_conv_dilations)
        else:
            self.branch2_dilated_conv = None
            
        if self.enable_branch3:
            self.branch3_transformer = self._build_branch3_transformer(transformer_depth, transformer_heads)
        else:
            self.branch3_transformer = None
        
        # 2. Build learnable fusion weights
        active_ratios = []
        if self.enable_branch1: active_ratios.append(branch1_ratio)
        if self.enable_branch2: active_ratios.append(branch2_ratio)
        if self.enable_branch3: active_ratios.append(branch3_ratio)
        
        if len(active_ratios) > 0:
            self.fusion_weights = nn.Parameter(torch.tensor(active_ratios))
        else:
            self.fusion_weights = None
        
        # 3. Construct final layer normalization and activation functions
        # If it is a direct connection(0branch), and no projection is required, these layers may also need to be skipped?
        # Usually direct connections should not go throughLayerNorm/Activation, unless it is to adaptDecoderEnter the distribution.
        # Considering the semantics of "direct connection", it usually meansIdentity.
        # but keepLNMay help train stability. Here we keep it as is, if there is output then passLN.
        self.final_layer_norm = nn.LayerNorm(self.output_dim) if use_layer_norm else nn.Identity()
        self.final_activation = self._get_activation(activation)

    def _get_activation(self, name: str):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        else:
            return nn.Identity()

    def _build_branch2_dilated_conv(self, kernel_size, dilations):
        """Construct dilated convolution branches"""
        return DilatedConv1DBlock(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
            kernel_size=kernel_size,
            dilations=dilations or [2, 3, 4, 5, 6], # default value
            dropout=self.dropout
        )

    def _build_branch3_transformer(self, depth, heads):
        """buildTransformerbranch"""
        return TransformerBlock(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            depth=depth,
            heads=heads,
            dropout=self.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward propagation
        Args:
            x: input tensor [batch, time, channels]
        Returns:
            output tensor [batch, time, channels]
        """
        # 0. Check if any branch is enabled (Ablation experiment:0branch situation)
        if self.fusion_weights is None:
            # Direct passthrough (Direct Connection)
            # Note: The premise is input_dim == output_dim
            if self.input_dim != self.output_dim:
                 # Why: 0-branch 直连模式只服务于“特定 encoder/decoder 天然对齐”的消融实验；
                 # 一旦维度不对齐就必须立刻报错，避免把实验配置错误误当成普通前向逻辑继续跑下去。
                 raise ValueError(
                     f"0-branch direct mode requires aligned dimensions, but got "
                     f"input_dim={self.input_dim} and output_dim={self.output_dim}."
                 )
            
            # Direct transparent transmission without passing throughLayerNormandActivation, keep the original flavor
            return x

        # Normalized fusion weight
        softmax_weights = F.softmax(self.fusion_weights, dim=0)
        
        # All convolution branches need to be transposed: [B, T, C] -> [B, C, T]
        x_transposed = x.transpose(1, 2)
        
        branch_outputs = []
        weight_idx = 0

        # branch1: Parallel convolution
        if self.enable_branch1:
            branch1_out_transposed = self.branch1_parallel_conv(x_transposed) # [B, C, T]
            branch1_out = branch1_out_transposed.transpose(1, 2) # [B, T, C]
            branch_outputs.append(softmax_weights[weight_idx] * branch1_out)
            weight_idx += 1
        
        # branch2: dilated convolution
        if self.enable_branch2:
            branch2_out_transposed = self.branch2_dilated_conv(x_transposed) # [B, C, T]
            branch2_out = branch2_out_transposed.transpose(1, 2) # [B, T, C]
            branch_outputs.append(softmax_weights[weight_idx] * branch2_out)
            weight_idx += 1
        
        # branch3: Transformer (Use raw input directly)
        if self.enable_branch3:
            branch3_out = self.branch3_transformer(x) # [batch, time, output_dim]
            branch_outputs.append(softmax_weights[weight_idx] * branch3_out)
            weight_idx += 1
        
        # weighted fusion
        if len(branch_outputs) > 0:
            fused_output = sum(branch_outputs)
        else:
            # Theoretically, we won't reach this point because we have dealt with it before. fusion_weights is None situation
            fused_output = x
        
        # Apply final activations and layer normalization
        activated_output = self.final_activation(fused_output)
        normalized_output = self.final_layer_norm(activated_output)
        
        return normalized_output

    def get_fusion_weights(self):
        """Get the current fusion weight"""
        if self.fusion_weights is None:
            return {
                'branch1_parallel_conv': 0.0,
                'branch2_dilated_conv': 0.0, 
                'branch3_transformer': 0.0,
                'mode': 'Direct Connection (No Branches)'
            }
            
        with torch.no_grad():
            weights = F.softmax(self.fusion_weights, dim=0)
            result = {}
            weight_idx = 0
            
            if self.enable_branch1:
                result['branch1_parallel_conv'] = weights[weight_idx].item()
                weight_idx += 1
            else:
                result['branch1_parallel_conv'] = 0.0
                
            if self.enable_branch2:
                result['branch2_dilated_conv'] = weights[weight_idx].item()
                weight_idx += 1
            else:
                result['branch2_dilated_conv'] = 0.0
                
            if self.enable_branch3:
                result['branch3_transformer'] = weights[weight_idx].item()
                weight_idx += 1
            else:
                result['branch3_transformer'] = 0.0
                
            return result
    
    def get_info(self) -> Dict[str, str]:
        """Get adapter information"""
        fusion_weights = self.get_fusion_weights()
        
        enabled_branches = []
        if self.enable_branch1: enabled_branches.append("ParallelConv")
        if self.enable_branch2: enabled_branches.append("DilatedConv")
        if self.enable_branch3: enabled_branches.append("Transformer")
        
        arch_str = " + ".join(enabled_branches) if enabled_branches else "Direct Connection"
        
        return {
            "input_dim": str(self.input_dim),
            "output_dim": str(self.output_dim),
            "type": "Adapter",
            "fusion_weights": str(fusion_weights),
            "architecture": f"Ablation Mode: {arch_str}"
        }
