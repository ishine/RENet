import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import GRU, Linear, LayerNorm, Dropout

import math
import numpy as np


class ComplexConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(ComplexConv1d, self).__init__()
        ## Model components
        self.conv_re = nn.Conv1d(in_channel, out_channel, kernel_size, **kwargs, bias=False)
        self.conv_im = nn.Conv1d(in_channel, out_channel, kernel_size, **kwargs, bias=False)

    def forward(self, real, imag):
            
        real_conv_real = self.conv_re(real)
        real_conv_imag = self.conv_re(imag)
        imag_conv_real = self.conv_im(real)
        imag_conv_imag = self.conv_im(imag)

        real_ = real_conv_real - imag_conv_imag
        imaginary_ = real_conv_imag + imag_conv_real
        
        return real_, imaginary_

class ComplexConvTranspose1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(ComplexConvTranspose1d, self).__init__()
        ## Model components
        self.conv_re = nn.ConvTranspose1d(in_channel, out_channel, kernel_size, **kwargs, bias=False)
        self.conv_im = nn.ConvTranspose1d(in_channel, out_channel, kernel_size, **kwargs, bias=False)

    def forward(self, real, imag):
            
        real_conv_real = self.conv_re(real)
        real_conv_imag = self.conv_re(imag)
        imag_conv_real = self.conv_im(real)
        imag_conv_imag = self.conv_im(imag)

        real_ = real_conv_real - imag_conv_imag
        imaginary_ = real_conv_imag + imag_conv_real
        
        return real_, imaginary_

class ComplexRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-10):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x_r, x_i) -> tuple[torch.Tensor, torch.Tensor]:
        mag_sq = x_r**2 + x_i**2 

        mean_mag_sq = mag_sq.mean(dim=-1, keepdim=True)

        inv_rms = torch.rsqrt(mean_mag_sq + self.eps)

        gamma = self.gamma.view(1, 1, -1)
        
        # (B, L, C) * (B, L, 1) * (1, 1, C)
        x_r = x_r * inv_rms * gamma
        x_i = x_i * inv_rms * gamma

        return x_r, x_i

class ComplexFFN(nn.Module):
    def __init__(self, chn=16, chn_inner=64, guide_chn=48, conv1d_kernel=4, conv1d_shift=1, dropout=0., **kwargs):
        super().__init__()

        self.chn = chn
        self.chn_inner = chn_inner
        self.guide_chn = guide_chn
        self.conv1d = ComplexConv1d(chn, chn_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.gate_act = nn.SiLU()
        self.gate_ln = LayerNorm(chn_inner)
        self.deconv1d = ComplexConvTranspose1d(chn_inner, chn, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x_real_orig, x_imag_orig):
        """ forward
        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        b, s2, h = x_real_orig.shape
        # x = x.contiguous().view(b * s1, s2, h)
        x_real = x_real_orig.transpose(-1, -2)
        x_imag = x_imag_orig.transpose(-1, -2)
        # padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x_real = F.pad(x_real, (self.diff_ks, seq_len - s2 - self.diff_ks))
        x_imag = F.pad(x_imag, (self.diff_ks, seq_len - s2 - self.diff_ks))
        # conv-deconv1d
        x_real, x_imag = self.conv1d(x_real, x_imag)
        x_act = torch.sqrt(x_real[..., self.chn_inner:, :] ** 2 + x_imag[..., self.chn_inner:, :] ** 2 + 1e-9)
        x_act = self.gate_ln(x_act.permute(0,2,1)).permute(0,2,1)
        gate = self.gate_act(x_act)
        x_real = x_real[..., : self.chn_inner, :] * gate
        x_imag = x_imag[..., : self.chn_inner, :] * gate

        x_real, x_imag = self.deconv1d(x_real, x_imag)
        x_real = x_real.transpose(-1, -2)
        x_imag = x_imag.transpose(-1, -2)
        # cut necessary part
        x_real = x_real[..., self.diff_ks : self.diff_ks + s2, :]
        x_imag = x_imag[..., self.diff_ks : self.diff_ks + s2, :]
        return x_real, x_imag

class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0.1):
        super(FFN, self).__init__()
        self.gru = GRU(d_model, d_model*2, 1, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            self.linear = Linear(d_model*2*2, d_model)
        else:
            self.linear = Linear(d_model*2, d_model)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_r = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_i = nn.Parameter(torch.empty(out_features, in_features))

        self.use_bias = bias
        if self.use_bias:
            self.bias_r = nn.Parameter(torch.empty(out_features))
            self.bias_i = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_i', None)

        self.reset_parameters()
    def reset_parameters(self) -> None:
        # Initialize weights similar to nn.Linear for stable training
        nn.init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_i, a=math.sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_r, -bound, bound)
            nn.init.uniform_(self.bias_i, -bound, bound)

    def forward(self, x_r, x_i):
  
        out_r = F.linear(x_r, self.weight_r) - F.linear(x_i, self.weight_i)
        out_i = F.linear(x_r, self.weight_i) + F.linear(x_i, self.weight_r)

        if self.use_bias:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i

        return out_r, out_i

class CustomAttention(nn.Module):
    def __init__(self, amp_dim=48, ang_dim=16, num_heads=4, amp_qk_head_dim=12, ang_qk_head_dim=6, amp_v_head_dim=12, ang_v_head_dim=6):
        super().__init__()
        self.num_heads = num_heads
        
        # Dimensions per head
        self.amp_qk_head_dim = amp_qk_head_dim
        self.ang_qk_head_dim = ang_qk_head_dim
        self.amp_v_head_dim = amp_v_head_dim
        self.ang_v_head_dim = ang_v_head_dim
        
        # Total internal dimensions (Head_Dim * Num_Heads)
        self.inner_dim_qk_amp = num_heads * amp_qk_head_dim
        self.inner_dim_qk_ang = num_heads * ang_qk_head_dim
        self.inner_dim_v_amp = num_heads * amp_v_head_dim
        self.inner_dim_v_ang = num_heads * ang_v_head_dim

        # Projections
        # Q and K
        self.to_q_amp = nn.Linear(amp_dim, self.inner_dim_qk_amp, bias=False)
        self.to_k_amp = nn.Linear(amp_dim, self.inner_dim_qk_amp, bias=False)
        self.to_q_ang = ComplexLinear(ang_dim, self.inner_dim_qk_ang, bias=False)
        self.to_k_ang = ComplexLinear(ang_dim, self.inner_dim_qk_ang, bias=False)
        # V (Values)
        self.to_v_amp = nn.Linear(amp_dim, self.inner_dim_v_amp, bias=False)
        self.to_v_ang = ComplexLinear(ang_dim, self.inner_dim_v_ang, bias=False)
        
        # Output projections
        self.to_out_amp = nn.Linear(self.inner_dim_v_amp, amp_dim)
        self.to_out_ang = ComplexLinear(self.inner_dim_v_ang, ang_dim, bias=False)

    def forward(self, x_ang, x_amp):
        B, L, CA = x_ang.shape
        ang_r, ang_i = torch.split(x_ang, [CA//2, CA//2], dim=-1)

        # ==========================================================
        # 1. Project Q, K, V
        # ==========================================================
        
        # --- Query & Key ---
        q_amp = self.to_q_amp(x_amp)      # [B, L, H * D_qk_amp]
        k_amp = self.to_k_amp(x_amp)
        q_ang_r, q_ang_i = self.to_q_ang(ang_r, ang_i) # [B, L, H * D_qk_ang]
        k_ang_r, k_ang_i = self.to_k_ang(ang_r, ang_i)

        # Reshape and Transpose Q/K
        # Function to reshape: (B, L, H*D) -> (B, H, L, D)
        def reshape_head(x, head_dim):
            return x.view(B, L, self.num_heads, head_dim).transpose(1, 2)

        q_amp = reshape_head(q_amp, self.amp_qk_head_dim)
        k_amp = reshape_head(k_amp, self.amp_qk_head_dim)
        q_ang_r = reshape_head(q_ang_r, self.ang_qk_head_dim)
        q_ang_i = reshape_head(q_ang_i, self.ang_qk_head_dim)
        k_ang_r = reshape_head(k_ang_r, self.ang_qk_head_dim)
        k_ang_i = reshape_head(k_ang_i, self.ang_qk_head_dim)
        # Concatenate parts for Q and K
        # Structure: [Real_Angle, Imag_Angle, Amplitude]
        q = torch.cat((q_ang_r, q_ang_i, q_amp), dim=-1)
        k = torch.cat((k_ang_r, k_ang_i, k_amp), dim=-1)

        # --- Value (V) ---
        v_amp = self.to_v_amp(x_amp)
        v_ang_r, v_ang_i = self.to_v_ang(ang_r, ang_i)

        # Reshape V to (B, H, L, D)
        v_amp = reshape_head(v_amp, self.amp_v_head_dim)
        v_ang_r = reshape_head(v_ang_r, self.ang_v_head_dim)
        v_ang_i = reshape_head(v_ang_i, self.ang_v_head_dim)

        # Concatenate V: [Amplitude, Real_Angle, Imag_Angle]
        v_combined = torch.cat((v_ang_r, v_ang_i, v_amp), dim=-1)

        # ==========================================================
        # 2. Attention
        # ==========================================================
        # out shape: [B, H, L, Total_V_Head_Dim]

        out = F.scaled_dot_product_attention(q, k, v_combined)

        # ==========================================================
        # 3.  Output Processing
        # ==========================================================
        
        split_sizes = [self.ang_v_head_dim, self.ang_v_head_dim, self.amp_v_head_dim]
        out_ang_r, out_ang_i, out_amp = torch.split(out, split_sizes, dim=-1)

        # Function to merge heads: (B, H, L, D) -> (B, L, H*D)
        def merge_head(x):
            return x.transpose(1, 2).contiguous().view(B, L, -1)

        # Merge heads individually for each component

        out_amp = merge_head(out_amp)      # [B, L, H * amp_v_dim]
        out_ang_r = merge_head(out_ang_r)  # [B, L, H * ang_v_dim]
        out_ang_i = merge_head(out_ang_i)  # [B, L, H * ang_v_dim]

        # ==========================================================
        # 4. Final Projection
        # ==========================================================

        out_amp = self.to_out_amp(out_amp)
        out_ang_r, out_ang_i = self.to_out_ang(out_ang_r, out_ang_i)

        return out_ang_r, out_ang_i, out_amp

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(RMSNorm, self).__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        # Create a learnable parameter for the gain (gamma), initialized to 1
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 1. Calculate the Root Mean Square (RMS) along the last dimension
        norm = torch.mean(x ** 2, dim=-1, keepdim=True)
        
        # 2. Normalize x by dividing by RMS (adding eps for stability)
        x_normed = x * torch.rsqrt(norm + self.eps)

        # 3. Apply the learnable gain (affine transformation)
        return self.g * x_normed
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        h,
        dropout=0.0
    ):
        super().__init__()
        amp_chn = h.amp_chn
        ang_chn = h.ang_chn
        self.amp_chn = amp_chn
        self.ang_chn = ang_chn
        self.norm1 = RMSNorm(amp_chn)
        self.cnorm1 = ComplexRMSNorm(ang_chn)

        self.att = CustomAttention(amp_chn, ang_chn, h.n_heads, h.amp_attnhead_dim, h.ang_attnhead_dim, h.amp_attnhead_dim, h.ang_attnhead_dim)

        self.norm2 = RMSNorm(amp_chn)
        self.cnorm2 = ComplexRMSNorm(ang_chn)

        self.amp_ffn = FFN(amp_chn, dropout=dropout)
        self.ang_ffn = ComplexFFN(ang_chn, dropout=dropout)

        self.norm3 = RMSNorm(amp_chn)
        self.cnorm3 = ComplexRMSNorm(ang_chn)

    def forward(self, x):
        x_ang_r, x_ang_i, x_amp = torch.split(x, [self.ang_chn, self.ang_chn, self.amp_chn], dim=-1)

        x_amp_t = self.norm1(x_amp)
        x_ang_r_t, x_ang_i_t = self.cnorm1(x_ang_r, x_ang_i)
        x_ang_r_t, x_ang_i_t, x_amp_t = self.att(torch.cat((x_ang_r_t, x_ang_i_t), dim=-1), x_amp_t)

        x_amp, x_ang_r, x_ang_i = x_amp + x_amp_t, x_ang_r + x_ang_r_t, x_ang_i + x_ang_i_t

        x_amp_t = self.norm2(x_amp)
        x_amp_t = self.amp_ffn(x_amp_t)
        x_amp = x_amp + x_amp_t
        x_amp = self.norm3(x_amp)

        x_ang_r_t, x_ang_i_t = self.cnorm2(x_ang_r, x_ang_i)
        x_ang_r_t, x_ang_i_t = self.ang_ffn(x_ang_r_t, x_ang_i_t)
        x_ang_r, x_ang_i = x_ang_r + x_ang_r_t, x_ang_i + x_ang_i_t
        x_ang_r, x_ang_i = self.cnorm3(x_ang_r, x_ang_i)

        return torch.cat((x_ang_r, x_ang_i, x_amp), dim=-1)

def main():
    x = torch.randn(4, 64, 401, 201)
    b, c, t, f = x.size()
    x = x.permute(0, 3, 2, 1).contiguous().view(b, f*t, c)
    transformer = TransformerBlock(d_model=64, n_heads=4)
    x = transformer(x)
    x =  x.view(b, f, t, c).permute(0, 3, 2, 1)
    print(x.size())

if __name__ == '__main__':
    main()