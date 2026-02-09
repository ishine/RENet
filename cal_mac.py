import torch
import torch.nn as nn
import math

from models.model import MPNet, InteConvBlock, InteConvBlockTranspose
from models.transformer import RMSNorm, ComplexRMSNorm, ComplexFFN, CustomAttention

class EndToEndModel(nn.Module):
    
    """
    Wrapper to include STFT and iSTFT in the model pipeline.
    """
    def __init__(self, model, n_fft, hop_size, win_size):
        super(EndToEndModel, self).__init__()
        self.model = model
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        
        # Register window as buffer to be part of the model state (optional but good practice)
        self.register_buffer('window', torch.hann_window(win_size))

    def forward(self, waveform):
        from dataset import mag_pha_istft, mag_pha_stft
        # 1. STFT (Waveform -> Mag, Pha)
        # Input waveform: (B, Time)
        mag, pha, com = mag_pha_stft(
            waveform, 
            self.n_fft, 
            self.hop_size, 
            self.win_size, 
            center=True
        )
        
        # 2. MPNet Inference
        # We pass the calculated mag and pha to the inner model
        # Output is expected to be (refined_mag, refined_pha) or just refined_pha
        outputs = self.model(mag, pha)
        
        # Handle Output Unpacking
        if isinstance(outputs, (tuple, list)):
            out_mag, out_pha = outputs[0], outputs[1]
        else:
            # If model only returns phase, reuse input magnitude
            out_mag = mag
            out_pha = outputs

        # 3. iSTFT (Mag, Pha -> Waveform)
        rec_waveform = mag_pha_istft(
            out_mag, 
            out_pha, 
            self.n_fft, 
            self.hop_size, 
            self.win_size, 
            center=True
        )
        
        return rec_waveform
    
def count_complex_rms_norm(m, x, y):
    """
    ComplexRMSNorm:
    1. Mag Sq: r^2 + i^2 (2 muls)
    2. Norm: r * inv_rms, i * inv_rms (2 muls)
    3. Affine: r * gamma, i * gamma (2 muls)
    Total ~ 6 muls per complex element pair.
    """
    x_r = x[0] # Real part tensor
    # x_i = x[1] # Imag part tensor (same size)
    
    total_elements = x_r.numel()
    
    # 6 ops per element
    total_macs = 6 * total_elements
    m.total_ops += torch.DoubleTensor([int(total_macs)])
    
def count_inte_conv_block(m, x, y):
    """
    Counts FLOPs for InteConvBlock (Black Box approach).
    Includes: Main Convs + Norms + Interaction Path
    """
    # y is the output tensor: [B, Cout_Total, F_out, T_out]
    B, _, F_out, T_out = y.shape
    
    total_macs = 0
    
    # --- 1. Main Convolutions (Standard & Complex) ---
    kh, kw = m.conv_amp.kernel_size
    
    # A. conv_amp (Standard Conv2d)
    cin_amp = m.amp_in_chn
    cout_amp = m.amp_out_chn
    # MACs = Output_Pixels * Cout * (Cin * K * K)
    total_macs += (B * F_out * T_out) * cout_amp * (cin_amp * kh * kw)

    # B. conv_ang (ComplexConv -> 4x Standard Conv)
    # We look at the internal conv_re kernel size
    kh_c, kw_c = m.conv_ang.conv_re.kernel_size
    cin_ang = m.ang_in_chn
    cout_ang = m.ang_out_chn
    
    # 4 passes (RR, RI, IR, II)
    total_macs += 4 * ((B * F_out * T_out) * cout_ang * (cin_ang * kh_c * kw_c))
    
    # --- 2. Normalizations (Merged Logic) ---
    
    # A. norm_amp (CwiseRMSNorm)
    # Logic: 2 MACs per element (Standardization + Affine)
    norm_amp_elements = B * m.amp_out_chn * F_out * T_out
    total_macs += 2 * norm_amp_elements

    # B. norm_ang (CFWiseComplexRMSNorm)
    # Logic: 6 MACs per element pair (Amp calc + Norm + Gain)
    # Note: 'elements' here refers to the size of just the real part (or imag part)
    norm_ang_elements = B * m.ang_out_chn * F_out * T_out
    total_macs += 6 * norm_ang_elements

    # --- 3. Interaction Path ---
    if not m.simple:
        # A. pconv_ang2amp (1x1 Conv: Cin * Cout)
        total_macs += (B * F_out * T_out) * m.amp_out_chn * m.ang_out_chn
        
        # B. pconv_amp2ang (1x1 Conv: Cin * Cout)
        total_macs += (B * F_out * T_out) * m.ang_out_chn * m.amp_out_chn
        
        # C. LearnableSigmoid3d (3 ops per pixel)
        sig_ops = 3
        total_macs += sig_ops * norm_amp_elements # act_ang2amp
        total_macs += sig_ops * norm_ang_elements # act_amp2ang
        
        # D. Masking (Element-wise multiplications)
        # 3 muls per pixel location (w_amp*cos, w_amp*sin, w_ang*std)
        total_macs += 3 * norm_ang_elements

    m.total_ops += torch.DoubleTensor([int(total_macs)])


def count_custom_attention(m, x, y):
    """
    Handler for CustomAttention.
    Must manually count:
    1. Linear Projections (Amp and Ang)
    2. Attention Mechanism (The O(L^2) part)
    3. Output Projections
    """
    # Inputs: x_ang (B, L, 2*Ang_Dim), x_amp (B, L, Amp_Dim)
    x_ang = x[0]
    x_amp = x[1]
    
    B, L, _ = x_amp.shape
    
    # --- 1. Projections ---
    # A. Amplitude Projections (Standard Linear)
    # 3 projections: to_q_amp, to_k_amp, to_v_amp
    # Cost: B*L * In * Out
    macs_proj_amp = 0
    macs_proj_amp += (B * L * m.to_q_amp.in_features * m.to_q_amp.out_features)
    macs_proj_amp += (B * L * m.to_k_amp.in_features * m.to_k_amp.out_features)
    macs_proj_amp += (B * L * m.to_v_amp.in_features * m.to_v_amp.out_features)
    
    # B. Angle Projections (Complex Linear)
    # 3 projections: to_q_ang, to_k_ang, to_v_ang
    # ComplexLinear cost = 4 * Standard_Linear
    # Note: Input to ComplexLinear here is split (ang_dim), so we use m.to_q_ang.in_features
    macs_proj_ang = 0
    macs_proj_ang += 4 * (B * L * m.to_q_ang.in_features * m.to_q_ang.out_features)
    macs_proj_ang += 4 * (B * L * m.to_k_ang.in_features * m.to_k_ang.out_features)
    macs_proj_ang += 4 * (B * L * m.to_v_ang.in_features * m.to_v_ang.out_features)
    
    # --- 2. Attention Mechanism ---
    # Shapes after projection and concat:
    # Q, K: [B, H, L, D_qk]
    # V:    [B, H, L, D_v]
    
    # D_qk = amp_qk_head_dim + 2 * ang_qk_head_dim (Real + Imag)
    d_qk = m.amp_qk_head_dim + 2 * m.ang_qk_head_dim
    # D_v  = amp_v_head_dim + 2 * ang_v_head_dim
    d_v  = m.amp_v_head_dim + 2 * m.ang_v_head_dim
    
    num_heads = m.num_heads
    
    # Score Calculation: Q @ K.T -> [B, H, L, D_qk] @ [B, H, D_qk, L] -> [B, H, L, L]
    macs_score = B * num_heads * (L * L) * d_qk
    
    # Weighted Sum: Attn @ V -> [B, H, L, L] @ [B, H, L, D_v] -> [B, H, L, D_v]
    macs_weight = B * num_heads * (L * L) * d_v
    
    # --- 3. Output Projections ---
    # Amp Output (Standard Linear)
    macs_out_amp = (B * L * m.to_out_amp.in_features * m.to_out_amp.out_features)
    
    # Ang Output (Complex Linear = 4x)
    macs_out_ang = 4 * (B * L * m.to_out_ang.in_features * m.to_out_ang.out_features)
    
    total_macs = macs_proj_amp + macs_proj_ang + macs_score + macs_weight + macs_out_amp + macs_out_ang
    
    m.total_ops += torch.DoubleTensor([int(total_macs)])


def count_inte_conv_block_transpose(m, x, y):
    """
    Counts FLOPs for InteConvBlockTranspose (assuming groups=1).
    """
    input_tensor = x[0] # Low Res: [B, Cin, F_in, T_in]
    output_tensor = y   # High Res: [B, Cout, F_out, T_out]
    
    # Dimensions
    B, _, F_in, T_in = input_tensor.shape
    _, _, F_out, T_out = output_tensor.shape
    
    total_macs = 0
    
    # --- 1. Main Convolutions (Low Resolution) ---
    
    # A. conv_amp (Standard Conv2d)
    conv_layer_amp = m.conv_amp.conv
    kh, kw = conv_layer_amp.kernel_size
    cin_amp = conv_layer_amp.in_channels
    cout_amp_expanded = conv_layer_amp.out_channels # amp_out_chn * r
    
    # MACs = Input_Pixels * Cout_Expanded * (Cin * K * K)
    total_macs += (B * F_in * T_in) * cout_amp_expanded * (cin_amp * kh * kw)

    # B. conv_ang (ComplexConv -> 4x Standard Conv)
    conv_layer_ang = m.conv_ang.conv.conv_re
    kh_c, kw_c = conv_layer_ang.kernel_size
    cin_ang = conv_layer_ang.in_channels
    cout_ang_expanded = conv_layer_ang.out_channels # ang_out_chn * r
    
    # 4 passes (RR, RI, IR, II)
    total_macs += 4 * ((B * F_in * T_in) * cout_ang_expanded * (cin_ang * kh_c * kw_c))
    
    # --- 2. Normalizations (High Resolution) ---
    
    # A. norm_amp (CwiseRMSNorm)
    # 2 MACs per element
    norm_amp_elements = B * m.amp_out_chn * F_out * T_out
    total_macs += 2 * norm_amp_elements

    # B. norm_ang (CFWiseComplexRMSNorm)
    # 6 MACs per element pair
    norm_ang_elements = B * m.ang_out_chn * F_out * T_out
    total_macs += 6 * norm_ang_elements

    # --- 3. Interaction Path (High Resolution) ---
    if not hasattr(m, 'simple') or not m.simple:
        # A. pconv_ang2amp (1x1 Conv)
        total_macs += (B * F_out * T_out) * m.amp_out_chn * m.ang_out_chn
        
        # B. pconv_amp2ang (1x1 Conv)
        total_macs += (B * F_out * T_out) * m.ang_out_chn * m.amp_out_chn
        
        # C. LearnableSigmoid3d (3 ops per pixel)
        sig_ops = 3
        total_macs += sig_ops * norm_amp_elements 
        total_macs += sig_ops * norm_ang_elements 
        
        # D. Masking (Element-wise multiplications)
        total_macs += 3 * norm_ang_elements

    m.total_ops += torch.DoubleTensor([int(total_macs)])


def count_rms_norm(m, x, y):
    """
    Handler for standard RMSNorm.
    Formula: x * rsqrt(mean(x^2)) * gain
    
    Ops breakdown per element:
    1. x^2 (1 mul)
    2. mean (1 add/div effectively)
    3. rsqrt (1 op)
    4. x * rsqrt (1 mul)
    5. result * gain (1 mul)
    Total: ~5 ops per element.
    """
    input_tensor = x[0]
    total_elements = input_tensor.numel()
    total_ops = 5 * total_elements
    m.total_ops += torch.DoubleTensor([int(total_ops)])
def count_complex_ffn(m, x, y):
    """
    Counts FLOPs for ComplexFFN.
    Structure:
    1. ComplexConv1d (Up-projection): Cin -> Inner*2 (Stride S)
    2. Gating/Norms (Element-wise)
    3. ComplexConvTranspose1d (Down-projection): Inner -> Cin (Stride S)
    
    Note: Complex Convs = 4x Standard Convs
    """
    # x is (x_real, x_imag), but ComplexFFN forward expects (B, L, C) format originally
    # Based on your forward pass: forward(x_real_orig, x_imag_orig)
    x_real = x[0] # [B, Seq_Len, Cin]
    
    B, L_in, Cin = x_real.shape
    
    # Get parameters from the internal layers
    # m.conv1d is ComplexConv1d, which has .conv_re
    conv_up = m.conv1d.conv_re
    conv_down = m.deconv1d.conv_re
    
    # --- 1. Up-Projection (ComplexConv1d) ---
    # Input: L_in. Stride: S.
    # The kernel slides approx L_in / S times.
    # Output Channels: chn_inner * 2
    
    kernel_size = m.conv1d_kernel
    stride = m.conv1d_shift
    inner_dim_expanded = m.chn_inner * 2
    
    # Calculate effective steps (output length of the downsampling conv)
    # Using ceil to approximate the padding logic in your forward pass
    L_mid = math.ceil(L_in / stride)
    
    # Standard MACs = Output_Steps * Cout * (Cin * K)
    # Complex MACs = 4 * Standard
    ops_up = 4 * (B * L_mid * inner_dim_expanded * (Cin * kernel_size))
    
    # --- 2. Gating & Norms ---
    # Occurs at low resolution (L_mid)
    # 2.1 Norm (Gate LN): applied to chn_inner*2
    # 2.2 Magnitude calc (sqrt(r^2+i^2)): 2 ops
    # 2.3 SiLU Gate: 1 op
    # 2.4 Element-wise multiplication: 2 ops
    # Total approx 10 ops per element at L_mid
    ops_gate = 10 * (B * L_mid * inner_dim_expanded)
    
    # --- 3. Down-Projection (ComplexConvTranspose1d) ---
    # Input: L_mid. Stride: S. Output: L_in.
    # Transpose Conv logic: The kernel is applied to every pixel of the INPUT (L_mid).
    # Input Channels to deconv: chn_inner (after gating split)
    # Output Channels: Cin
    
    inner_dim_gated = m.chn_inner
    
    # Standard MACs = Input_Steps * Cout * (Cin * K)
    # Note: For transpose conv, "Input_Steps" is the low-res side (L_mid)
    ops_down = 4 * (B * L_mid * Cin * (inner_dim_gated * kernel_size))

    total_macs = ops_up + ops_gate + ops_down
    m.total_ops += torch.DoubleTensor([int(total_macs)]) 
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
def main():
    from thop import profile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Configuration ---
    # These match your EndToEndModel setup
    n_fft = 400
    hop_size = 100
    win_size = 400
    
    h = AttrDict({
        "num_tsconformers": 4,
        "amp_chn":48,
        "ang_chn":16,
        "n_heads":4,
        "amp_attnhead_dim":12,
        "ang_attnhead_dim":6,
    })
    print("Creating MPNet model...")
    mpnet = MPNet(h)
    
    print("Wrapping model with STFT/iSTFT...")
    model = EndToEndModel(mpnet, n_fft, hop_size, win_size)
    model = model.to(device)
        
    # Approx 1 second of audio at 16kHz
    seq_len = 16000
    input_waveform = torch.randn(1, seq_len).to(device)
    
    print("Profiling End-to-End Model...")
    # --- Register Custom Operations ---
    custom_ops = {
        CustomAttention: count_custom_attention,
        InteConvBlock: count_inte_conv_block,
        InteConvBlockTranspose: count_inte_conv_block_transpose,
        ComplexRMSNorm: count_complex_rms_norm,
        RMSNorm: count_rms_norm,
        ComplexFFN: count_complex_ffn
    }
    
    # Run profile
    macs, params = profile(model, inputs=(input_waveform,), custom_ops=custom_ops, verbose=False)
    
    print("-" * 30)
    print(f"Total Params: {params/1e6:.2f} M")
    print(f"Total MACs:   {macs/1e9:.2f} G")
    print("-" * 30)

if __name__ == "__main__":
    main()