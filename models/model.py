import torch
import torch.nn as nn
import numpy as np
from models.transformer import TransformerBlock
from pesq import pesq
from joblib import Parallel, delayed
import math


class CwiseRMSNorm(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 n_freqs: int,
                 affine = True,
                 ):
        super(CwiseRMSNorm, self).__init__()
        self.nband = n_freqs
        self.feature_dim = feature_dim
        self.affine = affine
        self.eps = 1e-5
        self.gain_matrix = nn.Parameter(torch.ones([1, feature_dim, 1, 1]))
        self.bias_matrix = nn.Parameter(torch.zeros([1, feature_dim, 1, 1]))

    def forward(self, input):
            # RMS = sqrt(mean(x^2))
            rms_ = torch.sqrt(torch.mean(input ** 2, dim=(2,3), keepdim=True) + self.eps) 
            # 2. Normalize
            input = input / rms_
            # 3. Affine transform
            if self.affine:
                input = input * self.gain_matrix + self.bias_matrix
                
            return input

class CFWiseComplexRMSNorm(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 n_freqs: int,
                 ):
        super(CFWiseComplexRMSNorm, self).__init__()
        self.nband = n_freqs
        self.feature_dim = feature_dim
        self.gain_matrix = nn.Parameter(torch.ones([1, feature_dim, 1, n_freqs]))

    def forward(self, input_r, input_i):
        input_amp = torch.sqrt(input_r ** 2 + input_i ** 2 + 1e-9)
        rms = torch.sqrt(torch.mean(input_amp ** 2, dim=(2,3), keepdim=True) + 1e-7)
        input_r = input_r / rms
        input_i = input_i / rms
        output_real = input_r * self.gain_matrix
        output_imag = input_i * self.gain_matrix
        return output_real, output_imag
        
class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(ComplexConv, self).__init__()
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, **kwargs, bias=False)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, **kwargs, bias=False)
        

    def forward(self, real, imag):
        real_conv_real = self.conv_re(real)
        real_conv_imag = self.conv_re(imag)
        imag_conv_real = self.conv_im(real)
        imag_conv_imag = self.conv_im(imag)
        
        real_ = real_conv_real - imag_conv_imag
        imaginary_ = real_conv_imag + imag_conv_real
        
        return real_, imaginary_

class SPConvTranspose2dComplex(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, r=1, up=False, **kwargs):
        super(SPConvTranspose2dComplex, self).__init__()
        self.pad1 = nn.ConstantPad2d((kernel_size[1]//2, kernel_size[1]//2, kernel_size[0]-1, 0), value=0.)
        self.out_channels = out_channels
        self.conv = ComplexConv(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1), **kwargs)
        self.r = r
        if up == True:
            self.conv._init_phase_preserving(init_mode='up')
    def forward(self, real, imag):
        real = self.pad1(real)
        imag = self.pad1(imag)
        real_out, imag_out = self.conv(real, imag)
        batch_size, nchannels, H, W = real_out.shape
        real_out = real_out.view((batch_size, self.r, nchannels // self.r, H, W))
        real_out = real_out.permute(0, 2, 3, 4, 1)
        real_out = real_out.contiguous().view((batch_size, nchannels // self.r, H, -1))

        imag_out = imag_out.view((batch_size, self.r, nchannels // self.r, H, W))
        imag_out = imag_out.permute(0, 2, 3, 4, 1)
        imag_out = imag_out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return real_out, imag_out

class LearnableSigmoid3d(nn.Module):
    def __init__(self, in_features_1, in_features_2, initial_beta=3.0):
        super().__init__()
        param_shape_original = (in_features_1, 1, in_features_2)
        self.slope = nn.Parameter(torch.ones(param_shape_original))
        self.slope.requiresGrad = True
        self.beta = initial_beta # nn.Parameter(torch.ones(param_shape_original)*initial_beta)
        # self.beta.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1, **kwargs):
        super(SPConvTranspose2d, self).__init__()
        self.pad = nn.ConstantPad2d((kernel_size[1]//2, kernel_size[1]//2, kernel_size[0]-1, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1), **kwargs)
        self.r = r
        
    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

class InteConvBlock(nn.Module):

    def __init__(self,  kernel_size, amp_in_chn=1, ang_in_chn=1, amp_out_chn=1, ang_out_chn=1, n_freqs=201, separate_grad=False, simple=False, **kwargs):
        super().__init__()
        self.ang_in_chn = ang_in_chn
        self.amp_in_chn = amp_in_chn

        self.amp_out_chn = amp_out_chn
        self.ang_out_chn = ang_out_chn

        self.separate_grad = separate_grad
        self.simple = simple
        self.conv_amp = nn.Conv2d(amp_in_chn, amp_out_chn, kernel_size, **kwargs)
        self.conv_ang = ComplexConv(ang_in_chn, ang_out_chn, kernel_size, **kwargs)
        self.norm_amp = CwiseRMSNorm(amp_out_chn, n_freqs, affine=True)
        self.norm_ang = CFWiseComplexRMSNorm(ang_out_chn, n_freqs)
        self.act_amp = nn.SiLU()

        if not self.simple:
            self.pconv_ang2amp = nn.Conv2d(ang_out_chn, amp_out_chn, kernel_size=(1,1))
            self.pconv_amp2ang = nn.Conv2d(amp_out_chn, ang_out_chn, kernel_size=(1,1))
            torch.nn.init.constant_(self.pconv_ang2amp.weight, 0)
            torch.nn.init.constant_(self.pconv_amp2ang.weight, 0)
            torch.nn.init.constant_(self.pconv_ang2amp.bias, -math.log(2))
            torch.nn.init.constant_(self.pconv_amp2ang.bias, -math.log(2))

            self.act_ang2amp = LearnableSigmoid3d(amp_out_chn, n_freqs, initial_beta=3.0)
            self.act_amp2ang = LearnableSigmoid3d(ang_out_chn, n_freqs, initial_beta=3.0)
        else:
            self.pconv_ang2amp=None
            self.pconv_amp2ang=None
            self.act_ang2amp=None
            self.act_amp2ang=None

    def forward(self, x):
        cos_in, sin_in, std_in = torch.split(x, [self.ang_in_chn, self.ang_in_chn, self.amp_in_chn], dim=1)

        cos_out, sin_out = self.conv_ang(cos_in, sin_in)
        cos_out, sin_out = self.norm_ang(cos_out, sin_out)
        ang_amp_out = torch.sqrt(cos_out ** 2 + sin_out ** 2 + 1e-9)
        std_out = self.act_amp(self.norm_amp(self.conv_amp(std_in)))
        if not self.simple:
            if self.separate_grad:
                w_amp2ang = self.act_amp2ang((self.pconv_amp2ang(std_out.detach())))
                w_ang2amp = self.act_ang2amp((self.pconv_ang2amp(ang_amp_out.detach())))

            else:
                w_amp2ang = self.act_amp2ang((self.pconv_amp2ang(std_out)))
                w_ang2amp = self.act_ang2amp((self.pconv_ang2amp(ang_amp_out)))
            return torch.cat((w_amp2ang * cos_out, w_amp2ang * sin_out, w_ang2amp * std_out), dim=1)
        else:
            return torch.cat((cos_out, sin_out, std_out), dim=1)

class InteConvBlockTranspose(nn.Module):
    def __init__(self, kernel_size, amp_in_chn=1, ang_in_chn=1, amp_out_chn=1, ang_out_chn=1, n_freqs=129, separate_grad=False, **kwargs):
        super().__init__()

        self.ang_in_chn = ang_in_chn
        self.amp_in_chn = amp_in_chn
        self.amp_out_chn = amp_out_chn
        self.ang_out_chn = ang_out_chn

        self.conv_amp = SPConvTranspose2d(amp_in_chn, amp_out_chn, kernel_size, **kwargs)
        self.conv_ang = SPConvTranspose2dComplex(ang_in_chn, ang_out_chn, kernel_size, **kwargs)

        self.pconv_ang2amp = nn.Conv2d(ang_out_chn, amp_out_chn, 1)
        self.pconv_amp2ang = nn.Conv2d(amp_out_chn, ang_out_chn, 1)
        torch.nn.init.constant_(self.pconv_ang2amp.weight, 0)
        torch.nn.init.constant_(self.pconv_amp2ang.weight, 0)
        torch.nn.init.constant_(self.pconv_ang2amp.bias, -math.log(2))
        torch.nn.init.constant_(self.pconv_amp2ang.bias, -math.log(2))
        self.norm_amp = CwiseRMSNorm(amp_out_chn, n_freqs, affine=True)
        self.norm_ang = CFWiseComplexRMSNorm(ang_out_chn, n_freqs)

        self.act_amp = nn.SiLU()
        self.act_ang2amp = LearnableSigmoid3d(amp_out_chn, n_freqs, initial_beta=3.0)
        self.act_amp2ang = LearnableSigmoid3d(ang_out_chn, n_freqs, initial_beta=3.0)
        self.separate_grad = separate_grad
    def forward(self, x):
        cos_in, sin_in, std_in = torch.split(x, [self.ang_in_chn, self.ang_in_chn, self.amp_in_chn], dim=1)

        cos_out, sin_out = self.conv_ang(cos_in, sin_in)
        cos_out, sin_out = self.norm_ang(cos_out, sin_out)
        std_out = self.act_amp(self.norm_amp(self.conv_amp(std_in)))
        ang_amp_out = torch.sqrt(cos_out ** 2 + sin_out ** 2 + 1e-9)

        if self.separate_grad:
            w_amp2ang = self.act_amp2ang((self.pconv_amp2ang(std_out.detach())))
            w_ang2amp = self.act_ang2amp((self.pconv_ang2amp(ang_amp_out.detach())))
            # print(w_ang2amp)
        else:
            w_amp2ang = self.act_amp2ang((self.pconv_amp2ang(std_out)))
            w_ang2amp = self.act_ang2amp((self.pconv_ang2amp(ang_amp_out)))

        return torch.cat((w_amp2ang * cos_out, w_amp2ang * sin_out, w_ang2amp * std_out), dim=1)

class TSTransformerBlock(nn.Module):
    def __init__(self,h):
        super(TSTransformerBlock, self).__init__()

        # t_pe = RotaryEmbedding(16)
        # f_pe = RotaryEmbedding(16)
        self.time_transformer = TransformerBlock(h)
        self.freq_transformer = TransformerBlock(h)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()
        return x

class DenseBlock(nn.Module):
    def __init__(self, kernel_size=(2, 3), depth=4, amp_in_chn=48, ang_in_chn=16, n_freqs=201, separate_grad=False):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.amp_in_chn = amp_in_chn
        self.ang_in_chn = ang_in_chn
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** i
            pad_length = dilation
                        
            dense_conv_fan = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.),
                InteConvBlock(kernel_size, 
                    amp_in_chn=self.amp_in_chn*(i+1), 
                    ang_in_chn=self.ang_in_chn*(i+1), 
                    amp_out_chn=self.amp_in_chn, 
                    ang_out_chn=self.ang_in_chn,
                    dilation=(dilation, 1), 
                    n_freqs=n_freqs,
                    separate_grad=separate_grad)
            )
            self.dense_block.append(dense_conv_fan)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            x_cos, x_sin, x_std = torch.split(x, [self.ang_in_chn, self.ang_in_chn, self.amp_in_chn], dim=1)
            _, C, _, _ = skip.shape
            skip_cos, skip_sin, skip_std = torch.split(skip, [self.ang_in_chn*(i+1), self.ang_in_chn*(i+1), self.amp_in_chn*(i+1)], dim=1)
            skip = torch.cat([x_cos, skip_cos, x_sin, skip_sin, x_std, skip_std], dim=1)
        return x

class DenseEncoder(nn.Module):
    def __init__(self, in_channel, ang_chn=16, amp_chn=48):
        super(DenseEncoder, self).__init__()

        self.dense_conv_1 = InteConvBlock((1,1), amp_in_chn=1, ang_in_chn=1, amp_out_chn=amp_chn, ang_out_chn=ang_chn, n_freqs=201, separate_grad=True, simple=True)
        self.dense_block = DenseBlock(depth=4, amp_in_chn=amp_chn, ang_in_chn=ang_chn, n_freqs=201)
        
        self.dense_conv_3 = InteConvBlock(
            kernel_size=(1, 3), 
            amp_in_chn=amp_chn,
            ang_in_chn=ang_chn,
            amp_out_chn=amp_chn,
            ang_out_chn=ang_chn,
            stride=(1, 2), 
            padding=(0, 1),
            n_freqs=101
        )

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channel=1, ang_chn=16, amp_chn=48):
        super(Decoder, self).__init__()
        self.ang_chn = ang_chn
        self.amp_chn = amp_chn
 
        self.dense_block = DenseBlock(depth=4, ang_in_chn=ang_chn, amp_in_chn=amp_chn, n_freqs=101, separate_grad=False)

        self.upsample_layer = InteConvBlockTranspose(
            kernel_size=(1, 3), 
            amp_in_chn=amp_chn, 
            ang_in_chn=ang_chn,
            amp_out_chn=amp_chn, 
            ang_out_chn=ang_chn,
            r=2,
            n_freqs=202,
            separate_grad=False
        )

        self.amp_conv_layer = nn.Sequential(nn.Conv2d(
            in_channels=self.amp_chn, 
            out_channels=out_channel, 
            kernel_size=(1, 2)),
            nn.ReLU())
  
        self.ang_conv = ComplexConv(self.ang_chn, out_channel, (1,2))

    def forward(self, x):

        x = self.dense_block(x)
        # x = self.pad1d(x)
        x = self.upsample_layer(x)
        x_ang_r, x_ang_i, x_amp = torch.split(x, [self.ang_chn, self.ang_chn, self.amp_chn], dim=1)
        x_r, x_i = self.ang_conv(x_ang_r, x_ang_i)
        x_ang = torch.atan2(x_i + 1e-9, x_r + 1e-9)
        x_ang = x_ang.permute(0, 3, 2, 1).squeeze(-1)

        x_amp = self.amp_conv_layer(x_amp)
        x_amp = x_amp.permute(0, 3, 2, 1).squeeze(-1) # [B, F, T]
        return x_amp, x_ang
    
class MPNet(nn.Module):

    def __init__(self, h, **kwargs):
        super(MPNet, self).__init__()
        self.num_tscblocks = h.num_tsconformers
        amp_chn = h.amp_chn
        ang_chn = h.ang_chn
        self.dense_encoder = DenseEncoder(in_channel=3, amp_chn=amp_chn, ang_chn=ang_chn)
        self.TSTransformer = nn.ModuleList([TSTransformerBlock(h) for _ in range(self.num_tscblocks)])
        self.decoder = Decoder(out_channel=1, amp_chn=amp_chn, ang_chn=ang_chn)
        
    def forward(self, noisy_amp, noisy_pha):
        # Encoder
        x = torch.stack((torch.cos(noisy_pha), torch.sin(noisy_pha), noisy_amp), dim=1) # [B, 3, F, T]
        x = x.permute(0, 1, 3, 2) # [B, C, T, F]

        x_encoded = self.dense_encoder(x)

        # Transformer Blocks
        x_transformed = x_encoded
        for i in range(self.num_tscblocks):
            x_transformed = self.TSTransformer[i](x_transformed)
        # Decoders
        denoised_amp, denoised_pha = self.decoder(x_transformed)
        
        # Reconstruct complex spectrum
        denoised_com = torch.stack((denoised_amp * torch.cos(denoised_pha),
                                      denoised_amp * torch.sin(denoised_pha)), dim=-1)
        return denoised_amp, denoised_pha, denoised_com
    
def phase_losses(phase_r, phase_g):

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss, gd_loss, iaf_loss
    
def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def pesq_score(utts_r, utts_g, h):

    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(), 
                            16000)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score

def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        pesq_score = -1

    return pesq_score

