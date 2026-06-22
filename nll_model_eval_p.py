import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import GDN, conv3x3, subpel_conv3x3, ResidualBlock
from compressai.models.utils import update_registered_buffers
from compressai.entropy_models import GaussianConditional
from compression_model import CompressionModel

from logisticmixturemodel import LogisticMixtureModel
from custom_layers import conv1x1, subpel_conv1x1, downsample_conv1x1, SWin_Attention, maskedconv7x7_parallel, Conv2d_cond

import math
import random
from global_nii_max import nii_max, nii_max_f
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

EVAL_DDIM_STEPS = 20



def patchify_to_batch(x: torch.Tensor, r: int) -> torch.Tensor:
    """
    x: [n, m, H, W]  (assumed valid: H%r==0, W%r==0)
    return: [n*(H//r)*(W//r), m, r, r]
    """
    n, m, H, W = x.shape
    kH, kW = H // r, W // r

    # [n, m, kH, r, kW, r]
    y = x.view(n, m, kH, r, kW, r)
    # [n, kH, kW, m, r, r]
    y = y.permute(0, 2, 4, 1, 3, 5)
    # [n*kH*kW, m, r, r]
    y = y.reshape(n * kH * kW, m, r, r)
    return y

def unpatchify_from_batch(y: torch.Tensor, n: int, H: int, W: int) -> torch.Tensor:
    """
    y: [n*(H//r)*(W//r), m, r, r]
    return: [n, m, H, W]

    需要 n, H, W 来恢复形状（它们通常你是知道的，比如来自 dataloader）。
    """
    N, m, r, _ = y.shape
    kH, kW = H // r, W // r

    # [n, kH, kW, m, r, r]
    x = y.view(n, kH, kW, m, r, r)
    # [n, m, kH, r, kW, r]
    x = x.permute(0, 3, 1, 4, 2, 5)
    # [n, m, H, W]
    x = x.reshape(n, m, H, W)
    return x
class ResidualFusion(nn.Module):
    def __init__(self, in_ch_double, out_ch):
        super().__init__()
        self.block1 = ResidualBlock(in_ch_double, in_ch_double)
        self.block2 = ResidualBlock(in_ch_double, in_ch_double)
        self.block3 = ResidualBlock(in_ch_double, out_ch)  # 降维

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class AnalysisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch, stride=2)
        self.gdn = GDN(out_ch)
        self.skip = conv3x3(in_ch, out_ch, stride=2)
        self.rb = ResidualBlock(out_ch, out_ch)
        
    def forward(self, input):
        out = self.conv1(input)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)
        out = out + self.skip(input)

        out = self.rb(out)
        return out
        

class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rb = ResidualBlock(in_ch, out_ch)
        self.conv_up = subpel_conv3x3(out_ch, out_ch, r=2)
        self.igdn = GDN(out_ch, inverse=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.upsample = subpel_conv3x3(out_ch, out_ch, r=2)
                
    def forward(self, input):
        out1 = self.rb(input)
        out = self.conv_up(out1)
        out = self.igdn(out)
        out = self.conv(out)
        out = self.leaky_relu(out)
        
        out = out + self.upsample(out1)
        
        return out
    

class Encoder_I(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            AnalysisBlock(in_ch, out_ch),
            AnalysisBlock(out_ch, out_ch),
            SWin_Attention(dim=out_ch, num_heads=8, window_size=8),
            AnalysisBlock(out_ch, out_ch),
            conv3x3(out_ch, out_ch, stride=2),
            SWin_Attention(dim=out_ch, num_heads=8, window_size=4),
        )
        
    def forward(self, input):
        out = self.layers(input)
        return out
    
class Encoder_P(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block1 = AnalysisBlock(in_ch, out_ch)
        self.block2_extra = AnalysisBlock(in_ch, out_ch)  # 对input2使用的block
        self.fusion = ResidualFusion(out_ch * 2, out_ch)  # 简单融合层，可替换为更复杂结构
        self.block2 = AnalysisBlock(out_ch, out_ch)
        self.attn1 = SWin_Attention(dim=out_ch, num_heads=8, window_size=8)
        self.block3 = AnalysisBlock(out_ch, out_ch)
        self.down = conv3x3(out_ch, out_ch, stride=2)
        self.attn2 = SWin_Attention(dim=out_ch, num_heads=8, window_size=4)

    def forward(self, input1, input2):
        input1=F.pixel_unshuffle(input1, 4)
        input2=F.pixel_unshuffle(input2, 4)
        out1 = self.block1(input1)         # 主路径
        out2 = self.block2_extra(input2)   # 分支路径
        fused = self.fusion(torch.cat([out1, out2], dim=1))  # 通道维度拼接融合
        out = self.block2(fused)
        
        out = self.attn1(out)

        out = self.block3(out)
        out = self.down(out)

        out = self.attn2(out)
        return out

        
class Decoder_I(nn.Module):
    def __init__(self, in_ch, out_ch, prior_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            SWin_Attention(dim=in_ch, num_heads=8, window_size=4),
            SynthesisBlock(in_ch, in_ch),
            SynthesisBlock(in_ch, in_ch),
            SWin_Attention(dim=in_ch, num_heads=8, window_size=8),
            SynthesisBlock(in_ch, in_ch)
        )
        self.conv_rec = subpel_conv3x3(in_ch, out_ch, r=2)
        self.conv_prior = subpel_conv3x3(in_ch, prior_ch, r=2)

    def forward(self, input):
        out = self.layers(input)
        rec = self.conv_rec(out)
        prior = self.conv_prior(out)
        return rec, prior
        
class Decoder_P(nn.Module):
    def __init__(self, in_ch, out_ch, prior_ch):
        super().__init__()
        self.attn1 = SWin_Attention(dim=in_ch, num_heads=8, window_size=4)
        self.block1 = SynthesisBlock(in_ch, in_ch)
        self.block2_extra = AnalysisBlock(out_ch, in_ch)  # 处理 input2 的路径
        self.fusion = ResidualFusion(in_ch * 2, in_ch)  # 融合 out1 和 out2
        self.block2 = SynthesisBlock(in_ch, in_ch)
        self.attn2 = SWin_Attention(dim=in_ch, num_heads=8, window_size=8)
        self.block3 = SynthesisBlock(in_ch, in_ch)

        self.conv_rec = subpel_conv3x3(in_ch, out_ch, r=2)
        self.conv_prior1 = subpel_conv3x3(in_ch, prior_ch, r=2)
        self.conv_prior2 = subpel_conv3x3(prior_ch, prior_ch, r=2)
        self.conv_prior3 = subpel_conv3x3(prior_ch, prior_ch, r=2)
        

    def forward(self, input1, input2):
        input2=F.pixel_unshuffle(input2, 4)
        
        out = self.attn1(input1)

        out = self.block1(out)

        out = self.block2(out)

        out = self.attn2(out)

        out = self.block3(out)

        out2 = self.block2_extra(input2)

        fused = self.fusion(torch.cat([out, out2], dim=1))  # 特征融合
        rec = F.pixel_shuffle(self.conv_rec(fused),4)
        prior = self.conv_prior3(self.conv_prior2(self.conv_prior1(fused)))
        return rec, prior            

class HyperEncoder(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            downsample_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            downsample_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch)
        )
        
    def forward(self, input):
        out = self.layers(input)
        return out
    
    
class HyperDecoder(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers_mu = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
        )

        self.layers_sigma = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
        )
        
    def forward(self, input):
        mu = self.layers_mu(input)
        sigma = self.layers_sigma(input)
        return mu, sigma


class ResBlock_1x1(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        out = self.layers(input)
        out = out + input

        return out
    def run(self, input, tau):
        out = self.leaky_relu1(self.conv_cond1.run(input, tau))
        out = self.leaky_relu2(self.conv_cond2.run(out, tau))
        out = self.leaky_relu3(self.conv_cond3.run(out, tau))

        out = out + input

        return out  

class ResBlock_1x1_cond(nn.Module):
    def __init__(self, num_ch, cond_num, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_cond1 = Conv2d_cond(cond_num, num_ch, num_ch, kernel_size=1, bias=False)
        self.conv_cond2 = Conv2d_cond(cond_num, num_ch, num_ch, kernel_size=1, bias=False)
        self.conv_cond3 = Conv2d_cond(cond_num, num_ch, num_ch, kernel_size=1, bias=False)
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)
        self.leaky_relu2 = nn.LeakyReLU(inplace=True)
        self.leaky_relu3 = nn.LeakyReLU(inplace=True)


    def forward(self, input, cond):
        out = self.leaky_relu1(self.conv_cond1(input, cond))
        out = self.leaky_relu2(self.conv_cond2(out, cond))
        out = self.leaky_relu3(self.conv_cond3(out, cond))

        out = out + input

        return out
        
    def run(self, input, tau):
        out = self.leaky_relu1(self.conv_cond1.run(input, tau))
        out = self.leaky_relu2(self.conv_cond2.run(out, tau))
        out = self.leaky_relu3(self.conv_cond3.run(out, tau))

        out = out + input

        return out       
        
class LossyCompressor_I(CompressionModel):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(entropy_bottleneck_channels=num_ch, *args, **kwargs)
        self.encoder = Encoder_I(1, num_ch)
        self.decoder = Decoder_I(num_ch, 1, 256)
        self.hyperencoder = HyperEncoder(num_ch)
        self.hyperdecoder = HyperDecoder(num_ch)
        
        self.gaussian_conditional = GaussianConditional(None)
    
    def forward(self, input):
        y = self.encoder(input/nii_max_f)
        z = self.hyperencoder(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, sigma_hat, means=mu_hat)
        x_hat, res_prior = self.decoder(y_hat)
        x_hat = x_hat*nii_max_f
        
        return {
            "x_hat":x_hat,
            "res_prior":res_prior,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, input):
        y = self.encoder(input/nii_max_f)
        z = self.hyperencoder(y)

        z_strings = self.entropy_bottleneck.compress(z)

        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:], z.size()[:2])

        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        N, C, H, W = mu_hat.shape

        indexes = self.gaussian_conditional.build_indexes(sigma_hat.reshape(1, N*C, H, W))
        y_strings = self.gaussian_conditional.compress(y.reshape(1, N*C, H, W), indexes, means=mu_hat.reshape(1, N*C, H, W))

        return {"img_strings": [y_strings, z_strings], "shape": z.size()[-4:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[2:], shape[:2])

        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        N, C, H, W = mu_hat.shape

        indexes = self.gaussian_conditional.build_indexes(sigma_hat.reshape(1, N*C, H, W))
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=mu_hat.reshape(1, N*C, H, W))
        y_hat = y_hat.reshape(N, C, H, W)
        x_hat, res_prior = self.decoder(y_hat)
        x_hat = torch.floor(x_hat*nii_max_f + 0.5).clamp_(min=0, max=nii_max)

        return {"x_hat": x_hat, "res_prior": res_prior}


class ResidualCompressor(nn.Module):
    def __init__(self, num_ch, num_mixtures, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ResBlock = ResBlock_1x1(num_ch)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, 4*num_mixtures),
        )
         
    def forward(self, input):
        out = self.ResBlock(input)
        out = self.layers(out)
        
        return out


class ResidualCompressor_cond(nn.Module):
    def __init__(self, num_ch, num_mixtures, cond_num, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ResBlock_cond = ResBlock_1x1_cond(num_ch, cond_num)
        self.conv_cond1 = Conv2d_cond(cond_num, num_ch, num_ch, kernel_size=1, bias=False)
        self.conv_cond2 = Conv2d_cond(cond_num, num_ch, 4*num_mixtures, kernel_size=1, bias=False)
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)
         
    def forward(self, input, cond):
        out = self.ResBlock_cond(input, cond)
        out = self.leaky_relu1(self.conv_cond1(out, cond))
        out = self.conv_cond2(out, cond)
        
        return out
    def run(self, input, tau):
        out = self.ResBlock_cond.run(input, tau)
        out = self.leaky_relu1(self.conv_cond1.run(out, tau))
        out = self.conv_cond2.run(out, tau)
        
        return out
        
class LossyCompressor_P(CompressionModel):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(entropy_bottleneck_channels=num_ch, *args, **kwargs)
        self.encoder = Encoder_P(16, num_ch)
        self.decoder = Decoder_P(num_ch, 16, 256)
        self.hyperencoder = HyperEncoder(num_ch)
        self.hyperdecoder = HyperDecoder(num_ch)
        
        self.gaussian_conditional = GaussianConditional(None)
    
    def forward(self, input, pre):


        y = self.encoder(input/nii_max_f, pre/nii_max_f)
        z = self.hyperencoder(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, sigma_hat, means=mu_hat)
        x_hat, res_prior = self.decoder(y_hat, pre/nii_max_f)
        x_hat = x_hat*nii_max_f

        
        return {
            "x_hat":x_hat,
            "res_prior":res_prior,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }
    def compress(self, input, pre):
        y = self.encoder(input/nii_max_f,  pre/nii_max_f)
        z = self.hyperencoder(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:], z.size()[:2])

        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        N, C, H, W = mu_hat.shape

        indexes = self.gaussian_conditional.build_indexes(sigma_hat.reshape(1, N*C, H, W))
        y_strings = self.gaussian_conditional.compress(y.reshape(1, N*C, H, W), indexes, means=mu_hat.reshape(1, N*C, H, W))

        return {"img_strings": [y_strings, z_strings], "shape": z.size()[-4:]}

    def decompress(self, strings, shape, pre):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[2:], shape[:2])

        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        N, C, H, W = mu_hat.shape

        indexes = self.gaussian_conditional.build_indexes(sigma_hat.reshape(1, N*C, H, W))
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=mu_hat.reshape(1, N*C, H, W))
        y_hat = y_hat.reshape(N, C, H, W)
        x_hat, res_prior = self.decoder(y_hat,  pre/nii_max_f)
        x_hat = torch.floor(x_hat*nii_max_f + 0.5).clamp_(min=0, max=nii_max)

        return {"x_hat": x_hat, "res_prior": res_prior}
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


class NearLosslessCompressor_I(nn.Module):
    def __init__(self, num_ch, cond_num, mask_type="3P", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lossy_compressor = LossyCompressor_I(num_ch)
        self.mask_conv = maskedconv7x7_parallel(1, 256, mask_type)
        self.residual_compressor = ResidualCompressor(256, 5)
        self.fusion = conv1x1(256 * 2, 256)

        self.residual_compressor_cond = ResidualCompressor_cond(256, 5, cond_num)
        self.fusion_cond = Conv2d_cond(cond_num, 256 * 2, 256, kernel_size=1, bias=False)

        self.cond_num = cond_num


    def forward(self, input, tau):
        lossy_out = self.lossy_compressor(input)
        if self.training:
            noise = torch.empty_like(input).uniform_(-0.5, 0.5)
            x_hat = torch.clamp(lossy_out["x_hat"] + noise, min = 0., max = nii_max_f)
            rec_base = lossy_out["x_hat"]
        else:
            x_hat = torch.floor(lossy_out["x_hat"] + 0.5).clamp_(min=0, max=nii_max_f)
            rec_base = x_hat
        
        res = input - x_hat
        res_n = res/nii_max_f*2
        
        context = self.mask_conv(res_n)
        res_prior_context = self.fusion(torch.cat((lossy_out["res_prior"], context), 1))

        lmm_params = self.residual_compressor(res_prior_context)
        mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)

        lmm = LogisticMixtureModel(mu, log_sigma, weights, coeffs)
        
        log_res_likelihoods = lmm(res_n)

        # scalable nll

        res_q = torch.where(res>=0, (2*tau+1) * torch.floor((res+tau)/(2*tau+1)),
                            (2*tau+1) * torch.ceil((res-tau)/(2*tau+1)))
        context_q = self.mask_conv(res_q/nii_max_f*2)
        res_prior_context_bias_cat = torch.cat((lossy_out["res_prior"], context_q), 1).detach()
        
        batch_sz = input.shape[0]
        tau_list = torch.ones(batch_sz, dtype=torch.long, device=input.device) * (tau-1)
        cond = nn.functional.one_hot(tau_list, num_classes=self.cond_num).float()   

        res_prior_context_bias = self.fusion_cond(res_prior_context_bias_cat, cond)
        lmm_params_cond = self.residual_compressor_cond(res_prior_context_bias, cond)
        mu_cond, log_sigma_cond, coeffs_cond, weights_cond = torch.split(lmm_params_cond, 5, dim=1)
        lmm_model_cond = LogisticMixtureModel(mu_cond, log_sigma_cond, weights_cond, coeffs_cond)

        log_res_likelihoods_cond = lmm_model_cond(res_n.detach())

        return {
            "x_hat": x_hat,
            "res_likelihoods": log_res_likelihoods,
            "res_likelihoods_cond": log_res_likelihoods_cond,
            "img_likelihoods": lossy_out["likelihoods"],
            "rec_img":torch.clamp( ( rec_base + res_q), min = 0., max = nii_max_f)
        }


    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.lossy_compressor.gaussian_conditional,
            "lossy_compressor.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        return super().load_state_dict(state_dict, strict=strict)

    def parameters_wo_cond(self):
        parameters = (
            p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and not n.startswith("residual_compressor_cond.") and not n.startswith("fusion_cond.")
        )
    
        return parameters

    def parameters_w_cond(self):
        parameters = (
            p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and (n.startswith("residual_compressor_cond.") or n.startswith("fusion_cond."))
        )
    
        return parameters

    

class NearLosslessCompressor_P(nn.Module):
    def __init__(self, num_ch, cond_num, mask_type="3P", context_type="simple", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if context_type not in ("legacy", "simple", "diffusion_fusion"):
            raise ValueError(f"Unsupported context_type={context_type!r}.")
        self.context_type = context_type
        # self.lossy_compressor = LossyCompressor_P(num_ch)
        self.lossy_compressor = LossyCompressor_P_diff_sit(num_ch, embed_dim=32)
        self.mask_conv = maskedconv7x7_parallel(1, 256, mask_type)
        self.fusion = conv1x1(256 * 2, 256)
        if context_type == "legacy":
            self.residual_compressor = ResidualCompressor(256, 5)
        elif context_type == "simple":
            self.residual_compressor = DiffusionLMMHead(cond_ch=256, num_mixtures=5)
        else:
            self.residual_compressor = DiffusionContextLMMHead(
                prior_ch=256, context_ch=256, num_mixtures=5
            )

        self.residual_compressor_cond = ResidualCompressor_cond(256, 5, cond_num)
        self.fusion_cond = Conv2d_cond(cond_num, 256 * 2, 256, kernel_size=1, bias=False)

        self.cond_num = cond_num

    def _residual_context(self, res_norm, res_prior):
        if self.context_type in ("legacy", "simple"):
            context = self.mask_conv(res_norm)
            return self.fusion(torch.cat((res_prior, context), 1))
        raise RuntimeError(f"Unsupported context_type={self.context_type!r}")

    def _lmm_params_from_context(self, res_prior, residual_context):
        if self.context_type in ("legacy", "simple"):
            res_prior_context = self.fusion(torch.cat((res_prior, residual_context), 1))
            return self.residual_compressor(res_prior_context)
        if self.context_type == "diffusion_fusion":
            return self.residual_compressor(res_prior, residual_context)
        raise RuntimeError(f"Unsupported context_type={self.context_type!r}")


    def forward(self, input, pre, tau):

        lossy_out = self.lossy_compressor(input,pre)
        lossy_out["x_hat"]=patchify_to_batch(lossy_out["x_hat"],64)
        lossy_out["res_prior"]=patchify_to_batch(lossy_out["res_prior"],64)

        input = patchify_to_batch(input,64)
        if self.training:
            noise = torch.empty_like(input).uniform_(-0.5, 0.5)
            x_hat = torch.clamp(lossy_out["x_hat"] + noise, min = 0., max = nii_max_f)
            rec_base = lossy_out["x_hat"]
        else:
            x_hat = torch.floor(lossy_out["x_hat"] + 0.5).clamp_(min=0, max=nii_max_f)
            rec_base = x_hat
        
        res = input - x_hat
        res_n = res/nii_max_f*2
        
        context = self.mask_conv(res_n)
        lmm_params = self._lmm_params_from_context(lossy_out["res_prior"], context)
        mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)

        lmm = LogisticMixtureModel(mu, log_sigma, weights, coeffs)
        
        log_res_likelihoods = lmm(res_n)

        # scalable nll
        res_q = torch.where(res>=0, (2*tau+1) * torch.floor((res+tau)/(2*tau+1)),
                            (2*tau+1) * torch.ceil((res-tau)/(2*tau+1)))
        context_q = self.mask_conv(res_q/nii_max_f*2)
        res_prior_context_bias_cat = torch.cat((lossy_out["res_prior"], context_q), 1).detach()
        
        batch_sz = input.shape[0]
        tau_list = torch.ones(batch_sz, dtype=torch.long, device=input.device) * (tau-1)
        cond = nn.functional.one_hot(tau_list, num_classes=self.cond_num).float()   

        res_prior_context_bias = self.fusion_cond(res_prior_context_bias_cat, cond)
        lmm_params_cond = self.residual_compressor_cond(res_prior_context_bias, cond)
        mu_cond, log_sigma_cond, coeffs_cond, weights_cond = torch.split(lmm_params_cond, 5, dim=1)
        lmm_model_cond = LogisticMixtureModel(mu_cond, log_sigma_cond, weights_cond, coeffs_cond)

        log_res_likelihoods_cond = lmm_model_cond(res_n.detach())

        return {
            "x_hat": x_hat,
            "res_likelihoods": log_res_likelihoods,
            "res_likelihoods_cond": log_res_likelihoods_cond,
            "img_likelihoods": lossy_out["likelihoods"],
            "rec_img":torch.clamp( ( rec_base + res_q), min = 0., max = nii_max_f)
        }


    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.lossy_compressor.gaussian_conditional,
            "lossy_compressor.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        return super().load_state_dict(state_dict, strict=strict)

    def parameters_wo_cond(self):
        parameters = (
            p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles")
            and not n.startswith("residual_compressor_cond.")
            and not n.startswith("fusion_cond.")
            and not n.startswith("residual_compressor.teacher.")
        )

        return parameters

    def parameters_w_cond(self):
        parameters = (
            p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and (n.startswith("residual_compressor_cond.") or n.startswith("fusion_cond."))
        )

        return parameters


# ============================================================================
# CDC Diffusion Utilities & LossyCompressor_P_diff_sit (Eval Version)
# ============================================================================

import numpy as np
from functools import partial
from einops import rearrange


def _cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def _extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class CDC_LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class CDC_PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = CDC_LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class CDC_Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class CDC_Block(nn.Module):
    def __init__(self, dim, dim_out, large_filter=False):
        super().__init__()
        k = 7 if large_filter else 3
        p = 3 if large_filter else 1
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, k, padding=p),
            CDC_LayerNorm(dim_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class CDC_ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, large_filter=False):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None else None
        )
        self.block1 = CDC_Block(dim, dim_out, large_filter)
        self.block2 = CDC_Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if time_emb is not None and self.mlp is not None:
            h = h + self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class CDC_LinearAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=None):
        super().__init__()
        if dim_head is None:
            dim_head = dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class CDC_Upsample(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim_in
        self.conv = nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class CDC_Downsample(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim_in
        self.conv = nn.Conv2d(dim_in, dim_out, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# --- Encoder_P_CDC: reduced downsampling (H/16 instead of H/64) ---

class Encoder_P_CDC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block1 = AnalysisBlock(in_ch, out_ch)
        self.block2_extra = AnalysisBlock(in_ch, out_ch)
        self.fusion = ResidualFusion(out_ch * 2, out_ch)
        self.block2 = AnalysisBlock(out_ch, out_ch)
        self.attn1 = SWin_Attention(dim=out_ch, num_heads=8, window_size=8)

    def forward(self, input1, input2):
        input1 = F.pixel_unshuffle(input1, 4)
        input2 = F.pixel_unshuffle(input2, 4)
        out1 = self.block1(input1)
        out2 = self.block2_extra(input2)
        out = self.fusion(torch.cat([out1, out2], dim=1))
        out = self.block2(out)
        out = self.attn1(out)
        return out


# --- ContextDecoder: y_hat -> multi-scale features for U-Net ---

class ContextDecoder(nn.Module):
    def __init__(self, latent_ch, embed_dim=64):
        super().__init__()
        self.up3 = nn.Sequential(CDC_ResnetBlock(latent_ch, embed_dim * 4), CDC_Upsample(embed_dim * 4))
        self.up2 = nn.Sequential(CDC_ResnetBlock(embed_dim * 4, embed_dim * 3), CDC_Upsample(embed_dim * 3))
        self.up1 = nn.Sequential(CDC_ResnetBlock(embed_dim * 3, embed_dim * 2), CDC_Upsample(embed_dim * 2))
        self.up0 = nn.Sequential(CDC_ResnetBlock(embed_dim * 2, embed_dim), CDC_Upsample(embed_dim))

    def forward(self, y_hat):
        f3 = self.up3(y_hat)
        f2 = self.up2(f3)
        f1 = self.up1(f2)
        f0 = self.up0(f1)
        return [f0, f1, f2, f3]


# --- RefEncoder: reference frame -> multi-scale features ---

class RefEncoder(nn.Module):
    def __init__(self, in_ch, embed_dim=64):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, 7, padding=3),
            CDC_LayerNorm(embed_dim),
            nn.ReLU()
        )
        self.down1 = nn.Sequential(CDC_ResnetBlock(embed_dim, embed_dim * 2), CDC_Downsample(embed_dim * 2))
        self.down2 = nn.Sequential(CDC_ResnetBlock(embed_dim * 2, embed_dim * 3), CDC_Downsample(embed_dim * 3))
        self.down3 = nn.Sequential(CDC_ResnetBlock(embed_dim * 3, embed_dim * 4), CDC_Downsample(embed_dim * 4))

    def forward(self, ref):
        f0 = self.conv_in(ref)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        return [f0, f1, f2, f3]


# --- UNet_P: conditional diffusion U-Net for P-frame decoding ---

class UNet_P(nn.Module):
    def __init__(self, dim=64, channels=1, dim_mults=(1, 2, 3, 4),
                 context_dim_mults=(1, 2, 3, 4), ref_dim_mults=(1, 2, 3, 4)):
        super().__init__()
        self.channels = channels
        self.embd_type = "01"

        time_dim = dim
        self.time_mlp = nn.Sequential(nn.Linear(1, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

        dims = [channels, *[dim * m for m in dim_mults]]
        ctx_dims = [dim * m for m in context_dim_mults]
        ref_dims = [dim * m for m in ref_dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            first_ch = dim_in + ctx_dims[ind] + ref_dims[ind] if ind < len(ctx_dims) else dim_in
            self.downs.append(nn.ModuleList([
                CDC_ResnetBlock(first_ch, dim_out, time_dim, large_filter=(ind == 0)),
                CDC_ResnetBlock(dim_out, dim_out, time_dim),
                CDC_Residual(CDC_PreNorm(dim_out, CDC_LinearAttention(dim_out))),
                CDC_Downsample(dim_out) if not is_last else nn.Identity(),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = CDC_ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = CDC_Residual(CDC_PreNorm(mid_dim, CDC_LinearAttention(mid_dim)))
        self.mid_block2 = CDC_ResnetBlock(mid_dim, mid_dim, time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(nn.ModuleList([
                CDC_ResnetBlock(dim_out * 2, dim_in, time_dim),
                CDC_ResnetBlock(dim_in, dim_in, time_dim),
                CDC_Residual(CDC_PreNorm(dim_in, CDC_LinearAttention(dim_in))),
                CDC_Upsample(dim_in),
            ]))

        self.final_conv = nn.Sequential(CDC_LayerNorm(dim), nn.Conv2d(dim, channels, 7, padding=3))

    def forward(self, x, time, context, ref_context, return_features=False):
        t = self.time_mlp(time)

        h = []
        for idx, (b1, b2, attn, downsample) in enumerate(self.downs):
            if idx < len(context):
                x = torch.cat([x, context[idx], ref_context[idx]], dim=1)
            x = b1(x, t)
            x = b2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = b1(x, t)
            x = b2(x, t)
            x = attn(x)
            x = upsample(x)

        feat = x
        out = self.final_conv(x)
        if return_features:
            return out, feat
        return out


# --- ResPriorHead: U-Net features -> res_prior ---

class ResPriorHead(nn.Module):
    def __init__(self, embed_dim=64, out_ch=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, feat):
        return self.net(feat)


# --- LossyCompressor_P_diff_sit ---

class LossyCompressor_P_diff_sit(CompressionModel):
    def __init__(self, num_ch, embed_dim=64, num_timesteps=1000,
                 pred_mode='x', loss_weight_min=5.0, *args, **kwargs):
        super().__init__(entropy_bottleneck_channels=num_ch, *args, **kwargs)

        self.encoder = Encoder_P_CDC(16, num_ch)
        self.hyperencoder = HyperEncoder(num_ch)
        self.hyperdecoder = HyperDecoder(num_ch)
        self.gaussian_conditional = GaussianConditional(None)

        self.context_decoder = ContextDecoder(num_ch, embed_dim)
        self.ref_encoder = RefEncoder(1, embed_dim)
        self.denoise_fn = UNet_P(dim=embed_dim, channels=1)
        self.res_prior_head = ResPriorHead(embed_dim, 256)

        self.num_timesteps = num_timesteps
        self.pred_mode = pred_mode
        self.loss_weight_min = loss_weight_min

        to_torch = partial(torch.tensor, dtype=torch.float32)
        betas = _cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        self.register_buffer("train_sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("train_sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("train_snr", to_torch(alphas_cumprod / (1.0 - alphas_cumprod)))
        self.register_buffer("train_alphas_cumprod", to_torch(alphas_cumprod))

    def q_sample(self, x_start, t, noise):
        return (
            _extract(self.train_sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.train_sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0, alphas_cumprod):
        sqrt_recip = torch.sqrt(1.0 / _extract(alphas_cumprod, t, x_t.shape))
        sqrt_recipm1 = torch.sqrt(1.0 / _extract(alphas_cumprod, t, x_t.shape) - 1)
        return (sqrt_recip * x_t - x0) / sqrt_recipm1

    @torch.no_grad()
    def ddim_sample(self, shape, context, ref_context, sample_steps=20, eta=0):
        device = self.train_alphas_cumprod.device
        b = shape[0]

        if sample_steps != 1:
            indice = torch.linspace(0, self.num_timesteps - 1, sample_steps, device=device).long()
        else:
            indice = torch.tensor([self.num_timesteps - 1], device=device).long()

        alphas_cumprod = self.train_alphas_cumprod[indice]
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_alphas_cumprod_prev = torch.sqrt(alphas_cumprod_prev)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1.0 - alphas_cumprod_prev)
        sigma = sqrt_one_minus_alphas_cumprod_prev / sqrt_one_minus_alphas_cumprod * \
                torch.sqrt(1.0 - alphas_cumprod / alphas_cumprod_prev)

        img = torch.zeros(shape, device=device)
        feat = None

        for i in reversed(range(sample_steps)):
            time_val = indice[i].float().unsqueeze(0).unsqueeze(-1) / self.num_timesteps
            time_input = time_val.expand(b, -1)
            fx, feat = self.denoise_fn(img, time_input, context, ref_context, return_features=True)

            if self.pred_mode == 'x':
                x_recon = fx
            else:
                raise NotImplementedError

            x_recon = x_recon.clamp(-1.0, 1.0)
            noise = self.predict_noise_from_start(img, torch.full((b,), i, device=device, dtype=torch.long),
                                                   x_recon, alphas_cumprod)
            x_next = (
                _extract(sqrt_alphas_cumprod_prev, torch.full((b,), i, device=device, dtype=torch.long), img.shape) * x_recon
                + torch.sqrt(
                    (_extract(sqrt_one_minus_alphas_cumprod_prev ** 2,
                              torch.full((b,), i, device=device, dtype=torch.long), img.shape)
                     - (eta * _extract(sigma, torch.full((b,), i, device=device, dtype=torch.long), img.shape)) ** 2).clamp(min=0)
                ) * noise
            )
            img = x_next

        return img, feat

    def forward(self, input, pre):
        x = input / nii_max_f * 2 - 1
        ref = pre / nii_max_f * 2 - 1

        y = self.encoder(input / nii_max_f, pre / nii_max_f)
        z = self.hyperencoder(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, sigma_hat, means=mu_hat)

        context_feats = self.context_decoder(y_hat)
        ref_feats = self.ref_encoder(ref)

        if self.training:
            B = x.shape[0]
            t = torch.randint(0, self.num_timesteps, (B,), device=x.device).long()
            noise = torch.randn_like(x)
            x_noisy = self.q_sample(x, t, noise)

            time_input = t.float().unsqueeze(-1) / self.num_timesteps
            x_pred, feat = self.denoise_fn(x_noisy, time_input, context_feats, ref_feats, return_features=True)

            if self.pred_mode == 'x':
                snr = self.train_snr[t]
                weight = snr.clamp(max=self.loss_weight_min)
                diff_loss = (weight * F.mse_loss(x_pred, x, reduction='none').mean(dim=(1, 2, 3))).mean()
            else:
                diff_loss = F.mse_loss(x_pred, x)

            res_prior = self.res_prior_head(feat)
            x_hat = ((x_pred.clamp(-1, 1) + 1) / 2) * nii_max_f

            return {
                "x_hat": x_hat,
                "res_prior": res_prior,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                "diffusion_loss": diff_loss,
            }
        else:
            x_hat, feat = self.ddim_sample(x.shape, context_feats, ref_feats, sample_steps=EVAL_DDIM_STEPS)
            res_prior = self.res_prior_head(feat)
            x_hat = ((x_hat.clamp(-1, 1) + 1) / 2) * nii_max_f

            return {
                "x_hat": x_hat,
                "res_prior": res_prior,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }

    def compress(self, input, pre):
        y = self.encoder(input / nii_max_f, pre / nii_max_f)
        z = self.hyperencoder(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:], z.size()[:2])

        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        N, C, H, W = mu_hat.shape

        indexes = self.gaussian_conditional.build_indexes(sigma_hat.reshape(1, N * C, H, W))
        y_strings = self.gaussian_conditional.compress(
            y.reshape(1, N * C, H, W), indexes, means=mu_hat.reshape(1, N * C, H, W)
        )

        return {"img_strings": [y_strings, z_strings], "shape": z.size()[-4:]}

    def decompress(self, strings, shape, pre, sample_steps=None):
        if sample_steps is None:
            sample_steps = EVAL_DDIM_STEPS
        ref = pre / nii_max_f * 2 - 1

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[2:], shape[:2])
        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        N, C, H, W = mu_hat.shape

        indexes = self.gaussian_conditional.build_indexes(sigma_hat.reshape(1, N * C, H, W))
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=mu_hat.reshape(1, N * C, H, W)
        )
        y_hat = y_hat.reshape(N, C, H, W)

        context_feats = self.context_decoder(y_hat)
        ref_feats = self.ref_encoder(ref)

        x_hat, feat = self.ddim_sample(
            (N, 1, pre.shape[2], pre.shape[3]), context_feats, ref_feats, sample_steps=sample_steps
        )
        res_prior = self.res_prior_head(feat)
        x_hat = ((x_hat.clamp(-1, 1) + 1) / 2) * nii_max_f
        x_hat = torch.floor(x_hat + 0.5).clamp_(min=0, max=nii_max)

        return {"x_hat": x_hat, "res_prior": res_prior}

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


# ============================================================================
# DiffusionLMMHead: condition-diffusion replacement for ResidualCompressor
# ============================================================================


class LMM_UNet(nn.Module):
    """Lightweight conditional U-Net inside DiffusionLMMHead.

    Predicts clean LMM-parameter tensor (x-prediction) from a noised
    parameter tensor and a conditioning feature map.
    """

    def __init__(self, params_ch=20, cond_ch=256, base_dim=64):
        super().__init__()
        time_dim = base_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_dim * 4),
            nn.GELU(),
            nn.Linear(base_dim * 4, time_dim),
        )

        self.conv_in = nn.Conv2d(params_ch + cond_ch, base_dim, 3, padding=1)

        self.down1 = CDC_ResnetBlock(base_dim, base_dim, time_dim)
        self.down1_sample = CDC_Downsample(base_dim, base_dim * 2)
        self.down2 = CDC_ResnetBlock(base_dim * 2, base_dim * 2, time_dim)
        self.down2_sample = CDC_Downsample(base_dim * 2, base_dim * 4)

        self.mid_block1 = CDC_ResnetBlock(base_dim * 4, base_dim * 4, time_dim)
        self.mid_block2 = CDC_ResnetBlock(base_dim * 4, base_dim * 4, time_dim)

        self.up2_sample = CDC_Upsample(base_dim * 4, base_dim * 2)
        self.up2 = CDC_ResnetBlock(base_dim * 2 * 2, base_dim * 2, time_dim)
        self.up1_sample = CDC_Upsample(base_dim * 2, base_dim)
        self.up1 = CDC_ResnetBlock(base_dim * 2, base_dim, time_dim)

        self.conv_out = nn.Sequential(
            CDC_LayerNorm(base_dim),
            nn.Conv2d(base_dim, params_ch, 3, padding=1),
        )

    def forward(self, params_t, time, cond):
        H, W = params_t.shape[-2:]
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h or pad_w:
            params_t = F.pad(params_t, (0, pad_w, 0, pad_h), mode='replicate')
            cond = F.pad(cond, (0, pad_w, 0, pad_h), mode='replicate')

        t = self.time_mlp(time)

        x = torch.cat([params_t, cond], dim=1)
        x = self.conv_in(x)

        d1 = self.down1(x, t)
        x = self.down1_sample(d1)
        d2 = self.down2(x, t)
        x = self.down2_sample(d2)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        x = self.up2_sample(x)
        x = torch.cat([x, d2], dim=1)
        x = self.up2(x, t)
        x = self.up1_sample(x)
        x = torch.cat([x, d1], dim=1)
        x = self.up1(x, t)

        out = self.conv_out(x)
        if pad_h or pad_w:
            out = out[..., :H, :W]
        return out


class DiffusionLMMHead(nn.Module):
    """Drop-in replacement for ResidualCompressor (eval mirror).

    Generates LMM parameters by running K-step DDIM reverse sampling
    conditioned on res_prior_context. Public interface
    (forward(cond) -> [B, 4*num_mixtures, H, W]) matches ResidualCompressor.

    bit-exact: eta=0, deterministic-seed start noise, fp32 throughout.
    """

    def __init__(self, cond_ch=256, num_mixtures=5, num_timesteps=200,
                 sample_steps=3, base_dim=64, noise_seed=20260526,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params_ch = 4 * num_mixtures
        self.num_timesteps = num_timesteps
        self.sample_steps = sample_steps
        self.noise_seed = noise_seed

        self.unet = LMM_UNet(
            params_ch=self.params_ch,
            cond_ch=cond_ch,
            base_dim=base_dim,
        )

        self.teacher = ResidualCompressor(cond_ch, num_mixtures)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self._teacher_enabled = True

        to_torch = partial(torch.tensor, dtype=torch.float32)
        betas = _cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             to_torch(np.sqrt(1.0 - alphas_cumprod)))

        self.last_diff_loss = None

    def disable_teacher(self):
        self._teacher_enabled = False

    def enable_teacher(self):
        self._teacher_enabled = True

    def load_teacher(self, teacher_state_dict):
        self.teacher.load_state_dict(teacher_state_dict, strict=True)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def _q_sample(self, x_start, t, noise):
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _deterministic_noise(self, shape, device):
        gen = torch.Generator(device=device)
        gen.manual_seed(self.noise_seed)
        return torch.randn(shape, generator=gen, device=device, dtype=torch.float32)

    def _ddim_reverse(self, cond):
        B, _, H, W = cond.shape
        device = cond.device
        shape = (B, self.params_ch, H, W)

        sample_steps = self.sample_steps
        if sample_steps != 1:
            indice = torch.linspace(0, self.num_timesteps - 1, sample_steps,
                                    device=device).long()
        else:
            indice = torch.tensor([self.num_timesteps - 1], device=device).long()

        alphas_cumprod = self.alphas_cumprod[indice]
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_alphas_cumprod_prev = torch.sqrt(alphas_cumprod_prev)
        sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1.0 - alphas_cumprod_prev)

        params_t = self._deterministic_noise(shape, device)

        last_step = 0
        for i in reversed(range(sample_steps)):
            use_grad = self.training and (i == last_step)
            grad_ctx = nullcontext() if use_grad else torch.no_grad()
            with grad_ctx:
                t_idx = torch.full((B,), i, device=device, dtype=torch.long)
                time_val = indice[i].float().unsqueeze(0).unsqueeze(-1) / self.num_timesteps
                time_input = time_val.expand(B, -1)

                params_pred = self.unet(params_t, time_input, cond)

                sqrt_recip = torch.sqrt(1.0 / _extract(alphas_cumprod, t_idx, params_t.shape))
                sqrt_recipm1 = torch.sqrt(1.0 / _extract(alphas_cumprod, t_idx, params_t.shape) - 1)
                noise_pred = (sqrt_recip * params_t - params_pred) / sqrt_recipm1

                params_t = (
                    _extract(sqrt_alphas_cumprod_prev, t_idx, params_t.shape) * params_pred
                    + _extract(sqrt_one_minus_alphas_cumprod_prev, t_idx, params_t.shape) * noise_pred
                )
            if not use_grad:
                params_t = params_t.detach()

        return params_t

    def forward(self, cond):
        if self.training and self._teacher_enabled:
            with torch.no_grad():
                gt_params = self.teacher(cond)
            B = cond.shape[0]
            t = torch.randint(0, self.num_timesteps, (B,), device=cond.device).long()
            noise = torch.randn_like(gt_params)
            params_noisy = self._q_sample(gt_params, t, noise)
            time_input = t.float().unsqueeze(-1) / self.num_timesteps
            params_pred = self.unet(params_noisy, time_input, cond)
            self.last_diff_loss = F.mse_loss(params_pred, gt_params)
            return gt_params.detach()

        params_0 = self._ddim_reverse(cond)
        self.last_diff_loss = None
        return params_0


class SimpleContextLMMTeacher(nn.Module):
    """Simple causal context path used as the teacher for diffusion fusion."""

    def __init__(self, prior_ch=256, context_ch=256, hidden_ch=256, num_mixtures=5):
        super().__init__()
        self.fusion = conv1x1(prior_ch + context_ch, hidden_ch)
        self.residual_compressor = ResidualCompressor(hidden_ch, num_mixtures)

    def forward(self, cond):
        fused = self.fusion(cond)
        return self.residual_compressor(fused)


class PointwiseLMM_UNet(nn.Module):
    """Pointwise diffusion denoiser for causal context fusion.

    This uses only 1x1 convolutions, so each output pixel depends only on the
    same pixel's lossy prior and already-causal residual context.
    """

    def __init__(self, params_ch=20, cond_ch=512, base_dim=64, depth=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_dim * 4),
            nn.GELU(),
            nn.Linear(base_dim * 4, base_dim),
        )
        self.conv_in = nn.Conv2d(params_ch + cond_ch, base_dim, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_dim, base_dim, 1),
                nn.GELU(),
                nn.Conv2d(base_dim, base_dim, 1),
            )
            for _ in range(depth)
        ])
        self.time_proj = nn.ModuleList([
            nn.Linear(base_dim, base_dim)
            for _ in range(depth)
        ])
        self.conv_out = nn.Conv2d(base_dim, params_ch, 1)

    def forward(self, params_t, time, cond):
        t = self.time_mlp(time)
        x = self.conv_in(torch.cat([params_t, cond], dim=1))
        for block, proj in zip(self.blocks, self.time_proj):
            h = x + proj(t).unsqueeze(-1).unsqueeze(-1)
            x = x + block(h)
        return self.conv_out(x)


class DiffusionContextLMMHead(DiffusionLMMHead):
    """Diffusion LMM head conditioned on lossy prior and causal residual context."""

    def __init__(self, prior_ch=256, context_ch=256, num_mixtures=5,
                 num_timesteps=200, sample_steps=3, base_dim=64,
                 noise_seed=20260526, *args, **kwargs):
        self.prior_ch = prior_ch
        self.context_ch = context_ch
        super().__init__(
            cond_ch=prior_ch + context_ch,
            num_mixtures=num_mixtures,
            num_timesteps=num_timesteps,
            sample_steps=sample_steps,
            base_dim=base_dim,
            noise_seed=noise_seed,
            *args,
            **kwargs,
        )
        self.unet = PointwiseLMM_UNet(
            params_ch=self.params_ch,
            cond_ch=prior_ch + context_ch,
            base_dim=base_dim,
        )
        self.teacher = SimpleContextLMMTeacher(prior_ch, context_ch, 256, num_mixtures)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def load_teacher(self, fusion_state_dict, residual_state_dict):
        self.teacher.fusion.load_state_dict(fusion_state_dict, strict=True)
        self.teacher.residual_compressor.load_state_dict(residual_state_dict, strict=True)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def forward(self, res_prior, residual_context):
        cond = torch.cat((res_prior, residual_context), 1)
        return super().forward(cond)
