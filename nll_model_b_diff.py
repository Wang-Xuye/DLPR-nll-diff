import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import GaussianConditional
from compressai.models import CompressionModel

from global_nii_max import nii_max, nii_max_f
from nll_model_b import (
    AnalysisBlock,
    HyperDecoder,
    HyperEncoder,
    NearLosslessCompressor_I,
    NearLosslessCompressor_P as BaseNearLosslessCompressor_P,
    ResidualFusion,
    get_scale_table,
    patchify_to_batch,
    unpatchify_from_batch,
)
from nll_model_p import (
    CDC_ResnetBlock,
    CDC_Upsample,
    ContextDecoder,
    RefEncoder,
    ResPriorHead,
    SWin_Attention,
    UNet_P,
    _cosine_beta_schedule,
    _extract,
)
import numpy as np
from functools import partial


class Encoder_B_CDC(nn.Module):
    """CDC encoder for static B frames with two reconstructed references."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block_cur = AnalysisBlock(in_ch, out_ch)
        self.block_ref_l = AnalysisBlock(in_ch, out_ch)
        self.block_ref_r = AnalysisBlock(in_ch, out_ch)
        self.fusion = ResidualFusion(out_ch * 3, out_ch)
        self.block2 = AnalysisBlock(out_ch, out_ch)
        self.attn1 = SWin_Attention(dim=out_ch, num_heads=8, window_size=8)

    def forward(self, current, ref_l, ref_r):
        current = F.pixel_unshuffle(current, 4)
        ref_l = F.pixel_unshuffle(ref_l, 4)
        ref_r = F.pixel_unshuffle(ref_r, 4)
        cur_feat = self.block_cur(current)
        ref_l_feat = self.block_ref_l(ref_l)
        ref_r_feat = self.block_ref_r(ref_r)
        out = self.fusion(torch.cat([cur_feat, ref_l_feat, ref_r_feat], dim=1))
        out = self.block2(out)
        out = self.attn1(out)
        return out


class DualRefEncoder(nn.Module):
    """Encode left/right reconstructed references as one diffusion condition."""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.encoder = RefEncoder(2, embed_dim)

    def forward(self, ref_l, ref_r):
        return self.encoder(torch.cat([ref_l, ref_r], dim=1))


class LossyCompressor_B_diff_sit(CompressionModel):
    """Diff-SIT lossy compressor adapted from P-frame to static B-frame refs."""

    def __init__(
        self,
        num_ch,
        embed_dim=64,
        num_timesteps=1000,
        pred_mode="x",
        loss_weight_min=5.0,
        sample_steps=5,
        *args,
        **kwargs,
    ):
        super().__init__(entropy_bottleneck_channels=num_ch, *args, **kwargs)

        self.encoder = Encoder_B_CDC(16, num_ch)
        self.hyperencoder = HyperEncoder(num_ch)
        self.hyperdecoder = HyperDecoder(num_ch)
        self.gaussian_conditional = GaussianConditional(None)

        self.context_decoder = ContextDecoder(num_ch, embed_dim)
        self.ref_encoder = DualRefEncoder(embed_dim)
        self.denoise_fn = UNet_P(dim=embed_dim, channels=1)
        self.res_prior_head = ResPriorHead(embed_dim, 256)

        self.num_timesteps = num_timesteps
        self.pred_mode = pred_mode
        self.loss_weight_min = loss_weight_min
        self.sample_steps = sample_steps

        to_torch = partial(torch.tensor, dtype=torch.float32)
        betas = _cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        self.register_buffer("train_sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "train_sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod)),
        )
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
    def ddim_sample(self, shape, context, ref_context, sample_steps=None, eta=0):
        sample_steps = sample_steps or self.sample_steps
        device = self.train_alphas_cumprod.device
        batch = shape[0]

        if sample_steps != 1:
            indice = torch.linspace(0, self.num_timesteps - 1, sample_steps, device=device).long()
        else:
            indice = torch.tensor([self.num_timesteps - 1], device=device).long()

        alphas_cumprod = self.train_alphas_cumprod[indice]
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_alphas_cumprod_prev = torch.sqrt(alphas_cumprod_prev)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1.0 - alphas_cumprod_prev)
        sigma = (
            sqrt_one_minus_alphas_cumprod_prev
            / sqrt_one_minus_alphas_cumprod
            * torch.sqrt(1.0 - alphas_cumprod / alphas_cumprod_prev)
        )

        img = torch.zeros(shape, device=device)
        feat = None
        for i in reversed(range(sample_steps)):
            step = torch.full((batch,), i, device=device, dtype=torch.long)
            time_val = indice[i].float().unsqueeze(0).unsqueeze(-1) / self.num_timesteps
            time_input = time_val.expand(batch, -1)
            fx, feat = self.denoise_fn(img, time_input, context, ref_context, return_features=True)

            if self.pred_mode != "x":
                raise NotImplementedError
            x_recon = fx.clamp(-1.0, 1.0)
            noise = self.predict_noise_from_start(img, step, x_recon, alphas_cumprod)
            x_next = (
                _extract(sqrt_alphas_cumprod_prev, step, img.shape) * x_recon
                + torch.sqrt(
                    (
                        _extract(sqrt_one_minus_alphas_cumprod_prev ** 2, step, img.shape)
                        - (eta * _extract(sigma, step, img.shape)) ** 2
                    ).clamp(min=0)
                )
                * noise
            )
            img = x_next

        return img, feat

    def _entropy_forward(self, input, ref_l, ref_r):
        y = self.encoder(input / nii_max_f, ref_l / nii_max_f, ref_r / nii_max_f)
        z = self.hyperencoder(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, sigma_hat, means=mu_hat)
        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def forward(self, input, ref_l, ref_r):
        x = input / nii_max_f * 2 - 1
        ref_l_norm = ref_l / nii_max_f * 2 - 1
        ref_r_norm = ref_r / nii_max_f * 2 - 1

        y_hat, likelihoods = self._entropy_forward(input, ref_l, ref_r)
        context_feats = self.context_decoder(y_hat)
        ref_feats = self.ref_encoder(ref_l_norm, ref_r_norm)

        if self.training:
            batch = x.shape[0]
            t = torch.randint(0, self.num_timesteps, (batch,), device=x.device).long()
            noise = torch.randn_like(x)
            x_noisy = self.q_sample(x, t, noise)
            time_input = t.float().unsqueeze(-1) / self.num_timesteps
            x_pred, feat = self.denoise_fn(
                x_noisy,
                time_input,
                context_feats,
                ref_feats,
                return_features=True,
            )

            if self.pred_mode == "x":
                snr = self.train_snr[t]
                weight = snr.clamp(max=self.loss_weight_min)
                diff_loss = (
                    weight
                    * F.mse_loss(x_pred, x, reduction="none").mean(dim=(1, 2, 3))
                ).mean()
            else:
                diff_loss = F.mse_loss(x_pred, x)

            res_prior = self.res_prior_head(feat)
            x_hat = ((x_pred.clamp(-1, 1) + 1) / 2) * nii_max_f
            return {
                "x_hat": x_hat,
                "res_prior": res_prior,
                "likelihoods": likelihoods,
                "diffusion_loss": diff_loss,
            }

        x_hat, feat = self.ddim_sample(x.shape, context_feats, ref_feats)
        res_prior = self.res_prior_head(feat)
        x_hat = ((x_hat.clamp(-1, 1) + 1) / 2) * nii_max_f
        return {"x_hat": x_hat, "res_prior": res_prior, "likelihoods": likelihoods}

    def compress(self, input, ref_l, ref_r):
        y = self.encoder(input / nii_max_f, ref_l / nii_max_f, ref_r / nii_max_f)
        z = self.hyperencoder(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        n, c, h, w = mu_hat.shape
        indexes = self.gaussian_conditional.build_indexes(sigma_hat.reshape(1, n * c, h, w))
        y_strings = self.gaussian_conditional.compress(
            y.reshape(1, n * c, h, w),
            indexes,
            means=mu_hat.reshape(1, n * c, h, w),
        )
        return {"img_strings": [y_strings, z_strings], "shape": z.size()[-4:]}

    def decompress(self, strings, shape, ref_l, ref_r):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[2:])

        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        n, c, h, w = mu_hat.shape
        indexes = self.gaussian_conditional.build_indexes(sigma_hat.reshape(1, n * c, h, w))
        y_hat = self.gaussian_conditional.decompress(
            strings[0],
            indexes,
            means=mu_hat.reshape(1, n * c, h, w),
        )
        y_hat = y_hat.reshape(n, c, h, w)

        context_feats = self.context_decoder(y_hat)
        ref_l_norm = ref_l / nii_max_f * 2 - 1
        ref_r_norm = ref_r / nii_max_f * 2 - 1
        ref_feats = self.ref_encoder(ref_l_norm, ref_r_norm)
        out_shape = (ref_l.shape[0], 1, ref_l.shape[-2], ref_l.shape[-1])
        x_hat, feat = self.ddim_sample(out_shape, context_feats, ref_feats)
        res_prior = self.res_prior_head(feat)
        x_hat = torch.floor(((x_hat.clamp(-1, 1) + 1) / 2) * nii_max_f + 0.5)
        x_hat = x_hat.clamp_(min=0, max=nii_max)
        return {"x_hat": x_hat, "res_prior": res_prior}

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


class NearLosslessCompressor_P(BaseNearLosslessCompressor_P):
    """Static B-frame near-lossless model with diffusion lossy compressor."""

    def __init__(self, num_ch, cond_num, mask_type="3P", *args, **kwargs):
        super().__init__(num_ch, cond_num, mask_type, *args, **kwargs)
        self.lossy_compressor = LossyCompressor_B_diff_sit(num_ch)

    def forward(self, input, pre, nxt, tau):
        lossy_out = self.lossy_compressor(input, pre, nxt)
        lossy_out["x_hat"] = patchify_to_batch(lossy_out["x_hat"], 64)
        lossy_out["res_prior"] = patchify_to_batch(lossy_out["res_prior"], 64)

        input = patchify_to_batch(input, 64)
        noise = torch.empty_like(input).uniform_(-0.5, 0.5)

        x_hat = torch.clamp(lossy_out["x_hat"] + noise, min=0.0, max=nii_max_f)
        res = input - x_hat
        res_n = res / nii_max_f * 2

        context = self.mask_conv(res_n)
        res_prior_context = self.fusion(torch.cat((lossy_out["res_prior"], context), 1))

        lmm_params = self.residual_compressor(res_prior_context)
        mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)
        lmm = self._build_lmm(mu, log_sigma, coeffs, weights)
        log_res_likelihoods = lmm(res_n)

        res_q = torch.where(
            res >= 0,
            (2 * tau + 1) * torch.floor((res + tau) / (2 * tau + 1)),
            (2 * tau + 1) * torch.ceil((res - tau) / (2 * tau + 1)),
        )
        context_q = self.mask_conv(res_q / nii_max_f * 2)
        res_prior_context_bias_cat = torch.cat(
            (lossy_out["res_prior"], context_q),
            1,
        ).detach()

        batch_sz = input.shape[0]
        tau_list = torch.ones(batch_sz, dtype=torch.long, device=input.device) * (tau - 1)
        cond = nn.functional.one_hot(tau_list, num_classes=self.cond_num).float()

        res_prior_context_bias = self.fusion_cond(res_prior_context_bias_cat, cond)
        lmm_params_cond = self.residual_compressor_cond(res_prior_context_bias, cond)
        mu_cond, log_sigma_cond, coeffs_cond, weights_cond = torch.split(
            lmm_params_cond,
            5,
            dim=1,
        )
        lmm_model_cond = self._build_lmm(mu_cond, log_sigma_cond, coeffs_cond, weights_cond)
        log_res_likelihoods_cond = lmm_model_cond(res_n.detach())

        out = {
            "x_hat": lossy_out["x_hat"],
            "res_likelihoods": log_res_likelihoods,
            "res_likelihoods_cond": log_res_likelihoods_cond,
            "img_likelihoods": lossy_out["likelihoods"],
            "rec_img": torch.clamp(lossy_out["x_hat"] + res_q, min=0.0, max=nii_max_f),
        }
        if "diffusion_loss" in lossy_out:
            out["diffusion_loss"] = lossy_out["diffusion_loss"]
        return out

    @staticmethod
    def _build_lmm(mu, log_sigma, coeffs, weights):
        from logisticmixturemodel import LogisticMixtureModel

        return LogisticMixtureModel(mu, log_sigma, weights, coeffs)
