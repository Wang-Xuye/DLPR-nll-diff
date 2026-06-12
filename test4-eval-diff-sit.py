import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import re
from nll_model_eval_p import NearLosslessCompressor_I, NearLosslessCompressor_P, unpatchify_from_batch, patchify_to_batch

from torchmetrics import PeakSignalNoiseRatio

import time
from PIL import Image
import numpy as np
import time
from tqdm import tqdm
import torchac
from global_nii_max import nii_max, nii_max_f
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============ 路径配置（按需修改） ============
DATA_PATH = "../Mosmed/test-full"
I_FRAME_CKP = "ckp_nll/0001/ckp.tar"
P_FRAME_CKP = "log4_diff/p-Ep30-0.000-0.000-0.000.pth"  # 修改为你的P帧checkpoint
LOG_FILE_PREFIX = "log-diff-sit"
# =============================================

enclossytime = 0
encrestime = 0
declossytime = 0
decrestime = 0


def replicate_pad(x, p):
    height = x.shape[-2]
    width = x.shape[-1]
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_right = new_w - width
    padding_bottom = new_h - height
    if padding_right == 0 and padding_bottom == 0:
        return x
    return np.pad(x, [(0, 0), (0, padding_bottom), (0, padding_right)], mode='edge')


class Dataset():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.clips = []
        for root, _, files in os.walk(self.root_dir):
            for filename in sorted(files):
                full_path = os.path.join(root, filename)
                self.clips.append(full_path)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip_path = self.clips[index]
        clip = np.load(clip_path).astype(np.float32).transpose((2, 0, 1))
        clip = replicate_pad(clip, 128)
        print(clip_path, clip.shape)
        return clip


def coding_order_table7x7(patch_sz=64, mask_type="3P"):
    if mask_type not in ("5P", "4P", "3P", "2P", "P"):
        raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    COT = torch.zeros(patch_sz, patch_sz, dtype=torch.int64)

    if mask_type == "5P":
        for i in range(patch_sz):
            start = 4 * i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
    elif mask_type == "4P":
        for i in range(patch_sz):
            start = 3 * i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
    elif mask_type == "3P":
        for i in range(patch_sz):
            start = 2 * i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
    elif mask_type == "2P":
        for i in range(patch_sz):
            start = i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
    elif mask_type == "P":
        for i in range(patch_sz):
            start = 1
            COT[i, :] = torch.arange(start, start + patch_sz)

    return COT


def img2patch(img, patch_sz):
    if len(img.shape) == 2:
        img = img.unsqueeze(2)
    h, w, _ = img.shape
    h_num = h // patch_sz - (patch_sz - (h % patch_sz)) // patch_sz + 1
    w_num = w // patch_sz - (patch_sz - (w % patch_sz)) // patch_sz + 1
    img_pad = nn.functional.pad(img, (0, 0, (patch_sz - w % patch_sz) % patch_sz, 0, (patch_sz - h % patch_sz) % patch_sz, 0))
    patch_h = torch.chunk(img_pad, h_num, dim=0)
    patch_h = torch.stack(patch_h, dim=0)
    patch_w = torch.chunk(patch_h, w_num, dim=2)
    patch = torch.cat(patch_w, dim=0)
    patch = patch.permute(0, 3, 1, 2)
    return patch


def patch2img(patch, img_sz):
    h, w = img_sz
    patch_sz = patch.shape[2]
    h_num = h // patch_sz - (patch_sz - (h % patch_sz)) // patch_sz + 1
    w_num = w // patch_sz - (patch_sz - (w % patch_sz)) // patch_sz + 1
    patch = patch.permute(0, 2, 3, 1)
    patch_w = torch.chunk(patch, w_num, dim=0)
    patch_h = torch.cat(patch_w, dim=2)
    patch_h = torch.chunk(patch_h, h_num, dim=0)
    img = torch.cat(patch_h, dim=1).squeeze(0)
    return img[-h:, -w:, :]


def compress(model, input_img, COT, tau=0, patch_sz=64, mix_num=5, last_frame=None):
    global enclossytime
    global encrestime
    norm_scale = 1 / nii_max_f * 2
    half = (0.5 + tau) * norm_scale
    code_res = []
    img_shape = input_img.shape[:2]
    bin_sz = 2 * tau + 1
    samples_end = (nii_max // bin_sz) * bin_sz

    x = None
    device = next(model.parameters()).device
    if last_frame is None:
        x = img2patch(torch.as_tensor(input_img), patch_sz=patch_sz).to(device)
    else:
        x = torch.as_tensor(input_img).unsqueeze(0).unsqueeze(0).to(device)
        last_frame = torch.as_tensor(last_frame).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        time_start = time.time()

        if last_frame is None:
            code_lossy = model.lossy_compressor.compress(x)
            rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"])
        else:
            code_lossy = model.lossy_compressor.compress(x, last_frame)
            rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"], last_frame)
        time_end_ls = time.time()
        if last_frame is not None:
            rec_lossy["x_hat"] = patchify_to_batch(rec_lossy["x_hat"], 64)
            rec_lossy["res_prior"] = patchify_to_batch(rec_lossy["res_prior"], 64)
            x = patchify_to_batch(x, 64)
        psnr = PeakSignalNoiseRatio()

        x_hat = rec_lossy["x_hat"]
        res_prior = rec_lossy["res_prior"]
        res = x - x_hat
        psnr_rrr = psnr(x.detach().cpu(), x_hat.detach().cpu())

        res_q = torch.where(res >= 0, (2 * tau + 1) * torch.floor((res + tau) / (2 * tau + 1)),
                            (2 * tau + 1) * torch.ceil((res - tau) / (2 * tau + 1)))

        res_q_max = int(torch.max(res_q).item())
        res_q_min = int(torch.min(res_q).item())

        res_q_max_norm = res_q_max * norm_scale
        res_q_min_norm = res_q_min * norm_scale

        res_q_max_idx = (res_q_max + samples_end) // bin_sz
        res_q_min_idx = (res_q_min + samples_end) // bin_sz
        print("Encode Res range:[{}({}),{}({})], elems num:{}".format(res_q_min, res_q_min_idx, res_q_max, res_q_max_idx, res_q_max_idx - res_q_min_idx + 1))

        samples = torch.arange(res_q_min, res_q_max + 1, step=bin_sz, dtype=torch.float32).to(device)
        samples = samples * norm_scale

        res_tmp = torch.zeros_like(x_hat)

        ctx_total = model.mask_conv(res_q * norm_scale)

        max_step = torch.max(COT)
        print(res.shape, max_step)
        for i in range(max_step):

            h_idx, w_idx = torch.nonzero(COT == i + 1, as_tuple=True)
            ctx = ctx_total[:, :, h_idx, w_idx].unsqueeze(3)
            rp = res_prior[:, :, h_idx, w_idx].unsqueeze(3)

            res_crop = res_q[:, :, h_idx, w_idx]
            res_tmp_crop = res_tmp[:, :, h_idx, w_idx]
            res_tmp_crop = res_tmp_crop.unsqueeze(3)

            if tau == 0:
                lmm_params = lmm_params_from_context(model, rp, ctx)
                mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)
                coeffs = torch.tanh(coeffs)
            elif tau > 0:
                if tau > 5:
                    tau = 5
                rp_ctx = model.fusion_cond.run(torch.cat((rp, ctx), 1), tau - 1)
                lmm_params = model.residual_compressor_cond.run(rp_ctx, tau - 1)
                mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)
                coeffs = torch.tanh(coeffs)

            for c in range(1):
                mu_c = mu.permute(0, 2, 1, 3)
                samples_centered = samples - mu_c
                inv_sigma = torch.exp(-log_sigma).permute(0, 2, 1, 3)
                plus_in = inv_sigma * (samples_centered + half)
                cdf_plus = torch.sigmoid(plus_in)
                min_in = inv_sigma * (samples_centered - half)
                cdf_min = torch.sigmoid(min_in)
                cdf_delta = cdf_plus - cdf_min
                one_minus_cdf_min = torch.exp(-F.softplus(min_in))
                cdf_plus = torch.exp(plus_in - F.softplus(plus_in))

                samples2 = samples - torch.zeros_like(mu_c)
                cdf_delta = torch.where(samples2 - half < res_q_min_norm, cdf_plus,
                                        torch.where(samples2 + half > res_q_max_norm, one_minus_cdf_min,
                                                    cdf_delta))

                weights_c = weights.permute(0, 2, 1, 3)
                m = torch.amax(weights_c, 2, keepdim=True)
                weights_c = torch.exp(weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True)))
                pmf = torch.sum(cdf_delta * weights_c, dim=2)

                pmf = pmf.clamp_(1. / 54800, 1.)
                pmf = pmf / torch.sum(pmf, dim=2, keepdim=True)
                cdf = torch.cumsum(pmf, dim=2).clamp_(0., 1.)
                cdf = F.pad(cdf, (1, 0))

                symbol = torch.div(res_crop[:, c, :].short() - res_q_min, bin_sz, rounding_mode='floor')
                res_stream = torchac.encode_float_cdf(cdf.cpu(), symbol.cpu(), needs_normalization=False, check_input_bounds=False)
                code_res.append(res_stream)
                res_tmp_crop[:, c, :, 0] = symbol.float() * bin_sz + res_q_min

            res_tmp[:, :, h_idx, w_idx] = res_tmp_crop.squeeze(3)

        time_end_res = time.time()
    enclossytime += time_end_ls - time_start
    encrestime += time_end_res - time_end_ls
    print("total runtime:{:.2f}, lossy runtime:{:.2f}, res runtime:{:.2f}".format(time_end_res - time_start, time_end_ls - time_start, time_end_res - time_end_ls))

    return code_lossy, code_res, img_shape, (res_q_min, res_q_max), psnr_rrr


def lmm_params_from_context(model, res_prior, residual_context):
    if hasattr(model, "_lmm_params_from_context"):
        return model._lmm_params_from_context(res_prior, residual_context)
    rp_ctx = model.fusion(torch.cat((res_prior, residual_context), 1))
    return model.residual_compressor(rp_ctx)


def decompress(model, code_lossy, code_res, img_shape, res_range, COT, tau=0, mix_num=5, patch_sz=64, last_frame=None):
    global declossytime
    global decrestime
    norm_scale = 1 / nii_max_f * 2
    half = (0.5 + tau) * norm_scale
    bin_sz = 2 * tau + 1
    samples_end = (nii_max // bin_sz) * bin_sz

    device = next(model.parameters()).device

    res_q_min, res_q_max = res_range
    res_q_max_norm = res_q_max * norm_scale
    res_q_min_norm = res_q_min * norm_scale

    samples = torch.arange(res_q_min, res_q_max + 1, step=bin_sz, dtype=torch.float32).to(device)
    samples = samples * norm_scale

    if last_frame is not None:
        last_frame = torch.as_tensor(last_frame).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():

        time_start = time.time()
        if last_frame is None:
            rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"])
        else:
            rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"], last_frame)

        time_end_ls = time.time()

        if last_frame is not None:
            rec_lossy["x_hat"] = patchify_to_batch(rec_lossy["x_hat"], 64)
            rec_lossy["res_prior"] = patchify_to_batch(rec_lossy["res_prior"], 64)

        x_hat = rec_lossy["x_hat"]
        res_prior = rec_lossy["res_prior"]

        res_tmp = torch.zeros_like(x_hat)
        max_step = torch.max(COT)

        j = 0
        for i in range(max_step):

            h_idx, w_idx = torch.nonzero(COT == i + 1, as_tuple=True)
            ctx = model.mask_conv(res_tmp * norm_scale)[:, :, h_idx, w_idx].unsqueeze(3)

            rp = res_prior[:, :, h_idx, w_idx].unsqueeze(3)
            res_crop = res_tmp[:, :, h_idx, w_idx].unsqueeze(3)

            if tau == 0:
                lmm_params = lmm_params_from_context(model, rp, ctx)
                mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)
                coeffs = torch.tanh(coeffs)
            elif tau > 0:
                if tau > 5:
                    tau = 5
                rp_ctx = model.fusion_cond.run(torch.cat((rp, ctx), 1), tau - 1)
                lmm_params = model.residual_compressor_cond.run(rp_ctx, tau - 1)
                mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)
                coeffs = torch.tanh(coeffs)

            for c in range(1):
                mu_c = mu.permute(0, 2, 1, 3)
                samples_centered = samples - mu_c
                inv_sigma = torch.exp(-log_sigma).permute(0, 2, 1, 3)
                plus_in = inv_sigma * (samples_centered + half)
                cdf_plus = torch.sigmoid(plus_in)
                min_in = inv_sigma * (samples_centered - half)
                cdf_min = torch.sigmoid(min_in)
                cdf_delta = cdf_plus - cdf_min
                one_minus_cdf_min = torch.exp(-F.softplus(min_in))
                cdf_plus = torch.exp(plus_in - F.softplus(plus_in))

                samples2 = samples - torch.zeros_like(mu_c)
                cdf_delta = torch.where(samples2 - half < res_q_min_norm, cdf_plus,
                                        torch.where(samples2 + half > res_q_max_norm, one_minus_cdf_min,
                                                    cdf_delta))

                weights_c = weights.permute(0, 2, 1, 3)
                m = torch.amax(weights_c, 2, keepdim=True)
                weights_c = torch.exp(weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True)))
                pmf = torch.sum(cdf_delta * weights_c, dim=2)

                pmf = pmf.clamp_(1. / 54800, 1.)
                pmf = pmf / torch.sum(pmf, dim=2, keepdim=True)
                cdf = torch.cumsum(pmf, dim=2).clamp_(0., 1.)
                cdf = F.pad(cdf, (1, 0))

                symbol_out = torchac.decode_float_cdf(cdf.cpu(), code_res[j], needs_normalization=False)
                res_crop[:, c, :, 0] = symbol_out.float() * bin_sz + res_q_min
                j += 1

            res_tmp[:, :, h_idx, w_idx] = res_crop.squeeze(3)

        time_end_res = time.time()
    declossytime += time_end_ls - time_start
    decrestime += time_end_res - time_end_ls
    print("total runtime:{:.2f}, lossy runtime:{:.2f}, res runtime:{:.2f}".format(time_end_res - time_start, time_end_ls - time_start, time_end_res - time_end_ls))

    x_rec = x_hat + res_tmp
    if last_frame is not None:
        x_rec = unpatchify_from_batch(x_rec, 1, 512, 512)
    else:
        x_rec = patch2img(x_rec, img_shape)
    x_rec = x_rec.clamp_(min=0, max=nii_max).squeeze()

    return x_rec


def remove_module_prefix(state_dict, prefix="module."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def convert_old_keys(state_dict):
    """Convert new-style keys (matrices.0) to old-style (_matrix0) for eval CompressionModel."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = re.sub(r'matrices\.(\d+)', r'_matrix\1', k)
        new_k = re.sub(r'biases\.(\d+)', r'_bias\1', new_k)
        new_k = re.sub(r'factors\.(\d+)', r'_factor\1', new_k)
        new_state_dict[new_k] = v
    return new_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=int, default=1, help="near-lossless constraint (0, 1, 2, 3, ...)")
    parser.add_argument("--steps", type=int, default=20, help="DDIM sample steps used during evaluation")
    parser.add_argument("--context-type", type=str, default="simple",
                        choices=["simple", "diffusion_fusion"],
                        help="Residual context module type.")
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--i-frame-ckp", type=str, default=I_FRAME_CKP)
    parser.add_argument("--p-frame-ckp", type=str, default=P_FRAME_CKP)
    parser.add_argument("--log-prefix", type=str, default=LOG_FILE_PREFIX)
    args = parser.parse_args()
    tau = args.tau

    import nll_model_eval_p as _nme
    _nme.EVAL_DDIM_STEPS = args.steps
    print(f"[info] EVAL_DDIM_STEPS set to {args.steps}")

    dset = Dataset(args.data_path)
    device = torch.device('cuda')
    nll_module_I = NearLosslessCompressor_I(192, 5).eval().to(device)
    nll_module_P = NearLosslessCompressor_P(192, 5, context_type=args.context_type).eval().to(device)

    # Load I-frame model
    ckp_I = torch.load(args.i_frame_ckp, map_location=device, weights_only=False)
    nll_module_I.load_state_dict(ckp_I['model_state_dict'])
    nll_module_I.lossy_compressor.update(force=True)

    # Load P-frame model (diff-sit)
    ckp_P = torch.load(args.p_frame_ckp, map_location=device, weights_only=False)
    ckp_P = remove_module_prefix(ckp_P, prefix="module.")
    ckp_P = convert_old_keys(ckp_P)
    nll_module_P.load_state_dict(ckp_P)
    nll_module_P.lossy_compressor.update(force=True)

    I_PSNR = 0
    P_PSNR = 0
    i_count = 0
    p_count = 0
    i_pn = 0
    p_pn = 0
    i_zp = 0
    p_zp = 0
    i_res_bit = 0
    i_lossy_bit = 0
    p_res_bit = 0
    p_lossy_bit = 0
    m_max_diff = 0
    COT = coding_order_table7x7()
    log_file = '{}-{}.txt'.format(args.log_prefix, tau)

    for filez in dset:
        data = filez
        n = len(data)

        positions = []
        starts = list(range(0, n, 8))

        if len(starts) > 1 and n % 8 != 0:
            last_start = starts.pop()
            prev_end = n - 1
        else:
            prev_end = None

        for i, start in enumerate(starts):
            if i == len(starts) - 1 and prev_end is not None:
                end = prev_end
            else:
                end = min(start + 7, n - 1)
            positions.append((start, end))

        for slice_idx, (start, end) in enumerate(positions):
            print(start, end)
            last_frame = None
            for i in range(start, end + 1):
                frame = filez[i]

                if i == start:
                    # I-frame
                    code_lossy, code_res, img_shape, res_range, psnr_rrr = compress(nll_module_I, frame, COT, tau, 64, 5, None)
                    img_ls_sz = sum([len(code_lossy['img_strings'][0][j]) + len(code_lossy['img_strings'][1][j]) for j in range(len(code_lossy['img_strings'][0]))])
                    res_sz = sum([len(code_res[j]) for j in range(len(code_res))])
                    nll_sz = img_ls_sz + res_sz + 6 * 4

                    rec = decompress(nll_module_I, code_lossy, code_res, img_shape, res_range, COT, tau, 5, 64, None)

                    i_res_bit += res_sz * 8
                    i_lossy_bit += img_ls_sz * 8
                    i_zp += 6 * 4 * 8
                    i_pn += np.prod(frame.shape)
                    i_count += 1
                    I_PSNR += psnr_rrr

                    last_frame = rec.cpu().numpy().squeeze()
                    max_diff = np.max(np.abs(last_frame - frame))
                    if m_max_diff < max_diff:
                        m_max_diff = max_diff

                    print("I: bpsp:{:.4f}, img_bpsp:{:.4f}, res_bpsp:{:.4f}, psnr:{:.2f}".format(
                        nll_sz * 8 / np.prod(frame.shape), img_ls_sz * 8 / np.prod(frame.shape),
                        res_sz * 8 / np.prod(frame.shape), psnr_rrr))
                    with open(log_file, 'a') as f:
                        f.write("I: bpsp:{:.4f}, img_bpsp:{:.4f}, res_bpsp:{:.4f}, psnr:{:.2f}\n".format(
                            nll_sz * 8 / np.prod(frame.shape), img_ls_sz * 8 / np.prod(frame.shape),
                            res_sz * 8 / np.prod(frame.shape), psnr_rrr))
                else:
                    # P-frame
                    code_lossy, code_res, img_shape, res_range, psnr_rrr = compress(nll_module_P, frame, COT, tau, 64, 5, last_frame)
                    img_ls_sz = sum([len(code_lossy['img_strings'][0][j]) + len(code_lossy['img_strings'][1][j]) for j in range(len(code_lossy['img_strings'][0]))])
                    res_sz = sum([len(code_res[j]) for j in range(len(code_res))])
                    nll_sz = img_ls_sz + res_sz + 6 * 4

                    rec = decompress(nll_module_P, code_lossy, code_res, img_shape, res_range, COT, tau, 5, 64, last_frame)

                    p_res_bit += res_sz * 8
                    p_lossy_bit += img_ls_sz * 8
                    p_zp += 6 * 4 * 8
                    p_pn += np.prod(frame.shape)
                    p_count += 1
                    P_PSNR += psnr_rrr

                    last_frame = rec.cpu().numpy().squeeze()
                    max_diff = np.max(np.abs(last_frame - frame))
                    if m_max_diff < max_diff:
                        m_max_diff = max_diff

                    print("P: bpsp:{:.4f}, img_bpsp:{:.4f}, res_bpsp:{:.4f}, psnr:{:.2f}".format(
                        nll_sz * 8 / np.prod(frame.shape), img_ls_sz * 8 / np.prod(frame.shape),
                        res_sz * 8 / np.prod(frame.shape), psnr_rrr))
                    with open(log_file, 'a') as f:
                        f.write("P: bpsp:{:.4f}, img_bpsp:{:.4f}, res_bpsp:{:.4f}, psnr:{:.2f}\n".format(
                            nll_sz * 8 / np.prod(frame.shape), img_ls_sz * 8 / np.prod(frame.shape),
                            res_sz * 8 / np.prod(frame.shape), psnr_rrr))

    total_count = i_count + p_count
    total_pn = i_pn + p_pn
    total_bits = i_zp + i_res_bit + i_lossy_bit + p_zp + p_res_bit + p_lossy_bit

    with open(log_file, 'a') as f:
        f.write("\n===== Summary =====\n")
        f.write("I-frames: count={}, bpp={:.4f}\n".format(i_count, (i_zp + i_res_bit + i_lossy_bit) / i_pn if i_pn > 0 else 0))
        f.write("P-frames: count={}, bpp={:.4f}\n".format(p_count, (p_zp + p_res_bit + p_lossy_bit) / p_pn if p_pn > 0 else 0))
        f.write("Total: bpp={:.4f}, max_error={:.0f}\n".format(total_bits / total_pn if total_pn > 0 else 0, m_max_diff))
        if i_count > 0:
            f.write("I avg PSNR: {:.2f}\n".format(I_PSNR / i_count))
        if p_count > 0:
            f.write("P avg PSNR: {:.2f}\n".format(P_PSNR / p_count))
        f.write("Enc time: lossy={:.2f}s, res={:.2f}s\n".format(enclossytime, encrestime))
        f.write("Dec time: lossy={:.2f}s, res={:.2f}s\n".format(declossytime, decrestime))

    print("\n===== Done =====")
    print("Results saved to {}".format(log_file))
