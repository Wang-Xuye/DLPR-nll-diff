import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 指定只有第二张GPU对PyTorch可见
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from nll_model_eval_p import NearLosslessCompressor_I, NearLosslessCompressor_P, unpatchify_from_batch, patchify_to_batch

from torchmetrics import PeakSignalNoiseRatio

import time
from PIL import Image
import numpy as np
import time
from  tqdm import tqdm
import torchac
from global_nii_max import nii_max, nii_max_f
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
enclossytime=0
encrestime=0
declossytime=0
decrestime=0
def replicate_pad(x, p):
    height=x.shape[-2]
    width=x.shape[-1]
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
        self.crop = 64
        self.clips = []
        for root, _, files in os.walk(self.root_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                self.clips.append(full_path)
            
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, index):
        clip_path = self.clips[index]
        clip = np.load(clip_path).astype(np.float32).transpose((2,0,1))
        print(clip.shape)
        clip=replicate_pad(clip,128)
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

    if(len(img.shape)==2):
        img=img.unsqueeze(2)

    h, w, _ = img.shape

    h_num = h//patch_sz - (patch_sz -(h % patch_sz)) // patch_sz + 1
    w_num = w//patch_sz - (patch_sz -(w % patch_sz)) // patch_sz + 1

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
    h_num = h//patch_sz - (patch_sz -(h % patch_sz)) // patch_sz + 1
    w_num = w//patch_sz - (patch_sz -(w % patch_sz)) // patch_sz + 1


    patch = patch.permute(0, 2, 3, 1)

    patch_w = torch.chunk(patch, w_num, dim=0)
    patch_h = torch.cat(patch_w, dim=2)

    patch_h = torch.chunk(patch_h, h_num, dim=0)
    img = torch.cat(patch_h, dim=1).squeeze(0)

    return img[-h:, -w:, :]


def compress(model, input_img, COT, tau=0, patch_sz=64, mix_num=5, last_frame=None):
    global enclossytime
    global encrestime
    norm_scale = 1/nii_max_f*2
    half = (0.5 + tau) * norm_scale
    mix_num2 = 2 * mix_num
    code_res = []
    img_shape = input_img.shape[:2]
    bin_sz = 2 * tau + 1
    samples_end = (nii_max // bin_sz) * bin_sz

    x=None
    device = next(model.parameters()).device
    if last_frame is None:
        x = img2patch(torch.as_tensor(input_img), patch_sz=patch_sz).to(device)
    else:
        x=torch.as_tensor(input_img).unsqueeze(0).unsqueeze(0).to(device)
        last_frame=torch.as_tensor(last_frame).unsqueeze(0).unsqueeze(0).to(device)
        
    with torch.no_grad():
        time_start = time.time()

        if last_frame is None:
            code_lossy = model.lossy_compressor.compress(x)
            rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"])
        else:
            code_lossy = model.lossy_compressor.compress(x,last_frame)
            rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"], last_frame)
        time_end_ls = time.time()
        if last_frame is not None:
            rec_lossy["x_hat"]=patchify_to_batch(rec_lossy["x_hat"],64)
            rec_lossy["res_prior"]=patchify_to_batch(rec_lossy["res_prior"],64)   
            x=patchify_to_batch(x,64) 
        psnr = PeakSignalNoiseRatio()
        
        x_hat = rec_lossy["x_hat"]
        res_prior = rec_lossy["res_prior"]
        res = x - x_hat
        psnr_rrr=psnr(x.detach().cpu(), x_hat.detach().cpu())
        #print(torch.max(res),torch.min(res), torch.max(x_hat),torch.min(x_hat),torch.max(x),torch.min(x))
        
        res_q = torch.where(res>=0, (2*tau+1) * torch.floor((res+tau)/(2*tau+1)),
                            (2*tau+1) * torch.ceil((res-tau)/(2*tau+1)))


        res_q_max = int(torch.max(res_q).item())
        res_q_min = int(torch.min(res_q).item())

        res_q_max_norm = res_q_max * norm_scale
        res_q_min_norm = res_q_min * norm_scale

        res_q_max_idx = (res_q_max + samples_end) // bin_sz
        res_q_min_idx = (res_q_min + samples_end) // bin_sz
        print("Encode Res range:[{}({}),{}({})], elems num:{}".format(res_q_min, res_q_min_idx, res_q_max, res_q_max_idx, res_q_max_idx-res_q_min_idx+1))

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

                rp_ctx = model.fusion(torch.cat((rp, ctx), 1))
                lmm_params = model.residual_compressor(rp_ctx)
                mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)
                coeffs = torch.tanh(coeffs)
            elif tau > 0:
                if tau > 5:
                    tau = 5

                rp_ctx = model.fusion_cond.run(torch.cat((rp, ctx), 1), tau-1)
                lmm_params = model.residual_compressor_cond.run(rp_ctx, tau-1)
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
                one_minus_cdf_min = torch.exp(-F.softplus(min_in))  # res_q_max
                cdf_plus = torch.exp(plus_in - F.softplus(plus_in))  # res_q_min

                samples2 = samples - torch.zeros_like(mu_c)
                cdf_delta = torch.where(samples2 - half < res_q_min_norm, cdf_plus,
                                        torch.where(samples2 + half > res_q_max_norm, one_minus_cdf_min,
                                                    cdf_delta))

                weights_c = weights.permute(0, 2, 1, 3)
                m = torch.amax(weights_c, 2, keepdim=True)
                weights_c = torch.exp(weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True)))
                pmf = torch.sum(cdf_delta * weights_c, dim=2)

                pmf = pmf.clamp_(1./54800, 1.)
                pmf = pmf/torch.sum(pmf, dim=2, keepdim=True)
                cdf = torch.cumsum(pmf, dim=2).clamp_(0., 1.)
                cdf = F.pad(cdf, (1, 0))

                symbol = torch.div(res_crop[:, c, :].short() - res_q_min, bin_sz, rounding_mode='floor')
                res_stream = torchac.encode_float_cdf(cdf.cpu(), symbol.cpu(), needs_normalization=False, check_input_bounds=False)
                code_res.append(res_stream)
                res_tmp_crop[:, c, :, 0] = symbol.float() * bin_sz + res_q_min

            res_tmp[:, :, h_idx, w_idx] = res_tmp_crop.squeeze(3)

        time_end_res = time.time()
    enclossytime+=time_end_ls-time_start
    encrestime+=time_end_res-time_end_ls
    print("total runtime:{:.2f}, lossy runtime:{:.2f}, res runtime:{:.2f}".format(time_end_res-time_start, time_end_ls-time_start, time_end_res-time_end_ls))

    return code_lossy, code_res, img_shape, (res_q_min, res_q_max), psnr_rrr

def decompress(model, code_lossy, code_res, img_shape, res_range, COT, tau=0, mix_num=5, patch_sz=64, last_frame=None):
    global declossytime
    global decrestime
    norm_scale = 1/nii_max_f*2
    half = (0.5 + tau) * norm_scale
    mix_num2 = 2 * mix_num
    bin_sz = 2 * tau + 1
    samples_end = (nii_max // bin_sz) * bin_sz

    device = next(model.parameters()).device

    res_q_min, res_q_max = res_range
    res_q_max_norm = res_q_max * norm_scale
    res_q_min_norm = res_q_min * norm_scale

    res_q_max_idx = (res_q_max + samples_end) // bin_sz
    res_q_min_idx = (res_q_min + samples_end) // bin_sz
    #print("Decode Res range:[{}({}),{}({})]".format(res_q_min, res_q_min_idx, res_q_max, res_q_max_idx))

    samples = torch.arange(res_q_min, res_q_max + 1, step=bin_sz, dtype=torch.float32).to(device)
    samples = samples * norm_scale
    
    if last_frame is not None:
        last_frame=torch.as_tensor(last_frame).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():

        time_start = time.time()
        if last_frame is None:
            rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"])
        else: 
            rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"], last_frame)

        time_end_ls = time.time()

        if last_frame is not None:
            rec_lossy["x_hat"]=patchify_to_batch(rec_lossy["x_hat"],64)
            rec_lossy["res_prior"]=patchify_to_batch(rec_lossy["res_prior"],64)
            
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
                rp_ctx = model.fusion(torch.cat((rp, ctx), 1))
                lmm_params = model.residual_compressor(rp_ctx)
                mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)
                coeffs = torch.tanh(coeffs)
            elif tau > 0:
                if tau > 5:
                    tau = 5
                rp_ctx = model.fusion_cond.run(torch.cat((rp, ctx), 1), tau-1)
                lmm_params = model.residual_compressor_cond.run(rp_ctx, tau-1)
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
                one_minus_cdf_min = torch.exp(-F.softplus(min_in))  # res_q_max
                cdf_plus = torch.exp(plus_in - F.softplus(plus_in))  # res_q_min

                samples2 = samples - torch.zeros_like(mu_c)
                cdf_delta = torch.where(samples2 - half < res_q_min_norm, cdf_plus,
                                        torch.where(samples2 + half > res_q_max_norm, one_minus_cdf_min,
                                                    cdf_delta))


                weights_c = weights.permute(0, 2, 1, 3)
                m = torch.amax(weights_c, 2, keepdim=True)
                weights_c = torch.exp(weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True)))
                pmf = torch.sum(cdf_delta * weights_c, dim=2)

                pmf = pmf.clamp_(1./54800, 1.)
                pmf = pmf/torch.sum(pmf, dim=2, keepdim=True)
                cdf = torch.cumsum(pmf, dim=2).clamp_(0., 1.)
                cdf = F.pad(cdf, (1, 0))

                symbol_out = torchac.decode_float_cdf(cdf.cpu(), code_res[j], needs_normalization=False)
                res_crop[:, c, :, 0] = symbol_out.float() * bin_sz + res_q_min
                j += 1

            res_tmp[:, :, h_idx, w_idx] = res_crop.squeeze(3)

        time_end_res = time.time()
    declossytime+=time_end_ls-time_start
    decrestime+=time_end_res-time_end_ls
    print("total runtime:{:.2f}, lossy runtime:{:.2f}, res runtime:{:.2f}".format(time_end_res-time_start, time_end_ls-time_start, time_end_res-time_end_ls))

    x_rec= x_hat+res_tmp
    if last_frame is not None:
        x_rec=unpatchify_from_batch(x_rec,1, 512,512)
    else:
        x_rec=patch2img(x_rec, img_shape)
    x_rec =x_rec.clamp_(min=0, max=nii_max).squeeze()

    return x_rec
    
def remove_module_prefix(state_dict, prefix="module."):
    """
    去除 state_dict 中 key 的前缀（默认是 'module.'）
    Args:
        state_dict (dict): 原始 state_dict
        prefix (str): 需要去除的前缀
    Returns:
        new_state_dict (dict): 去除前缀后的 state_dict
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=int, default=1, help="near-lossless constraint (0, 1, 2, 3, ...)")
    args = parser.parse_args()
    tau = args.tau

    dset = Dataset('/root/autodl-tmp/Mosmed/test-full')
    sumz=0
    pn=0
    device = torch.device('cuda')
    nll_module_I = NearLosslessCompressor_I(192, 5).eval().to(device)
    nll_module_P = NearLosslessCompressor_P(192, 5).eval().to(device)


    #log=open("log.txt","w")
    I_PSNR=0
    P_PSNR=0
    I_F_PSNR=0
    P_F_PSNR=0
    ckp_I = torch.load("/root/autodl-tmp/DLPR-Mosmed-nll/ckp_nll/0001/ckp.tar", map_location=device)
    
    nll_module_I.load_state_dict(ckp_I['model_state_dict'])
    nll_module_I.lossy_compressor.update(force=True)

    ckp_P = torch.load("/root/autodl-tmp/DLPR-Mosmed-nll/log4/p-Ep541-0.179-4.892-5.286.pth", map_location=device)
    ckp_P=remove_module_prefix(ckp_P, prefix="module.")
    nll_module_P.load_state_dict(ckp_P)
    nll_module_P.lossy_compressor.update(force=True)

    i_count=0
    p_count=0
    i_pn=0
    p_pn=0
    i_zp=0
    p_zp=0
    i_res_bit=0
    i_lossy_bit=0
    p_res_bit=0
    p_lossy_bit=0
    m_max_diff=0
    COT = coding_order_table7x7()
    icount=0
    pcount=0
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
        
        # 生成每个切片的起始和结束位置
        for i, start in enumerate(starts):
            if i == len(starts) - 1 and prev_end is not None:
                end = prev_end
            else:
                end = min(start + 7, n - 1) 
            positions.append((start, end))
        for slice_idx, (start, end) in enumerate(positions):
            print(start, end)
            for i in range(start, end + 1):
                I = filez[i]
                icount+=1
                code_lossy, code_res, img_shape, res_range, psnr_rrr = compress(nll_module_I, I, COT, tau, 64, 5, None)
                    
                img_ls_sz = sum([len(code_lossy['img_strings'][0][i]) + len(code_lossy['img_strings'][1][i]) for i in range(len(code_lossy['img_strings'][0]))])
                res_sz = sum([len(code_res[i]) for i in range(len(code_res))])
                nll_sz = img_ls_sz + res_sz + 6 * 4 # lossy_image_bitstream + residual_bitstream + z_shape + image_shape + residual_range
                    
                I_nll = decompress(nll_module_I, code_lossy, code_res, img_shape, res_range, COT, tau, 5, 64, None)
                    
                i_res_bit+=res_sz*8
                i_lossy_bit+=img_ls_sz*8
                i_zp+= 6 * 4 * 8
                i_pn+=np.prod(I.shape)
                

                I_nll = I_nll.cpu().numpy().squeeze()
                last_frame=I_nll
    
                max_diff = np.max(np.abs(I_nll-I))
                mse_loss = np.mean((I-I_nll)**2)
                psnr_rrr2 = 10. * np.log10(255. ** 2 / mse_loss)
                
                if m_max_diff<max_diff:
                    m_max_diff=max_diff
                

                I_PSNR+=psnr_rrr
                I_F_PSNR+=psnr_rrr2
                print("i: bpsp:{:.4f}, img_bpsp:{:.4f}, res_bpsp:{:.4f}".format(nll_sz*8/np.prod(I.shape), img_ls_sz*8/np.prod(I.shape), res_sz*8/np.prod(I.shape)))

                with open('log-dlpr-'+str(tau)+'.txt', 'a') as f:
                    f.write("i: bpsp:{:.4f}, img_bpsp:{:.4f}, res_bpsp:{:.4f}\n".format(nll_sz*8/np.prod(I.shape), img_ls_sz*8/np.prod(I.shape), res_sz*8/np.prod(I.shape)))
 
        with open('log-dlpr-'+str(tau)+'.txt', 'a') as f:
            f.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(enclossytime/(icount+pcount), encrestime/(icount+pcount),  declossytime/(icount+pcount), decrestime/(icount+pcount), I_PSNR/icount,I_F_PSNR/icount))

    with open('log-dlpr-'+str(tau)+'.txt', 'a') as f:             
        f.write("icount:{:d}, i_pn:{:d}, i_zp:{:d}, i_res_bit:{:d}, i_lossy_bit:{:d}\n".format(i_count,i_pn,i_zp,i_res_bit,i_lossy_bit))
        f.write("I_bpp:{:4f}, i_res_bit:{:4f}, i_lossy_bit:{:4f}\n".format((i_zp+i_res_bit+i_lossy_bit)/i_pn,i_res_bit/i_pn,i_lossy_bit/i_pn))

        f.write("total_bpp{:4f}, max_error{:4f}\n".format( (i_zp+i_res_bit+i_lossy_bit+p_zp+p_res_bit+p_lossy_bit)/(i_pn+p_pn),  m_max_diff))
        f.write("\n")

