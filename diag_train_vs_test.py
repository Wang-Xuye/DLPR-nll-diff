"""
诊断 tau=0(无损)路径的 res_bpp。

关键事实(已确认):
- 训练和 eval 都把"整帧"传给 model_P,patchify=64 在 forward 内部完成。
  -> 不存在 "patch 训练 vs 整帧 eval" 的不一致。
- model_P.forward 不管 tau 是几都会算 cond 头,而 tau=0 时 one_hot(tau-1)=one_hot(-1)
  会触发 CUDA device-side assert。但真实编码 tau=0 走的是 non-cond 头
  (test4-eval-diff-sit.py:299 的 if tau==0),根本不碰 cond 代码。
- 因此这里手动复现 forward 的 non-cond 段(nll_model_eval_p.py:599-619),
  只算 res_bpp(non-cond),这就是训练里 1.7 的那个量,直接对上 tau=0 真实编码。

注意:forward 给 x_hat 加了 uniform(-0.5,0.5) 抖动噪声,真实编码用整数量化 res_q。
本脚本同时打印"加噪版"和"量化版"的 bpp,定位差距是否来自加噪vs量化。
"""
import math
import torch
from nll_model_eval_p import NearLosslessCompressor_P, patchify_to_batch
from logisticmixturemodel import LogisticMixtureModel
from train4_diff_sit import Dataset_test
from global_nii_max import nii_max_f

CKP = "./log4_warmup_stageB/p-Ep<最新>-0.005-1.718-5.064.pth"  # ← 改成你 Stage B 的真实 ckpt
DEVICE = "cuda:0"


def bpp_from_loglik(loglik, num_pixels):
    """loglik: 自然对数似然张量。返回 bits/pixel。"""
    return (loglik.sum() / (-math.log(2) * num_pixels)).item()


@torch.no_grad()
def res_bpp_noncond(model, frame, pre, quantize):
    """
    复现 nll_model_eval_p.py forward 的 non-cond 段(tau=0)。
    quantize=False: 用 forward 的加噪版 x_hat(训练 loss 路径,1.7 的来源)
    quantize=True : 用整数量化 res(贴近真实算术编码)
    返回 (res_bpp, num_pixels)
    """
    lossy_out = model.lossy_compressor(frame, pre)
    x_hat = patchify_to_batch(lossy_out["x_hat"], 64)
    res_prior = patchify_to_batch(lossy_out["res_prior"], 64)
    inp = patchify_to_batch(frame, 64)

    if quantize:
        # tau=0: res_q = round(res),无噪声
        res = inp - x_hat
        res = torch.round(res)
    else:
        # forward 原逻辑:加 uniform(-0.5,0.5) 抖动
        noise = torch.empty_like(inp).uniform_(-0.5, 0.5)
        x_hat_n = torch.clamp(x_hat + noise, min=0., max=nii_max_f)
        res = inp - x_hat_n

    res_n = res / nii_max_f * 2
    context = model.mask_conv(res_n)
    res_prior_context = model.fusion(torch.cat((res_prior, context), 1))
    lmm_params = model.residual_compressor(res_prior_context)
    mu, log_sigma, coeffs, weights = torch.split(lmm_params, 5, dim=1)
    lmm = LogisticMixtureModel(mu, log_sigma, weights, coeffs)
    log_res_likelihoods = lmm(res_n)

    # patchify 后整帧总像素
    N, _, H, W = inp.shape
    num_pixels = N * H * W
    return bpp_from_loglik(log_res_likelihoods, num_pixels)


def main():
    model = NearLosslessCompressor_P(192, 5).to(DEVICE).eval()
    sd = torch.load(CKP, map_location=DEVICE)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckp] missing={len(missing)} unexpected={len(unexpected)}")

    test_ds = Dataset_test("../Mosmed/test-slice")
    clip = test_ds[0]                       # numpy (T,1,H,W)
    x = torch.as_tensor(clip).to(DEVICE)
    print(f"[data] clip.shape = {tuple(x.shape)}")

    frame = x[1:2]      # (1,1,H,W) 整帧
    pre = x[0:1]
    print(f"[data] frame.shape = {tuple(frame.shape)}  (整帧,patchify 在 forward 内部)")

    bpp_noise = res_bpp_noncond(model, frame, pre, quantize=False)
    bpp_quant = res_bpp_noncond(model, frame, pre, quantize=True)

    print(f"\n[tau=0 non-cond] res_bpp 加噪版(=训练 loss)  = {bpp_noise:.4f}")
    print(f"[tau=0 non-cond] res_bpp 量化版(贴近真实编码) = {bpp_quant:.4f}")
    print(f"[diff] 量化 - 加噪 = {bpp_quant - bpp_noise:.4f} bpp")

    print("\n判读:")
    print("  - 加噪版 ≈ 你训练看到的 1.7,这是 PMF 积分理论值。")
    print("  - 若量化版也接近 1.7,但真实算术编码 ≈ 10.6")
    print("    -> 差距不在加噪vs量化,而在算术编码器/上下文构造,下一步去 compress dump PMF。")
    print("  - 若量化版就已经飙到 ~10,说明 round 后残差落在模型没学好的区间。")


if __name__ == "__main__":
    main()
