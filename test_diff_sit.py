import torch
from nll_model_p import LossyCompressor_P_diff_sit

model = LossyCompressor_P_diff_sit(192)
model.train()

inp = torch.rand(1, 1, 256, 256) * 4095
pre = torch.rand(1, 1, 256, 256) * 4095

out = model(inp, pre)
print("x_hat:", out["x_hat"].shape)
print("res_prior:", out["res_prior"].shape)
print("diffusion_loss:", out["diffusion_loss"].item())
print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
