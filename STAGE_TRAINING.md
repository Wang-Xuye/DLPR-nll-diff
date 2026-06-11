# DiffusionLMMHead 三阶段训练 — 实验运行手册

> 适用脚本：`train4_diff_sit.py` / `init_diff_lmm_from_baseline.py` / `verify_bitexact.py` / `test4-eval-diff-sit.py` / `residual_diagnose.py`
> 工作目录：`DLPR-Mosmed-nll/`
> 服务器路径：`/data_a/wxy/linc/DLPR-Mosmed-nll`

---

## 0. 一句话目标

把 P 帧的 `ResidualCompressor`（3 层 1×1 conv 直出 LMM 参数）替换成 `DiffusionLMMHead`（条件扩散 K=3 步 DDIM 出 LMM 参数），用三段式训练把 Mosmed 的 P 帧 lossless **bpp 从 4.83 → 4.3~4.6**（目标追平/超过论文 DINR baseline 4.72）。

baseline ckpt 命名格式：`p-Ep{epoch}-{img_bpp:.3f}-{img_bpp+res_bpp:.3f}-{img_bpp+res_bpp_cond:.3f}.pth`，**第二个数字（img+res_bpp）是关键指标**。

---

## 1. 三阶段速查表

| Stage | 解冻 | teacher | DDIM 反向 | 损失 | LR | save-dir 建议 |
|-------|------|---------|-----------|------|-----|--------------|
| **A** | 仅 `residual_compressor.unet` | enabled | 跳过 | `1.0 × diff_lmm_loss`（自蒸馏 MSE） | 1e-4 | `log4_warmup_stageA` |
| **B** | 仅 `residual_compressor.unet` | disabled | K=3 步，截断 BPTT（最后一步带梯度） | `img_bpp + res_bpp` | 1e-4 | `log4_warmup_stageB` |
| **C** | 全部解冻（teacher 永远冻） | disabled | K=3 步，截断 BPTT | `0.1·mse + img_bpp + res_bpp` | 1e-5 | `log4_warmup_stageC` |

**Stage A**：让随机初始化的 unet 学会"看 cond 复现 teacher 的 LMM 参数"。保底，不许退化。
**Stage B**：unet 摆脱 teacher 束缚，自由优化让 LMM 参数更准（中心更尖、尾部更胖）。这是 bpp 真正下降的阶段。
**Stage C**：让 CDC、mask_conv、fusion 等上下游模块协同适配新 unet，挤最后一点 bpp。

---

## 2. 完整 Pipeline（4 步）

### Step 0. 准备 Stage A 起点 ckp

把旧 baseline 的 `ResidualCompressor` 权重灌进新模型的 `DiffusionLMMHead.teacher`，`unet` 留随机初始化。

```bash
cd /data_a/wxy/linc/DLPR-Mosmed-nll

python init_diff_lmm_from_baseline.py \
    --old-ckp ./log4_diff_warmup_nobpp200/p-Ep1251-0.005-4.796-5.067.pth \
    --new-ckp ./log4_diff_warmup_nobpp200/p_init_with_teacher.pth
```

**为什么必须做这一步**：旧 ckpt 没有 `residual_compressor.unet.*` 和 `residual_compressor.teacher.*` 这两组键。如果直接 `--resume` 旧 ckpt 进 Stage A，teacher 会是空权重，自蒸馏目标就坏了。

### Step 1. Stage A — 自蒸馏 warmup

```bash
python train4_diff_sit.py --stage A \
    --resume ./log4_diff_warmup_nobpp200/p_init_with_teacher.pth \
    --save-dir log4_warmup_stageA --epochs 20000
```

**进阶判据**：
- TensorBoard 上 `train/diff_lmm_loss` 收敛到 ≤ 0.01 量级
- 落盘 ckpt 第二个数字（img+res_bpp，DDIM 推理路径）从初始 11.5+ 降到接近 baseline 4.892

> Stage A 第一份 ckpt 第二个数字一定很高（譬如 11.576），那是因为 eval 走 DDIM 推理路径（`self.training=False`，进 `_ddim_reverse`），unet 还没蒸馏到位输出基本是噪声。继续训即可。

### Step 2. Stage B — 联合微调

挑 `log4_warmup_stageA/` 里 epoch 最大、第二个数字最低的 ckpt，作为 Stage B 起点：

```bash
python train4_diff_sit.py --stage B \
    --resume ./log4_warmup_stageA/p-Ep501-0.005-5.102-5.065.pth \
    --save-dir log4_warmup_stageB --epochs 20000
```

**进阶判据**：第二个数字稳定低于 baseline 4.892，目标 4.3~4.6。

### Step 3. Stage C — 端到端微调（可选）

```bash
python train4_diff_sit.py --stage C \
    --resume ./log4_warmup_stageB/p-Ep{N}-{...}.pth \
    --save-dir log4_warmup_stageC --epochs 20000
```

**进阶判据**：第二个数字再下降一点，第一个数字（img_bpp）不要明显上升。

---

## 3. 续训（任意阶段中途断了）

`--resume` 指向同一 `--save-dir` 里最新的 `p-Ep{N}-...pth` 即可，`start_epoch` 自动从文件名解析、LR scheduler 自动快进对齐。

```bash
# 例：Stage A 训到 epoch 1500 断了，接着训
python train4_diff_sit.py --stage A \
    --resume ./log4_warmup_stageA/p-Ep1501-...pth \
    --save-dir log4_warmup_stageA --epochs 20000
```

如果 ckpt 文件名不是 `p-Ep{N}-...pth` 格式（比如手动改过名），加 `--start-epoch N` 显式指定。

---

## 4. 训练后验证（三步全跑）

### 4.1 bit-exact 检查（最关键）

lossless 算术编码要求编/解码两端完全一致。`DiffusionLMMHead` 用 deterministic noise + η=0 DDIM + fp32 保证 bit-exact，需要验证：

```bash
python verify_bitexact.py --ckp ./log4_warmup_stageC/p-Ep{N}-...pth --mode same --runs 3
python verify_bitexact.py --ckp ./log4_warmup_stageC/p-Ep{N}-...pth --mode save --ref-file ref.pt
python verify_bitexact.py --ckp ./log4_warmup_stageC/p-Ep{N}-...pth --mode check --ref-file ref.pt
```

期望：全部 PASS。一旦失败 → 检查是不是 deterministic_noise 的 seed/shape/dtype 在不同进程下漂了。

### 4.2 端到端 lossless bpp

```bash
python test4-eval-diff-sit.py --tau 0 --steps 5
```

期望：`Total bpp < 4.79`（低于 baseline），`max_error == 0`（lossless 严格）。

### 4.3 残差分布回测

```bash
python residual_diagnose.py \
    --p-ckp ./log4_warmup_stageC/p-Ep{N}-...pth \
    --i-ckp ckp_nll/0001/ckp.tar \
    --data-dir ../Mosmed/test-slice \
    --num-seqs 5 --max-p-frames 40
```

期望：`residual_diag_hist.png` 中 LMM 红线和蓝柱在中心更尖、尾部更胖，比 baseline 更贴合。

---

## 5. 代码改动索引（核心位置）

### 5.1 `nll_model_p.py` / `nll_model_eval_p.py`

新增（两文件结构对称）：

| 类 / 方法 | 大致行号 | 作用 |
|----------|---------|------|
| `from contextlib import nullcontext` | 文件头 import 区 | 截断 BPTT 用 |
| `class LMM_UNet` | ~1063 起 | 轻量 U-Net，base_dim=64，2 级下采样，cond 256ch + params 20ch concat |
| `class DiffusionLMMHead` | ~1130 起 | 包装 unet + teacher，统一对外 `forward(cond)` |
| `DiffusionLMMHead._ddim_reverse` | ~1185 起 | K=3 步 DDIM 反向，**训练时只有最后一步 (i==0) 带梯度**，前面 K-1 步 `no_grad` + `detach` |
| `DiffusionLMMHead.forward` | ~1227 起 | Stage A（training + teacher_enabled）跳过 `_ddim_reverse`，仅一次蒸馏前向；Stage B/C 走 DDIM |

替换：
- `NearLosslessCompressor_P.__init__`（line 505 附近）：`residual_compressor = DiffusionLMMHead(cond_ch=256, num_mixtures=5)`，接口与旧 `ResidualCompressor` 完全一致。

不动：
- `LogisticMixtureModel`、`mask_conv`、`LossyCompressor_P_diff_sit`、算术编码接口。

### 5.2 `train4_diff_sit.py`

| 行号 | 内容 |
|------|------|
| 7 | `SAVE_DIR_DEFAULT = "log4_diff_warmup_nobpp200"` |
| 232/268 | `eval_epoch(...)` 接受 `save_dir` 形参，落盘到 `f'{save_dir}/p-Ep{epoch+1}-{...}.pth'` |
| 271 | `train(... save_dir=SAVE_DIR_DEFAULT, start_epoch_override=None)` |
| 300-315 | 从 `--resume` 文件名解析 `start_epoch`，scheduler 快进到对齐 |
| 317-360 | 三阶段冻结/解冻 + 优化器构造，Stage B 的 aux_optimizer 用 `n.endswith(".quantiles")` 筛选 |
| 376 | `for epoch in range(start_epoch, epochs)` |
| 385 | `if epoch % 50 == 0` 触发 eval+落盘（含 epoch 0） |
| 400-409 | CLI：`--resume / --stage / --epochs / --save-dir / --start-epoch` |

---

## 6. 已踩过的坑 + 解决方案

### 坑 1：Stage A 直接 OOM（24G 显存炸）

**现象**（曾出现）：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB.
GPU 0 has a total capacity of 23.68 GiB of which 446.94 MiB is free.
```

**根因**：`DiffusionLMMHead.forward` 既跑 K=3 步 DDIM 反向（unet trainable，3 次激活全保），又额外跑一次蒸馏前向 = 4 次 unet 激活叠加。Baseline `ResidualCompressor` 几乎不占显存，换成 4 倍 unet 直接撑爆。

**解决**：
- Stage A：`forward` 跳过 `_ddim_reverse`，只跑 1 次 unet 前向（蒸馏）+ 1 次 teacher（无梯度）
- Stage B/C：`_ddim_reverse` 内部只让最后一步 (i==0) `nullcontext`（带梯度），前面 K-1 步 `torch.no_grad()` 后立即 `detach`

显存峰值降到 ≈ baseline + 1 个 unet。

### 坑 2：Stage B 启动报 `'LossyCompressor_P_diff_sit' object has no attribute 'aux_parameters'`

**根因**：项目用的 compressai 版本 `CompressionModel` 只有 `aux_loss()`，没有 `aux_parameters()` 方法。

**解决**：和 `configure_optimizers` (line 41) 保持一致，从 `named_parameters()` 里筛 `n.endswith(".quantiles")`：

```python
aux_params = [
    p for n, p in nll_module_P.lossy_compressor.named_parameters()
    if n.endswith(".quantiles") and p.requires_grad
]
aux_optimizer_P = optim.Adam(aux_params, lr=1e-3) if aux_params else None
```

Stage B 因为 lossy 端冻结，`aux_params` 实际为空 → `aux_optimizer_P=None`，`train_one_epoch` 里 `if aux_optimizer_P is not None` 跳过 aux 步骤，干净。

### 坑 3：Stage A 第一份 ckpt 第二个数字 11.576，不是预期的 ≈ 4.892

**根因**：那是 eval 走 DDIM 推理路径（`self.training=False`，进 `_ddim_reverse`，K=3 步用 unet 输出生成 LMM 参数）。unet 才训 1 个 epoch，输出基本是噪声。**这不是 bug**，是 Stage A 远没收敛的正常现象。

**判别**：
- 第三个数字（cond 路径）≈ baseline（5.0~5.3）→ 说明 lossy/mask_conv/teacher 初始化都正常
- 第二个数字偏高 → unet 没蒸馏到位，继续训

**进阶**：Stage A 训到第二个数字降到 ≤ 5.0 / 接近 4.892 再切 Stage B 最稳妥；至少 ≤ 5.5 也可以试探。

### 坑 4：直接 `--resume` 旧 baseline ckpt 进 Stage A

**风险**：旧 ckpt 没有 `residual_compressor.unet.*` / `residual_compressor.teacher.*` 这两组键。如果 `load_state_dict(strict=False)` 灰过去，teacher 是空权重，蒸馏目标就坏了。

**正确做法**：先跑 `init_diff_lmm_from_baseline.py` 准备一个 `_init_with_teacher.pth`，再用它启 Stage A（见 Step 0）。

### 坑 5：续训时 LR 没衰减

**根因**：原代码 `MultiStepLR(milestones=[350, 390, ...])` 每次重新构造时都从 epoch 0 算。续训到 epoch 500 的话，那些 milestone 早过了但 scheduler 不知道，LR 还是初始值 1e-4。

**解决**：续训时 `for _ in range(start_epoch): scheduler.step()` 把 scheduler 快进到 `start_epoch`，对齐到正确 LR。

---

## 7. 落盘文件名约定

```
p-Ep{epoch+1}-{img_bpp:.3f}-{img_bpp+res_bpp:.3f}-{img_bpp+res_bpp_cond:.3f}.pth
```

四个数字：

| 字段 | 含义 | baseline 参考 | 关注重点 |
|------|------|--------------|---------|
| epoch+1 | 训练轮数（从 1 计） | — | 续训时用作 `--start-epoch` |
| img_bpp | 有损图像 bpp | 0.179 | Stage C 不要明显上升 |
| img+res_bpp | **图像 + 残差 bpp（DDIM 推理路径）** | **4.892** | **核心指标**，目标 4.3~4.6 |
| img+res_bpp_cond | 图像 + 条件残差 bpp（cond 路径） | 5.286 | 辅助参考 |

落盘频率：`if epoch % 50 == 0`（含 epoch 0，即 resume 进来时立刻落一份基线）。

---

## 8. TensorBoard 关键曲线

```bash
tensorboard --logdir logs4_diff --port 6006
```

| 曲线 | Stage A | Stage B | Stage C |
|------|---------|---------|---------|
| `train/diff_lmm_loss` | **主要关注**，应单调下降到 ≤ 0.01 | 不计算（teacher off） | 不计算 |
| `train/res_bpp` | teacher 路径，≈ baseline 4.7 不动 | **主要关注**，目标低于 4.79 | **主要关注** |
| `train/img_bpp` | ≈ baseline 0.18 不动 | ≈ baseline 不动 | 不要明显上升 |
| `train/loss` | = `1.0 × diff_lmm_loss` | = `img_bpp + res_bpp` | = `0.1·mse + img_bpp + res_bpp` |

---

## 9. 退路（Stage B 100 epoch 还压不下来怎么办）

按优先级：

1. **Stage A 训得更久**：自蒸馏到第二个数字 ≤ 4.95 再进 Stage B
2. **加大 K**：把 `DiffusionLMMHead.sample_steps` 从 3 改到 5（推理慢一点，但 DDIM 反向能力更强）
3. **放弃此路线**：转方案 A（直接把 LMM `num_mixtures` 从 5 加大到 20），这是最朴素但稳定的对照
4. 若都不行：方案 B 在医学影像残差上不 work，写进论文 negative result

---

## 10. 一句话设计原则

> **A 让 unet 能复现 teacher（保底），B 让 unet 超越 teacher（涨点），C 让整个网络协同适配 unet（榨干最后一滴 bpp）。**
