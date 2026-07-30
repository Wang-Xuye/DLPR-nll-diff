# B帧修改思路与复现记录

本文记录当前在 `DLPR-Mosmed-nll` 中加入 B 帧双参考帧压缩的思路、已实现内容、训练测试方法和后续可扩展方向，便于之后复现实验。

## 1. 修改目标

原始 P 帧模型主要使用单个已重建参考帧引导当前帧压缩。B 帧方案的核心目标是：对当前帧同时使用左右两个已重建参考帧作为条件信息，使模型能够利用当前帧前后方向的结构相似性，降低残差复杂度，从而改善压缩性能。

本仓库当前实现的是“静态双参考帧 B 帧”版本：

- GOP 固定为 9 帧。
- 第 0 帧作为 I 帧。
- 第 8 帧先以第 0 帧作为参考重建，作为右端锚点。
- 其余帧按固定层级顺序编码，每一帧只引用已经重建过的参考帧。
- 测试时同样使用已重建帧作为参考，避免训练/测试不一致。

## 2. 当前静态参考帧结构

当前固定参考表为：

```python
STATIC_REF_MAP_9 = [
    [0, 0, 0],
    [8, 0, 0],
    [4, 0, 8],
    [2, 0, 4],
    [1, 0, 2],
    [3, 2, 4],
    [6, 4, 8],
    [5, 4, 6],
    [7, 6, 8],
]
```

每一项含义是：

```text
[target, ref_l, ref_r]
```

也就是编码 `target` 帧时，使用 `ref_l` 和 `ref_r` 两个已重建帧作为参考。

需要注意：

- 两个参考帧不一定总是当前帧的一前一后。
- 对 `[8, 0, 0]`，第 8 帧是右端锚点，此时左右参考都退化为第 0 帧。
- 对 `[4, 0, 8]`、`[2, 0, 4]`、`[6, 4, 8]` 等，两个参考帧分别来自当前帧两侧。
- 这个顺序保证了因果性：编码某个 B 帧时，它引用的参考帧都已经完成重建。

## 3. 模型融合方式

当前 B 帧模型入口仍然是 `NearLosslessCompressor_P`，但调用形式从单参考帧变成：

```python
out = model_p(current, ref_l, ref_r, tau)
```

其中：

- `current` 是当前待压缩帧。
- `ref_l` 是左参考帧的重建结果。
- `ref_r` 是右参考帧的重建结果。
- `tau` 是近无损误差约束参数。

双参考帧共同引导压缩的方式可以理解为：

1. 模型同时接收当前帧、左参考帧、右参考帧。
2. 有损压缩部分根据双参考条件生成当前帧的有损重建和上下文先验。
3. 残差压缩部分使用该上下文先验估计残差分布。
4. 解码端按同样的参考顺序重建参考缓冲区，因此能够复现编码端使用的条件信息。

因此，双参考帧不是简单地把两个参考图像平均，而是通过模型内部的特征融合和残差概率建模共同影响当前帧的重建与码率估计。

## 4. 已修改和新增的主要文件

### `train_static_b.py`

新增静态 B 帧训练脚本，主要功能：

- 使用固定 9 帧 GOP。
- 使用 `STATIC_REF_MAP_9` 进行静态双参考帧训练。
- 第 0 帧通过 I 帧模型重建后放入 `ref_buffer`。
- 后续帧按静态参考表依次训练 P/B 帧模型。
- 支持 `--gpu-id` 或 `--cuda-visible-devices` 在脚本内选择显卡。
- 支持 `--resume` 断点续训。
- 断点续训时从 `p-Ep{N}-...pth` 中解析轮数继续计数。
- 支持 `--start-epoch` 手动覆盖起始 epoch。
- 加入 NaN/Inf 防护，遇到非有限 loss 或梯度时跳过当前 step。
- 加入 `--likelihood-lower-bound`，避免 likelihood 太小导致 `log(0)` 或数值爆炸。

### `test_static_b.py`

新增静态 B 帧测试脚本，主要功能：

- 使用和训练一致的 `STATIC_REF_MAP_9`。
- 测试时使用已重建帧作为后续帧参考。
- 支持 `--tau` 测试不同近无损约束。
- 支持 `--gpu-id` 或 `--cuda-visible-devices` 选择显卡。
- 支持自动避免覆盖已有日志文件。
- 日志 summary 中记录 I 帧模型、P/B 帧模型、tau、静态参考表、bpp、max error、编码时间、解码时间。
- 修复 `compressai` 不同版本中 `entropy_bottleneck.decompress()` 参数数量不一致的问题。
- 对 `tau > 0` 的残差条件网络使用训练时一致的条件层调用方式：

```python
rp_ctx = model.fusion_cond(torch.cat((rp, ctx), 1), cond)
lmm_params = model.residual_compressor_cond(rp_ctx, cond)
```

### `nll_model_b.py`

为 B 帧训练和测试补充兼容逻辑：

- 增加 `get_scale_table()`。
- 为 I/P 模型加载旧 checkpoint 时兼容旧版 entropy bottleneck key。
- 加载模型时更新 entropy bottleneck 和 gaussian conditional 的 buffer。
- 为有损压缩器补充 `compress()` 和 `decompress()`。
- 修复新版 `compressai` 下 entropy bottleneck 解码 API 的调用方式。

### `使用方法.md`

已记录静态 B 帧训练、测试、断点续训、显卡选择等使用方法。后续如果继续加入动态 B 帧或其它脚本，也需要同步更新该文件。

## 5. 训练命令

从头训练：

```bash
cd /data_a/wxy/linc/DLPR-Mosmed-nll

python train_static_b.py \
  --train-path ../Mosmed/train-full \
  --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b \
  --log-dir logs_static_b \
  --epochs 10000 \
  --batch-size 1 \
  --gpu-id 0
```

断点续训：

```bash
cd /data_a/wxy/linc/DLPR-Mosmed-nll

python train_static_b.py \
  --train-path ../Mosmed/train-full \
  --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --resume ./log_static_b/p-EpXXX-a-b-c.pth \
  --save-dir log_static_b \
  --log-dir logs_static_b \
  --epochs 10000 \
  --batch-size 1 \
  --gpu-id 0
```

如果需要手动指定起始轮数：

```bash
python train_static_b.py \
  --train-path ../Mosmed/train-full \
  --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --resume ./log_static_b/p-EpXXX-a-b-c.pth \
  --start-epoch XXX \
  --save-dir log_static_b \
  --log-dir logs_static_b \
  --epochs 10000 \
  --batch-size 1 \
  --gpu-id 0
```

## 6. 测试命令

例如测试 `tau=1`：

```bash
cd /data_a/wxy/linc/DLPR-Mosmed-nll

python test_static_b.py \
  --tau 1 \
  --data-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --p-frame-ckp ./log_static_b/p-Ep501-0.173-5.045-5.314.pth \
  --log-file log-static-b-tau-1.txt \
  --gpu-id 1
```

测试无损或更严格误差约束时，只需要修改 `--tau`：

```bash
python test_static_b.py \
  --tau 0 \
  --data-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --p-frame-ckp ./log_static_b/p-Ep501-0.173-5.045-5.314.pth \
  --log-file log-static-b-tau-0.txt \
  --gpu-id 1
```

## 7. 常见问题与已处理方案

### 数据路径问题

如果报错：

```text
No clips with at least 9 frames under ../Mosmed/test-slice
```

说明当前路径下的数据不满足 9 帧 GOP 输入格式。静态 B 帧脚本应优先使用：

```text
../Mosmed/train-full
../Mosmed/test-full
```

### checkpoint key 不兼容

旧 checkpoint 可能保存为：

```text
entropy_bottleneck._matrix0
entropy_bottleneck._bias0
entropy_bottleneck._factor0
```

而当前 `compressai` 版本期望：

```text
entropy_bottleneck.matrices.0
entropy_bottleneck.biases.0
entropy_bottleneck.factors.0
```

当前代码已在加载 checkpoint 时做 key 兼容转换。

### entropy buffer size mismatch

如果 entropy bottleneck 的 `_offset`、`_quantized_cdf`、`_cdf_length` buffer 尺寸不匹配，原因通常是 checkpoint 里保存了已 update 的 entropy buffer，而新建模型初始化时 buffer 为空。当前 `nll_model_b.py` 已在加载时先更新相关 buffer。

### NaN checkpoint

如果出现：

```text
p-Ep301-nan-nan-nan.pth
```

常见原因是 likelihood 太小、`log_likelihood` 非有限、loss 爆炸或梯度非有限。当前训练脚本已经：

- 对 likelihood 做下界裁剪。
- 对 NaN/Inf loss 跳过当前 step。
- 对 NaN/Inf 梯度跳过 optimizer step。
- 避免保存非有限评估指标。

如果仍频繁出现跳过 step，建议优先降低学习率，或增大 `--likelihood-lower-bound`，例如从 `1e-9` 调整为 `1e-8`。

### `.run()` 报错

如果测试时报：

```text
AttributeError: 'ResidualCompressor_cond' object has no attribute 'run'
```

说明服务器上的 `test_static_b.py` 仍是旧版本。当前版本已经不再调用 `.run()`。可在服务器检查：

```bash
grep -n "\.run(" test_static_b.py
```

正常情况下不应有输出。

## 8. 与动态参考帧方案的关系

`DLPR-Mosmed-nll-b` 中的 `train7.py` 实现了动态参考帧思想，主要函数包括：

- `build_adaptive_ref_map(num_frames, max_ref_gap=None)`
- `build_adaptive_level_ref_map(num_frames, max_ref_gap=None, level_merge=1)`
- `get_ref_map_by_level(level_ref_map, level)`
- `get_max_ref_level(level_ref_map)`

动态参考帧方案不是固定使用 9 帧表，而是根据序列长度构建参考层级：

1. 先编码第 0 帧作为 I 帧。
2. 选择右侧锚点。
3. 递归或分层选择中间帧。
4. 每个目标帧只允许引用已经重建好的低层级参考帧。
5. `level_merge` 可以合并层级，控制参考层级的细粒度。

当前 `train_static_b.py` 没有直接使用动态参考帧，而是先实现固定 GOP=9 的静态双参考帧版本。这样做的原因是：

- 改动范围更小，便于先验证双参考帧本身是否有效。
- 训练和测试顺序更容易保持一致。
- 出现 bpp、max error 或 NaN 问题时更容易定位。
- 后续可以在稳定静态版本的基础上，再扩展为动态参考帧。

## 9. 后续扩展建议

建议后续按以下顺序推进：

1. 先完成静态 B 帧与原 P 帧模型的公平对比：同一数据集、同一 tau、同一 I 帧模型、同一测试脚本口径。
2. 如果静态 B 帧有效，再实现 `train_dynamic_b.py` 和 `test_dynamic_b.py`。
3. 动态版本应复用 `train7.py` 中的参考表生成逻辑，但训练和测试必须使用同一套 ref map。
4. 动态版本测试时必须继续使用重建帧作为参考，不能使用原始帧。
5. 动态版本需要在日志中保存实际使用的 ref map、level_merge、gop_size，方便复现实验。
6. 若要比较静态和动态，建议分别保存到不同目录，例如：

```text
log_static_b/
logs_static_b/
log_dynamic_b/
logs_dynamic_b/
```

## 10. 当前状态

当前主分支已经包含静态 B 帧训练和测试基础功能。静态 B 帧版本适合继续做以下实验：

- 不同 `tau` 下的 bpp、PSNR、max error 对比。
- 与原单参考 P 帧模型对比。
- 与 `DLPR-Mosmed-nll-b` 中动态参考帧实现对比。
- 在静态版本稳定后，进一步迁移动态参考帧策略。

## 11. 静态 B 帧接入 Diff-SIT 有损压缩

在当前基础上新增了一套独立的 diffusion 静态 B 帧实验入口，用于验证“扩散模型替换有损压缩部分”是否能提升静态双参考 B 帧效果。

新增文件：

```text
nll_model_b_diff.py
train_static_b_diff.py
test_static_b_diff.py
```

设计原则：

- 不修改原来的 `nll_model_b.py`、`train_static_b.py`、`test_static_b.py`，保留无扩散静态 B 帧作为 baseline。
- 只替换 P/B 帧有损压缩器，残差概率模型仍使用原来的简单上下文和条件残差头。
- 训练和测试继续使用相同的固定 9 帧静态参考表，避免引入动态参考帧变量。

模型结构：

- `LossyCompressor_B_diff_sit` 参考原 P 帧 `LossyCompressor_P_diff_sit` 实现。
- 输入从单参考 `current, pre` 改为双参考 `current, ref_l, ref_r`。
- `Encoder_B_CDC` 分别编码当前帧、左参考帧、右参考帧，然后融合三个特征。
- diffusion U-Net 的参考条件使用 `DualRefEncoder`，将左右参考帧拼接为 2 通道后编码为多尺度条件特征。
- diffusion 输出当前帧的有损重建，同时 `ResPriorHead` 输出残差先验 `res_prior`。
- 残差编码仍沿用原来的 `mask_conv + fusion/fusion_cond + residual_compressor/residual_compressor_cond`。

训练命令：

```bash
cd /data_a/wxy/linc/DLPR-Mosmed-nll

python train_static_b_diff.py \
  --train-path ../Mosmed/train-full \
  --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b_diff \
  --log-dir logs_static_b_diff \
  --epochs 10000 \
  --batch-size 1 \
  --gpu-id 0
```

测试命令：

```bash
python test_static_b_diff.py \
  --tau 1 \
  --data-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --p-frame-ckp ./log_static_b_diff/p-EpXXX-a-b-c.pth \
  --log-file log-static-b-diff-tau-1.txt \
  --gpu-id 1
```

注意事项：

- 当前 diffusion B 帧版本是新实验入口，不应和无扩散 `log_static_b` checkpoint 混用。
- diffusion 版本 checkpoint 建议单独保存到 `log_static_b_diff/`。
- 测试日志会写入 `Lossy compressor: diff_sit_static_b`。
- 这一步只验证“扩散替换有损压缩”；如果之后要验证“残差概率头也使用扩散”，应继续新建独立实验，避免变量混杂。

## 多卡并行训练与 checkpoint 兼容

`train_static_b.py` 和 `train_static_b_diff.py` 现在支持单卡/多卡训练，并兼容单卡和 `DataParallel` 多卡保存的 checkpoint。

指定单卡训练：

```bash
python train_static_b.py \
  --train-path ../Mosmed/train-full \
  --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b \
  --log-dir logs_static_b \
  --batch-size 1 \
  --gpu-id 1 \
  --parallel none
```

指定多卡训练，例如使用 0、1 号卡：

```bash
python train_static_b.py \
  --train-path ../Mosmed/train-full \
  --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b \
  --log-dir logs_static_b \
  --batch-size 2 \
  --gpu-id 0,1 \
  --parallel auto
```

Diff-SIT 静态 B 帧版本同理：

```bash
python train_static_b_diff.py \
  --train-path ../Mosmed/train-full \
  --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b_diff \
  --log-dir logs_static_b_diff \
  --batch-size 2 \
  --gpu-id 0,1 \
  --parallel auto
```

说明：

- `--gpu-id` 会在脚本 import torch 前设置 `CUDA_VISIBLE_DEVICES`，可写成 `0`、`1`、`0,1`、`2,3` 等。
- `--parallel auto` 是默认值：可见 GPU 数大于 1 时自动使用 `torch.nn.DataParallel`，否则使用单卡。
- `--parallel none` 强制不启用多卡。
- `--parallel data_parallel` 强制尝试 `DataParallel`，如果只看到 1 张卡会打印 warning 并退回单卡。
- 多卡时建议 `--batch-size` 至少等于可见 GPU 数；如果 batch size 仍为 1，`DataParallel` 基本不会带来加速。
- 训练保存 checkpoint 时会自动保存未包裹的模型参数，因此新保存的模型不会带 `module.` 前缀。
- 加载 checkpoint 时仍会自动去掉已有的 `module.` 前缀，所以单卡模型和多卡模型可以互相 resume/test。

## 多卡训练显存注意事项

如果多卡训练仍在 I 帧编码阶段 OOM，请确认使用的是包含本节修改后的版本。当前训练脚本已经做了两项显存修复：

- I 帧模型也会按 `--parallel` 使用 `DataParallel`，不再只让 0 号可见卡承担 I 帧 patch 编码。
- `encode_i_frame()` 使用 `torch.no_grad()`，冻结的 I 帧模型不会保留反向传播计算图。

四张 24GB 卡建议先这样跑 Diff-SIT 静态 B 帧：

```bash
python train_static_b_diff.py \
  --train-path ../Mosmed/train-full \
  --eval-path ../Mosmed/test-full \
  --i-frame-ckp ckp_nll/0001/ckp.tar \
  --save-dir log_static_b_diff \
  --log-dir logs_static_b_diff \
  --epochs 10000 \
  --batch-size 4 \
  --gpu-id 0,1,2,3 \
  --parallel auto
```

如果仍然 OOM，先降到 `--batch-size 2`，或者临时设置：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

然后重新运行训练命令。
