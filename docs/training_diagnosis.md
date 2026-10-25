# Apollo 训练诊断报告（2026-10-26）

本文档对应 run `39355`（6 小时 Slurm 任务，2 个完整 epoch），目的是判断
"开源代码能否正常运行、训练是否有效果"。所有数字都可用文末命令复现。

## 1. 环境与配置

| 项 | 值 |
|---|---|
| 数据 | MUSDB18-HQ `train/` 100 首 + MoisesDB 240 首 → `hdf5_train`（445 GB，11 类 stem，mixture 22226 片段） |
| 验证集 | `eval_holdout`，12 条 MUSDB **test** 片段（32/48/64/96 kbps），与训练无重叠 |
| 模型 | Apollo，B=6，N=256，win=20 ms，80 频带 |
| batch | 1 卡 × 8 步梯度累积（等效论文 8 卡 × 1） |
| 优化 | AdamW，G 1e-3 / D 1e-4，wd 0.01，梯度裁剪 5，StepLR 每 2 epoch ×0.98 |
| 损失 | L_rec + L_FM + L_adv，α=β=γ=1 |
| 每 epoch | 20000 样本 |

## 2. 本轮修复的工程问题

1. **验证集 codec 对齐**：MP3 编解码有固定 1105 样本（25.1 ms）延迟。旧
   `build_eval.py` 未做对齐，导致 codec-vs-ori 的 SI-SDR 是 −38 ~ −18 dB，
   于是 `val_loss` 恒为正（+20 ~ +29）、checkpoint 排序完全失效。改为复用
   `codec_simu` 的互相关对齐后，codec 基线恢复为 **13.28 dB**（12 条中 11 条 >8 dB，
   剩下 1 条 7.21 dB 是 32 kbps 片段本身的下限）。
2. **`--resume`**：从最新 `last*.ckpt` 恢复（模型/优化器/调度器/callback 状态），
   日志打印 `epoch` 与 `global_step`。已用真实 232 MB checkpoint 验证：
   `Resuming from … | epoch=1 global_step=10000`，随后从 epoch 2 继续。
3. **checkpoint 命名**：文件名带 `{step}`，删除 `every_n_train_steps`，
   避免同名 `-vN` 堆积和 top-k 互相挤占；旧 run 的 3.7 GB 文件归档到
   `checkpoints_old/`。
4. **损失分量日志**：每个 epoch 末打印 `L_rec / L_FM / L_adv` 与
   `D(real) / D(fake)`，可直接判断 GAN 是否在工作。
5. **超参对齐论文**：见上表；`num_samples` 40000 → 20000，实测单 epoch **2:43**。

## 3. 结果

| epoch | 时长 | val_loss（−SI-SDR） | L_rec | L_FM | L_adv | D(real) | D(fake) |
|---|---|---|---|---|---|---|---|
| 0 | 2:42:56 | −12.12 | 0.3607 | 0.0036 | 0.1272 | 0.6467 | 0.6467 |
| 1 | 2:43:19 | −12.41 | 0.2385 | 0.0002 | 0.1211 | 0.6520 | 0.6520 |

held-out 评估（`eval_39355_epoch1.json`）：codec 基线 **13.28 dB**，模型
**12.41 dB**（Δ = −0.86 dB，12 条逐条均为负）。

**结论**：代码链路正常、重建项在稳定下降（L_rec −34%，val 提升 0.29 dB），
但**尚未学到"修复"**：输出仍比压缩输入更差。

## 4. 根因：梯度累积污染了判别器

`L_G` 不仅依赖生成器，也依赖判别器（`D(fake)` 与特征图）。在本轮的累积实现里，
每个 micro-batch 都对 `L_G` 做反向，而 `zero_grad` 只在每 8 步的第一个
micro-batch 执行——于是 `dL_G/dD` 被累加进 D 的梯度缓冲，并在第 8 步由 D 的
优化器施加。**结果 D 被训练去最小化 L_G**，即被推向"对真假都给同一个高分"。

证据（三条独立且互相印证）：

1. 日志里 `D(real)` 与 `D(fake)` **精确相等**（0.6467/0.6467，0.6520/0.6520）；
   用 D loss 反推 `(0.6520−1)² + 0.6520² = 0.5462`，与记录的 0.5464 吻合，
   说明 D 对所有输入输出同一个常数。
2. `L_FM` 从 0.0036 掉到 0.0002 —— D 的中间层特征对真假不再有差异。
3. 修复后（G 反向时冻结 D）回归测试通过：
   `tests/test_grad_accum.py` → `OK: D grads untouched by L_G backward`。

因此，run 39355 的 "判别器恒输出" 是本轮实现的 bug，**不是** Apollo 方法本身的性质；
该 run 也不构成对论文 GAN 配方的有效验证。

## 5. 下一步假设与建议（不盲目续跑）

- **H1（首要）**：修复后 D 应恢复判别力。判据：新 run 的第一个 epoch 末
  `D(real) ≠ D(fake)` 且 `L_FM > 0`，同时 `L_adv` 应随训练下降。
  若满足，说明 GAN 生效，继续训练才有意义；若仍相等，进入 H2。
- **H2**：若修复后仍塌陷，按代价从低到高尝试
  (a) D 学习率 1e-4 → 2e-4；
  (b) R1 梯度惩罚（对 real 样本的梯度惩罚，LSGAN + spectral norm 常用解）；
  (c) 换 hinge loss。
  注意论文用 8 卡，单卡即便等效 batch 8，BN/统计与梯度噪声仍不同，需单独观察。
- **H3（指标）**：SI-SDR 只衡量波形保真度，GAN 的目标是感知质量。下一步应同时报告
  PESQ / ViSQOL（论文报的是 SDR/SI-SNR/VISQOL）。可能出现 "SI-SDR 略降但感知明显变好"
  的情况，不能只用 SI-SDR 判死刑。
- **H4（模型侧）**：若修复 GAN 并训练若干 epoch 后 SI-SDR 仍低于 codec 基线，再考虑
  结构改动：生成器残差输出（`y = x + Δ`）、或增加波形域/mel 重建项。**在这些之前不要改结构**，
  否则无法归因。
- **不要**用 run 39355 的 checkpoint 继续训练：其 D 权重已被错误目标污染，
  且 `checkpoints_old/` 里 8 epoch 的旧权重同样来自坏评估/坏目标。

## 6. 复现命令

```bash
# 解析日志 → CSV → 曲线
ssh ailab02 'python ~/code/Apollo/scripts/parse_train_log.py \
  /data/cse12212308/exp/logs/apollo-full_39355.out -o /tmp/loss.csv'
python scripts/plot_loss.py /tmp/loss.csv -o loss.png

# held-out 评估（codec 基线 vs 模型）
ssh ailab02 'python ~/code/Apollo/scripts/eval_checkpoint.py \
  --ckpt /data/cse12212308/exp/Apollo/checkpoints/epoch=1-step=10000-val_loss=-12.4123.ckpt \
  --eval-dir /data/cse12212308/data/eval_holdout'

# 梯度污染回归测试
ssh ailab02 'python ~/code/Apollo/tests/test_grad_accum.py'
```
