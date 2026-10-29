# 交接包：Apollo（band-sequence audio restoration）复现与迁移

> 目标读者：接手这件事的**新会话/新 agent**，以及要在 **target-speaker BWE** 上复用该方法的你。
> 读这一份即可掌握：环境在哪、代码改了什么、结论是什么、哪些坑必须绕开、迁移要改哪些地方。
>
> 生成时间：2026-10-29　对应仓库 HEAD：`9070291`（fork `Maples0127/Apollo`）

---

## 0. 一页结论

**Q：现在能说明"文章这个方法合适"吗？**

**A：能说明"方法本身有效"，但不能说明"论文指标可复现"。** 分三层说：

| 结论层级 | 是否有证据 | 依据 |
|---|---|---|
| ① 方法可跑通、训练稳定 | ✅ 有 | 10 个 epoch 单卡训练无崩溃；损失/梯度/GAN 分量全部可观测 |
| ② 方法确实产生**感知**修复 | ✅ 有 | 50 条 held-out test 片段：PESQ **3.203 → 3.876（+0.67）**，同一模型 SI-SDR 13.69 → 13.23（−0.46 dB） |
| ③ 复现论文数字（SDR/SI-SNR/VISQOL、5000 samples/case） | ❌ 没有 | 只训了 10/500 epoch（2%）、1 卡、50 条评测、未跑 ViSQOL、未与 SR-GAN 对照 |

**一句话**：这套"分频带 + 频带轴 Roformer + 时间轴 TCN + 多尺度复数谱判别器"的设计**在感知指标上是站得住的**，且"GAN 用失真换听感"的行为与论文描述一致；但我们的训练规模只到论文的 2%，**平台期已出现**，不能对外宣称复现了论文表格。

**Q：能迁移到 target-speaker BWE 吗？**
**A：可以，而且适配点很明确**（见 §10）：主干（band-split + Roformer + TCN）与判别器、损失都能直接复用；必须新增**说话人条件化**、**低/高频分离的输入输出构造**、以及**带说话人相似度的评测协议**。

---

## 1. 项目是什么

复现 ICASSP 2025 *Apollo: Band-sequence Modeling for High-Quality Audio Restoration*
（[arXiv 2409.08514](https://arxiv.org/abs/2409.08514)，作者 Kai Li / Yi Luo，代码 `JusperLee/Apollo`）。

任务：把 MP3 压坏的 44.1 kHz 音乐还原（预测式音频修复），GAN 训练框架。

**关键澄清（容易误解）**：Apollo 的**主干不是 GAN**。生成器是
`STFT → 80 频带切分 → 6×(频带轴 Roformer + 时间轴 TCN) → 逐带 RI 输出 → iSTFT`；
GAN 只是训练时套在外面的判别器 + 对抗损失。推理只跑生成器（16.5 M 参数）。

---

## 2. 环境与路径（全部在节点本地 /data，不在 NFS /home）

| 内容 | 绝对路径 |
|---|---|
| 代码（服务器） | `/home/cse12212308/code/Apollo`（git，fork = `Maples0127/Apollo`） |
| 代码（本机） | `~/Desktop/Apollo` |
| conda 环境 | `/data/cse12212308/envs/look2hear` |
| 关键版本 | Python 3.10.13 / torch 2.0.0+cu118 / torchaudio 2.0.1 / pytorch-lightning 2.2.1 / pesq(后装) |
| 训练 HDF5 | `/data/cse12212308/data/hdf5_train`（**445 GB**，11 类 stem + mixture） |
| 原始数据 | `/data/cse12212308/data/musdb18hq`（100 train + 50 test）、`/data/cse12212308/data/moisesdb/moisesdb_v0.1`（240 首） |
| 验证集 | `/data/cse12212308/data/eval_holdout50`（50 条 MUSDB **test** 片段，码率 32/48/64/96 kbps） |
| 实验输出 | `/data/cse12212308/exp/Apollo/`（checkpoints/、config.yaml、eval_*.json、pesq_*.jsonl） |
| 历史 run | `/data/cse12212308/exp/Apollo_run39355/`（含 `REPORT.md`） |
| 训练日志 | `/data/cse12212308/exp/logs/apollo-*_<jobid>.out`（进度条）与 `.err`（Lightning 消息） |
| 最佳 checkpoint | `/data/cse12212308/exp/Apollo/checkpoints/epoch=8-step=45000-val_loss=-12.8842.ckpt` |

**节点本地盘**：`/data`（3.6 T NVMe）；`/home` 是 NFS（ailab01 挂载、5 节点共享，实测直读 107 MB/s、被递归扫描时掉到 5.8 MB/s）。

---

## 3. 代码状态与关键修复（都在 fork 里）

| commit | 内容 | 为什么重要 |
|---|---|---|
| `5da8da2` | 单卡/离线配置、smoke 配置、sbatch 脚本 | 原配置写死 8 卡 + wandb 在线 |
| `fcd5865` | 修 `validation_step` 解包（eval 返回 3 元组） | 原代码一跑验证就崩 |
| `2ccbb5c` | `build_eval` 复用 `codec_simu` 的**互相关对齐** | MP3 有 1105 样本延迟，不对齐时 val 恒为正、checkpoint 排序全错 |
| `a1539a5` | `preprocess_hdf5.py`（按 stem 分组，支持 MoisesDB 目录结构）+ `build_eval.py` | 上游脚本对 MoisesDB 会产出错误目录 |
| `8f384b7`/`a2e92f5` | 只把真正的 track 目录当 track；加 shard 并行 | 否则父目录被误判为 track |
| `5f21c11` | `train.py --resume`、8 步梯度累积、L_G 三分量日志 | 2 天时限下必须能续训；没有分量日志无法诊断 GAN |
| `2b8ac7c` | **关键 bug 修复**：L_G 反向时冻结判别器 + 回归测试 `tests/test_grad_accum.py` | 否则 dL_G/dD 会累加进 D 的梯度，D 被训练成常量输出 |
| `bc758ce` | 去掉 `every_n_train_steps` | 否则同名 checkpoint 互相覆盖、top-k 被同指标占满 |
| `5aa11c9` | sbatch 支持 `APOLLO_RESUME=1` | 一行续训 |
| `ac0e3b0`/`741713f` | PESQ 评测脚本（容忍 `NoUtterancesError`） | 只报 SI-SDR 会误判方法无效 |
| `9070291` | `job_status.sh` 汇总 PESQ/SI-SDR | 一键看状态 |

---

## 4. 数据管线（踩过的坑）

1. **MUSDB18-HQ**：Zenodo 直连很慢（0.5 MB/s）→ 用 24 路 range 并发下载（~67 MB/s），MD5 校验 `12d4f2ec…`。
2. **MoisesDB**：仓库里的 R2 链接已失效（`InvalidArgument: Authorization`）→ 改用 Kaggle 镜像 `boyazhangnb/moisesdb`（89 GB）。**注意其 MD5 与官方不同**（`ebff6554…` vs 官方 `13cf74eda…`，镜像重新打包），内容已验证（240 首、11 类 stem、44.1 kHz/stereo/PCM_24）。
3. **预处理**：用 `scripts/preprocess_hdf5.py`
   - MUSDB：`<track>/{mixture,vocals,drums,bass,other}.wav`
   - MoisesDB：`<track>/<stem>/<source-uuid>.wav` → 同 stem 多音源**求和**；**无 mixture.wav，需用各 stem 求和合成**
   - 输出 `<out>/<stem>/sampleN.hdf5`，每段 6 s、float64、`(n, 2, 264600)`
   - 全量 150 + 240 首 → **445 GB**，mixture 片段 22226 个
4. **上游脚本的两个坑**（已修）：
   - `moisesdb_preprocess.py` 用 `root.split("/")[-1]` 当 stem 名 → 实际会按 track 输出，datamodule 读不到；
   - 上游 `musdb_preprocess.py` 把 **train 和 test 一起写进训练目录** → 验证集泄漏。我们只用 `train/` 构建训练集。
5. **验证集**：`build_eval.py` 从 MUSDB **test** 取片段，用 `codec_simu` 编码（含延迟对齐），写 `ori_wav.wav` / `codec_wav.wav`。

---

## 5. 训练配置（与论文对齐 + 我们的等效实现）

| 项 | 论文 | 本复现 |
|---|---|---|
| 数据 | MUSDB18-HQ + MoisesDB | 同（11 类 stem） |
| batch | 8×GPU × 1 | **1 GPU × 8 步梯度累积（等效 8）** |
| 优化器 | AdamW，G 1e-3 / D 1e-4，wd 0.01 | 同 |
| LR 调度 | 每 2 epoch ×0.98 | StepLR(step_size=2, gamma=0.98) ✓ |
| 梯度裁剪 | 5 | ✓ |
| 损失权重 | α=β=γ=1 | ✓ |
| 模型 | STFT 20/10 ms，带宽 160 Hz，N=256，B=6 | ✓（80 频带：79×5 + 1×47） |
| 判别器 | 窗 [32…2048] | ✓（代码 I=7，论文写 5） |
| 每 epoch | 未给 | 20000 样本 ≈ **2.7 h**（实测 2:42~2:50） |
| epoch 数 | 500（early stop patience 20） | **实际 10 个（2 天时限）** |

**机器**：ailab02（RTX 5880 Ada 49 GB，单卡），Slurm 分区 `RTX5880Ada`，`--nodelist=ailab02`（数据在节点本地盘）。

---

## 6. 实验结果

### 6.1 10 个 epoch 的损失（epoch 平均值）

| epoch | L_rec | L_FM | L_adv | L_G | L_D | D(real) | D(fake) | gap |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.2996 | 0.0141 | 0.2548 | 0.5686 | 0.5020 | 0.4974 | 0.4965 | 0.0009 |
| 1 | 0.2651 | 0.0260 | 0.2530 | 0.5441 | 0.4980 | 0.5018 | 0.4979 | 0.0039 |
| 2 | 0.2178 | 0.0307 | 0.2530 | 0.5015 | 0.4981 | 0.5018 | 0.4979 | 0.0039 |
| 3 | 0.2414 | 0.0428 | 0.2548 | 0.5390 | 0.4966 | 0.5030 | 0.4965 | 0.0065 |
| 4 | 0.2120 | 0.0432 | 0.2543 | 0.5095 | 0.4970 | 0.5027 | 0.4969 | 0.0058 |
| 5 | 0.2439 | 0.0573 | 0.2561 | 0.5573 | 0.4954 | 0.5041 | 0.4955 | 0.0086 |
| 6 | 0.2391 | 0.0625 | 0.2564 | 0.5581 | 0.4951 | 0.5044 | 0.4953 | 0.0091 |
| 7 | 0.2089 | 0.0583 | 0.2552 | 0.5224 | 0.4962 | 0.5034 | 0.4963 | 0.0071 |
| 8 | **0.1891** | 0.0565 | 0.2547 | **0.5003** | 0.4964 | 0.5032 | 0.4966 | 0.0066 |
| 9 | 0.2337 | **0.0750** | 0.2571 | 0.5658 | 0.4942 | 0.5051 | 0.4948 | **0.0103** |

### 6.2 held-out 指标（50 条，codec 基线 SI-SDR 13.69 dB / PESQ 3.203）

| epoch | SI-SDR | ΔSI-SDR | PESQ | ΔPESQ |
|---|---|---|---|---|
| 1 | 13.08 | −0.61 | 3.824 | +0.621 |
| 5 | 13.25 | −0.44 | 3.873 | +0.671 |
| 8 | **13.23** | **−0.46** | **3.876** | **+0.673** |
| 9 | 13.10 | −0.59 | 3.841 | +0.638 |

### 6.3 训练中 val_loss（12 条，噪声大，仅供参考）

−12.665 / −12.777 / −12.640 / −12.619 / −12.709 / −12.783 / −12.791 / −12.812 / **−12.884** / −12.720

---

## 7. 诊断结论（最重要的一条）

**我们第一轮（run 39355）的"训练无效"是自造 bug，不是方法问题。**

- 现象：`D(real)` 与 `D(fake)` **精确相等**（0.6467/0.6467），`L_FM → 0.0002`，模型学成"近恒等映射"，SI-SDR 比 codec 还低 0.72 dB。
- 根因：8 步梯度累积实现里，`L_G` 每 micro-batch 都反传，而 `L_G` 依赖判别器参数 → `dL_G/dD` 被累加进 D 的梯度缓冲，D 被训练去**最小化 L_G**（即对真假都给同一个高分）。
- 修复：G 反向时 `discriminator.requires_grad_(False)`，并加回归测试 `tests/test_grad_accum.py`。
- 修复后：`D(real)−D(fake)` 从 0.0009 涨到 0.010，`L_FM` 0.014 → 0.075，PESQ 由负转正（+0.67）。

**次重要**：判别器仍偏弱（gap ≈ 0.007–0.010），`L_adv` 恒定在 0.255 附近 → 优化几乎全由 `L_rec` 驱动。这可能是 epoch 5–8 平台期的原因，候选解法见 §10.4 / 原报告 H2。

---

## 8. 已知坑与防护（务必遵守）

1. **禁止对 `/home` 递归搜索**（`grep -r`、`find`、`du -s`）。曾有一次审计命令 `grep -rl "Resuming from" /data/... /home/...` 遗留运行 **11 小时**、读 31.5 GB，把 NFS 直读从 107 MB/s 压到 5.8 MB/s。
   - 现有一层 **ssh shim**（`~/.pi/agent/bin/ssh`）：拦截 `…/home` 的递归扫描，放行 `…/home/*/code`，并写 `~/pi-remote-ops.log`；
   - `rssh` 提供远端 `timeout`（默认 120 s，`RSSH_TIMEOUT=` 调）与审计日志。
2. **pi-web 会话不加载 pi-permission-system 扩展**（实测：deny 探针仍执行、debugLog 无文件、review 日志停在 9-10）。保密目录的 deny 规则在 pi-web 里**不生效**，需在 Admin1/TUI 会话里才有。
3. **checkpoint 命名**：不要用 `every_n_train_steps` + 单一 `monitor`，会生成 `-v1..vN` 重复文件并把早期 epoch 挤掉；用 `{epoch}-{step}-{val_loss:.4f}`。
4. **PESQ 会抛 `NoUtterancesError`**（音乐片段被判无语音），必须逐条 try/except 记 NaN 并报告覆盖率。
5. **PESQ 是语音指标**；论文用 **ViSQOL**。不要拿 PESQ 绝对值与论文表格比。
6. GPU 只能经 Slurm：`sbatch --nodelist=ailab02`（数据在节点本地盘，换节点要复制 445 GB）。
7. 队列经常满载；24 h 请求很难被 backfill，6 h 请求更容易排上（配合 `--resume`）。

---

## 9. 复现命令（照抄）

```bash
# 状态快照 / 保活
ssh ailab02 'bash ~/code/Apollo/scripts/job_status.sh 39556'
ssh ailab02 'bash ~/code/Apollo/scripts/job_heartbeat.sh 39556 20'

# 从最新 checkpoint 续训（6 h）
ssh ailab02 'cd ~/code/Apollo && APOLLO_RESUME=1 sbatch --time=06:00:00 --mem=24G \
  --cpus-per-task=6 --export=ALL,APOLLO_RESUME=1 --job-name=apollo-resume \
  scripts/train_apollo.sbatch configs/apollo.yaml'

# 50 条 held-out 的 SI-SDR
ssh ailab02 'cd ~/code/Apollo && /data/cse12212308/envs/look2hear/bin/python \
  scripts/eval_checkpoint.py --ckpt <ckpt> --eval-dir /data/cse12212308/data/eval_holdout50'

# 50 条 held-out 的 PESQ + SI-SDR
ssh ailab02 'cd ~/code/Apollo && /data/cse12212308/envs/look2hear/bin/python \
  scripts/eval_pesq.py --ckpt <ckpt> --eval-dir /data/cse12212308/data/eval_holdout50 \
  --out /tmp/pesq.jsonl'

# 日志 → CSV → 曲线
ssh ailab02 'python ~/code/Apollo/scripts/parse_train_log.py <log> -o /tmp/loss.csv'
python scripts/plot_loss.py /tmp/loss.csv -o loss.png
```

---

## 10. 迁移到 target-speaker BWE：改什么、怎么改

### 10.1 可以直接复用的部分（约 70% 代码）

- `look2hear/models/apollo.py` 的 **频带切分 + BSNet(band Roformer / time TCN) + 逐带 RI 输出**；
- `look2hear/discriminators/frequencydis.py` 的**多尺度复数谱判别器**；
- `look2hear/losses/gan_losses.py` 的 `L_rec / L_FM / L_adv` 组合；
- 工程件：分布/续训/评测脚本、`preprocess_hdf5.py`（把 BWE 数据也做成 HDF5）、`job_status.sh`。

### 10.2 必须新增/改造

| 方面 | 现状（Apollo） | target-speaker BWE 需要 |
|---|---|---|
| 输入/输出 | 同采样率、同带宽 | **窄带输入（如 16 kHz）→ 全带输出（44.1/48 kHz）**；输入上采样后高频置零，让模型"生成"高频，而不是"修复" |
| 条件化 | 无 | **说话人 embedding**（ECAPA/x-vector）经 FiLM 注入每个 BSNet 的逐带特征：`h ← h·(1+γ) + β`；或 cross-attention |
| 判别器 | 全带多尺度 | 增加**高频加权**或单独的高频判别器（BWE 的失真集中在高频） |
| 损失 | L_rec 全带 | 加 **低频一致性项**（$\lVert \hat{y}_{\text{low}}-y_{\text{low}}\rVert$，防止改坏低频）与**说话人一致性项**（输出的 ECAPA 与目标 speaker 的余弦相似度） |
| 训练目标 | 从压缩稿恢复 | 输入是**混合语音**（含干扰说话人）+ 注册语音，输出是**目标说话人的全带语音** → 本质接近"目标说话人提取 + BWE"，需要 permutation-free / 目标指定式构造 |
| 指标 | SI-SDR、PESQ | **高频 SI-SDR / LSD / HF 能量恢复**、PESQ-wb、ViSQOL、**说话人相似度**、以及**干扰人泄漏（SIR/SI-SDR vs 干扰源）** |

### 10.3 建议的数据与实验顺序

1. **先把 BWE 单独跑通**（不做目标说话人）：VCTK/LibriTTS 全带 → 低通+降采样到 16 k → 上采样回 48 k 作为输入；
2. 加 **speaker conditioning**：enrollment 用同一说话人的另一条语句，先做"说话人条件 BWE"，不做干扰人；
3. 再加 **混合与干扰人**（target-speaker 设定），用 speaker-disjoint 划分；
4. 每步都用**同一套 50–200 条 held-out 协议**，主指标从第一天就同时报 SI-SDR(HF)、PESQ、speaker-sim，避免像这次一样被单一 SI-SDR 误导。

### 10.4 从本次复现直接带走的教训

- **一定先验证"验证集本身是对的"**：我们浪费了一整轮训练，因为 MP3 延迟没对齐，val 是噪声（这次 +1105 样本延迟）。BWE 里的重采样同样有群延迟，必须显式对齐后再算 SI-SDR。
- **验证集 ≥ 50 条**，否则 checkpoint 选择等于抽奖（12 条时标准误 ≈ 0.7 dB）。
- **只报 SI-SDR 会误判**：本次 PESQ +0.67 而 SI-SDR −0.46；两种指标必须一起看。
- **GAN 的 D 很脆**：手动优化 + 梯度累积时，务必在 G 反向时冻结 D（我们踩过，见 `tests/test_grad_accum.py`）。
- **判别器太弱会撞平台期**：`L_adv` 恒定、`D(real)−D(fake)<0.01` 时，考虑 D lr 2e-4、R1 正则或 hinge loss。
- **生成器容易学成近恒等映射**：BWE 里建议**残差输出**（`ŷ = x + Δ`）或显式低/高频分离，避免模型只学"复制输入"。

### 10.5 风险提示

- Apollo 是为"盲修复"设计的，**没有任何说话人条件**；迁移不是改个 config，而是要在 BSNet 里插入条件化模块并重训。
- 论文的 5000 samples/case 评测我们没做；如果你要写论文，**评测规模要按你自己的目标重新设计**，不要照搬我们这 50 条。
- 本次复现只跑 10/500 epoch，**平台期结论只对"这个规模"成立**；迁移后请先做小规模 ablation（有无 D、有无 FM、有无 speaker 条件）再决定投入。

---

## 11. 遗留 TODO（接手后可直接做）

1. 用 epoch 8 checkpoint 在 **200–500 条** test 片段上出正式数字（SI-SDR/PESQ，可选 ViSQOL）；
2. 若要继续 Apollo：按 H2 把 D lr 提到 2e-4 或加 R1，从 epoch 8 续训 1 天对比；
3. `eval_holdout50` 已设为训练验证集；考虑扩到 100 条并把 PESQ 加进每个 epoch 的验证回调；
4. 磁盘：`/data` 剩 ~838 GB；可清理 `musdb18hq.zip`(22 G)、`moisesdb.zip`(83 G)（原始目录已解压并验证）。
