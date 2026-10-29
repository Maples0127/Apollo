# 新会话开场提示（可直接粘贴）

---

我要在集群上做 **target-speaker BWE（目标说话人带宽扩展）**，方法是把 Apollo
（band-sequence audio restoration，ICASSP 2025）的骨干迁移过来。

**先读交接包，再动手：**

- `~/Desktop/Apollo/docs/HANDOVER.md`（完整交接：环境、代码改动、结果、坑、迁移方案）
- `~/Desktop/Apollo/docs/handover/facts.json`（机器可读摘要）
- 服务器同步副本：`ailab02:/data/cse12212308/AGENT_CONTEXT.md`

**已知上下文（不要重复踩坑）：**

1. 集群：Slurm，分区 `RTX5880Ada`，**必须 `--nodelist=ailab02`**（训练数据在节点本地盘
   `/data/cse12212308/data`）；`/home` 是 NFS，**禁止 `grep -r /home`、`find /home`**
   （曾有一次遗留 11 小时，把 NFS 从 107 MB/s 压到 5.8 MB/s）。远程操作走 `rssh`。
2. 环境：`/data/cse12212308/envs/look2hear`（Python 3.10 / torch 2.0.0+cu118 / PL 2.2.1）。
3. 代码：`~/code/Apollo`（服务器）/ `~/Desktop/Apollo`（本机），HEAD `9070291`。
   已包含：`--resume` 续训、8 步梯度累积（等效 batch 8）、损失三分量日志、
   多尺度复数谱判别器、以及**修复过的 GAN 梯度污染 bug**（`tests/test_grad_accum.py`）。
4. 现有 Apollo 复现结论：10 个 epoch 后 held-out 50 条上 **PESQ +0.67**、
   **SI-SDR −0.46 dB**（相对 codec 输入），即"感知向有效、波形保真略降"，已进入平台期。
   **不能**说复现了论文指标（论文 500 样本/场景 × 5000 条评测）。

**我的目标：**

把 Apollo 的生成器（80 频带切分 → 频带轴 Roformer → 时间轴 TCN → 逐带复数谱输出）
改造成 **target-speaker BWE**：

- 输入：目标说话人的**窄带混合语音**（含干扰人）+ **enrollment 语音**；
- 输出：目标说话人的**全带语音**（44.1/48 kHz）；
- 需要新增：说话人条件化（FiLM 或 cross-attention）、低频一致性约束、
  高频判别/加权、以及 **speaker-similarity / 干扰人泄漏** 的评测；
- 数据：VCTK/LibriTTS（+ VoxCeleb enrollment），speaker-disjoint 划分，
  窄带用低通+降采样模拟。

**请你：**

1. 先给出**迁移方案与实验计划**（架构改动点、数据构造、指标、baseline、
   先做哪个 ablation），不要直接写代码；
2. 明确**风险与失败判据**（比如：SI-SDR 不升但 PESQ 升时如何判定；
   判别器塌陷如何监控）；
3. 计划确认后再动代码，并按"改一处、验证一处"推进。

---
