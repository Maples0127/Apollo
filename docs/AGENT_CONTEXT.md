# AGENT_CONTEXT — Apollo 训练环境地图（给所有 agent 与人看）

> 目的：让**任何** agent（主 agent、审计 agent、以后新开的会话）不必猜路径、
> 不必全盘搜索，就能在 10 秒内拿到"东西在哪、现在什么状态、怎么验证"。
> 修改本文件请同步 push 到 `Maples0127/Apollo`，并在服务器上更新副本
> `/data/cse12212308/AGENT_CONTEXT.md`。

最后更新：2026-10-26

---

## 1. 这个项目在做什么

复现 ICASSP 2025 论文 *Apollo: Band-sequence Modeling for High-Quality Audio
Restoration*（[arXiv 2409.08514](https://arxiv.org/abs/2409.08514)，代码
`JusperLee/Apollo`）：把 MP3 压坏的 44.1 kHz 音乐还原。当前阶段是**验证开源代码
能否跑通、训练是否有效果**，不是追论文指标。

训练数据 = MUSDB18-HQ `train/`（100 首）+ MoisesDB（240 首），11 类 stem。

---

## 2. 路径地图（**全部在节点本地 /data，不在 NFS /home**）

| 内容 | 绝对路径 |
|---|---|
| 代码（服务器） | `/home/cse12212308/code/Apollo` （git，fork=`Maples0127/Apollo`） |
| 代码（本机） | `~/Desktop/Apollo` |
| conda 环境 | `/data/cse12212308/envs/look2hear`（Python 3.10 / torch 2.0.0+cu118） |
| 训练 HDF5 | `/data/cse12212308/data/hdf5_train`（445 GB，11 stem + mixture） |
| 验证集 | `/data/cse12212308/data/eval_holdout`（12 条 MUSDB **test** 片段） |
| 实验输出 | `/data/cse12212308/exp/<exp.name>/`（checkpoints/、logs/、config.yaml） |
| 训练日志 | `/data/cse12212308/exp/logs/<jobname>_<jobid>.out`（进度条）与 `.err`（Lightning 消息） |
| 历史 run | `/data/cse12212308/exp/Apollo_run39355/`（含 `REPORT.md`、`eval_*.json`） |

训练脚本与配置：`train.py`、`configs/apollo.yaml`、`scripts/train_apollo.sbatch`。

---

## 3. 状态与验证（**只需这几条**，都在 /data 上，秒级）

```bash
# 一键快照：队列 / 最新进度 / epoch 分量 / checkpoint / 评估
bash ~/code/Apollo/scripts/job_status.sh          # 或传 jobid

# 只取最后一行进度
tail -c 2000 /data/cse12212308/exp/logs/apollo-fixed_39460.out \
  | tr '\r' '\n' | grep -E "Epoch [0-9]+:" | tail -1

# epoch 末的 G/D 分量（L_rec / L_FM / L_adv / D(real) / D(fake)）
tr '\r' '\n' < /data/cse12212308/exp/logs/apollo-fixed_39460.out \
  | grep -ao "\[epoch [0-9]*\][^[]*" | tail -3

# held-out 评估（codec 基线 vs 模型）
python ~/code/Apollo/scripts/eval_checkpoint.py \
  --ckpt <ckpt 路径> --eval-dir /data/cse12212308/data/eval_holdout

# 日志 → CSV → 曲线
python ~/code/Apollo/scripts/parse_train_log.py <log> -o /tmp/loss.csv
python ~/code/Apollo/scripts/plot_loss.py /tmp/loss.csv -o /tmp/loss.png
```

判读要点：
- `val_loss` 是**负 SI-SDR**（越负越好）；codec 基线约 `-13.28`。
- `D(real)` 与 `D(fake)` **精确相等** ⇒ 判别器塌陷，对抗项无效。
- `L_FM` 接近 0 ⇒ 判别器特征对真假无区分。

---

## 4. 关键结论（截至 2026-10-26）

- run `39355`（旧实现）：2 epoch，val −12.12 → −12.41，但 `D(real)==D(fake)`，
  根因是**梯度累积时 L_G 反向往 D 累加了梯度**（已在提交 `2b8ac7c` 修复并加回归测试
  `tests/test_grad_accum.py`）。详细分析见 `docs/training_diagnosis.md`。
- run `39460`（修复后）：进行中，用于验证修复是否让判别器恢复判别力。

---

## 5. 硬规则（误解这些会伤到集群或产出错误结论）

1. **禁止对 `/home` 做递归搜索**（`grep -r`、`find`、`du -s`、`tar`）。
   `/home` 是 NFS（ailab01 挂载，5 节点共享），实测直读 107 MB/s，被递归扫描时
   掉到 5.8 MB/s；曾有一次 `grep -rl` 遗留 11 小时。需要搜就用具体路径 +
   `--include='*.log'`，或直接读上面的日志文件。
2. **注意：pi-web 会话不加载 pi-permission-system 扩展**（2026-10-26 实测：
   deny 探针仍执行、`debugLog` 打开后没有生成日志、review 日志停在 9-10）。
   因此 `config.json` 里的规则（包括保密目录 deny）**在 pi-web 会话中不生效**；
   在 Admin1/TUI 会话中才生效。pi-web 里的实际护栏是
   `~/.pi/agent/bin/ssh` 这层 shim（位于 agent 的 PATH，拦截 /home 递归扫描，
   但放行 `…/code` 下的代码搜索），以及下面的 `rssh`。

3. **远程命令走 `rssh`**（本机，`~/bin/rssh`，已软链到 `~/.pi/agent/bin/rssh`
   因此在 pi 的 bash 里可直接调用）：内置远端 `timeout`（默认 120 s，
   `RSSH_TIMEOUT=600` 可调）、递归扫描拦截、以及 `~/pi-remote-ops.log` 审计日志。
   pi 权限系统另有同名 deny 规则（`~/.pi/agent/extensions/pi-permission-system/config.json`），
   但**配置在会话启动时加载**，改完要重启 pi 才生效（实测热改无效）。
4. **破坏性操作先问**（删除数据/checkpoint、覆盖 run 目录、改 git 历史）。
5. **不要在集群上跑 CPU 训练冒充验证**；GPU 只能通过 `sbatch`（分区
   `RTX5880Ada`，固定 `--nodelist=ailab02`，数据在节点本地盘）。
6. 结论必须带**可复核证据**：文件路径 + 行数/数值 + 命令。

---

## 6. 给审计 agent 的验证清单（照抄即可，全部 bounded）

```bash
# 1) 代码版本
ssh ailab02 'cd ~/code/Apollo && git log --oneline -3 && git status -s'

# 2) 修复是否在生效代码里
ssh ailab02 'cd ~/code/Apollo && grep -n "requires_grad_(False)" look2hear/system/audio_litmodule.py'

# 3) 回归测试（证明 L_G 反向不再污染 D）
ssh ailab02 'cd ~/code/Apollo && python tests/test_grad_accum.py'

# 4) 训练状态与分量
ssh ailab02 'bash ~/code/Apollo/scripts/job_status.sh'

# 5) 评估证据
ssh ailab02 'cat /data/cse12212308/exp/Apollo_run39355/eval_39355_epoch1.json'

# 6) 无参数、无 /home 扫描：确认没有遗留的大范围扫描进程
ssh ailab02 'ps -eo pid,etime,cmd | grep -E "[g]rep -r|[f]ind /home" || echo none'
```

---

## 7. 常用操作

```bash
# 提交（6 h，便于回填；2 天上限也可）
ssh ailab02 'cd ~/code/Apollo && sbatch --time=06:00:00 --mem=24G --cpus-per-task=6 \
  --job-name=apollo-fixed scripts/train_apollo.sbatch configs/apollo.yaml'

# 续训（从 exp 目录里最新的 last*.ckpt）
ssh ailab02 'cd ~/code/Apollo && python train.py --conf_dir=configs/apollo.yaml --resume'

# 取消
ssh ailab02 'scancel <jobid>'
```
