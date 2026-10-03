# RL Infra 系统调查：现代 RL Post-Training 的设计空间（2026-10）

> RL-infra 报告的相关工作底稿，也是本目录"工业对照"的一部分。来源为 2026-10 三轮公开资料
> 调查，关键结论附链接；这个方向每月都在变，时效敏感。
> 目的：① 为 RL-infra 报告的定位提供坐标；② 修正我们计划里两处不准确的表述（见 §1.2）。

## 0. 背景：为什么这成了最卷的系统方向

- **生成占训练时间 80–90%**（verl 论文的观察，被多篇综述引用）；我们实测 **71%**
  （generate 68s / step 96s）——同一个结构性事实，规模无关。
- 同步 RL 的两个系统病：① **straggler barrier**（等最长序列才算完一个 batch）；
  ② 生成与训练**互斥空转**（一边跑另一边闲着）。
- 2025–2026 的应对路线：共居 + sleep（OpenRLHF hybrid engine）→ 部分重叠
  （pause/resume）→ **全异步 + staleness 控制**（AReaL，NeurIPS 2025，2.77×）→
  **单 rollout 异步**（FlashREINFORCE，NVIDIA，2026-09，rollout 成本减半）。
- 另一条线是 **agentic RL**：rollout 变成多轮、长尾、带环境沙箱的长任务
  （SkyRL-Agent / ROLL / ProRL Agent），瓶颈从"批量生成"变成"沙箱 + 调度 + 尾部延迟"。

## 1. 设计空间：五条轴

骨架来自 [Anatomy of RL Frameworks](https://www.hanifleo.com/anatomy-of-rl-frameworks)
的框架拆解（OpenRLHF / verl / slime / Verifiers / AReaL 五家对照），此处合并我们的位置：

| 轴 | 取值 | 我们 rig 的位置 |
|---|---|---|
| ① Rollout 架构 | Engine（同进程/Ray placement group 共居）↔ Server（独立服务，HTTP/RPC） | **Server**（HTTP + 本地权重推送） |
| ② 权重同步 | resharding / NCCL / **CUDA IPC** / 版本化 / 显式 API / 中间件 | NCCLWeightTransferEngine（全量 2.9GB/步） |
| ③ 同步 vs 异步 | 全同步 → 部分重叠 → 全异步（+staleness 控制） | **全同步**（E7 的 32× off-policy 是离线模拟） |
| ④ 单轮 vs 多轮 | 单轮生成 ↔ 多轮环境交互（沙箱/工具） | 单轮（GSM8K） |
| ⑤ 调度编排 | Ray placement group / 单控制器 / 自研 | 自研启发式队列（进程外） |

### 1.1 轴 ①③ 的含义

- **Engine 模式**：同步便宜（内存直拷 / resharding 近零开销），但训练与生成的资源捆绑；
- **Server 模式**：解耦、故障隔离、可异构，但每次同步要跨进程——vLLM 的
  `pause → update_weights → reset_prefix_cache → resume` 就是这条路的代价；
- 我们的 rig = Server + 全量 NCCL 广播 = **最朴素的一角**。

### 1.2 轴 ② 的详情（含两处纠错）

| 机制 | 代表 | 要点 |
|---|---|---|
| resharding（device mesh） | verl engine 模式 | `torch.distributed.redistribute` 做 3D 张量重分布、原位改造、近零开销；**仅共居模式可行** |
| **direct broadcast** | OpenRLHF | Ray/NCCL 广播；**多卡走 NCCL，同机走 CUDA IPC** |
| 版本化更新 | AReaL | 权重带版本号 + 陈旧度追踪，rollout worker 按需拉取；旧样本交给 staleness-aware PPO |
| 显式 update API | slime | `update_weight()` + SGLang server API；Megatron→HF 经 mbridge |
| 同步中间件 | MoonshotAI [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine) | 引擎无关；1T 参数、数千卡 **~20 秒**；Kimi-k2 在用 |
| 直连 GPU↔GPU | LlamaRL DDMA / NVSHMEM | 绕过 host；GPU 富余时的高级选项 |

**纠错 ①**：我们计划里写的"colocate 必须换掉 NCCL（同卡两 rank 被禁）"不够准确——
工业界的表述是 **"同机共居用 CUDA IPC，跨卡才用 NCCL"**；"同卡两 rank"限制针对的是
我们现用的 NCCL 通信组建法，不是同步本身。

**纠错 ②**：**vLLM 的 sleep mode 就是为 RLHF 权重更新设计的**（官方文档原文）——
level 1：权重退到 CPU、丢 KV；**level 2：权重与 KV 全丢，唤醒后
`collective_rpc("reload_weights")` 恢复**。§5.3 的"同卡 colocate"方案直接用它，
不需要自创机制。

## 2. 系统卡片

### OpenRLHF（和我们的相似度最高）
Ray + vLLM + ZeRO-3。**Hybrid Engine**：Actor/Critic/Reward/Reference/vLLM **同卡共居、
sleep mode 时分复用**（生成时 vLLM 醒、DeepSpeed 睡；训练时相反）——这是"同卡两引擎"
的标准工业答案。权重同步 `--vllm.sync_backend nccl`（同机 CUDA IPC）；用
**vLLM pause/resume** 把权重同步叠进生成（与我们 `vllm_utils` 的原语一一对应）。
现已自称 **Agentic RL Framework**（多轮 agent 管线、remote RM、Async & partial rollout）。
→ RL-infra 报告对照节的主对象；§5.3 照它的 hybrid engine 改。

### verl（字节）
默认 engine 模式；`ActorRolloutRefWorker` 单类多角色；resharding 同步；agentic 走 server
（`rollout.mode=async`）。**"生成占 80–90%"的出处**。→ GRPO 报告 related work 已引；RL-infra 报告里作
resharding 路线代表。

### AReaL（蚂蚁 + 清华 IIIS，NeurIPS 2025）
**全异步**：rollout 池持续生成、训练池持续更新，无配对；staleness 控制 +
staleness-aware PPO；同卡数下 **最高 2.77×**、最终性能不降。→ E7 的产业终点；我们的
"gap 指标"就是这套术语里的陈旧度测量仪。

### slime（THUDM/智谱，GLM-5 系在用）
SGLang + Megatron 紧耦合；**Trainer/Rollout/Data Buffer 三模块严格分离，训练永不直接调
推理引擎**（checkpoint 通知制）；`--colocate` 一个开关切换共居/分离，同步↔异步只是挪一下
`ray.get`。→ server 模式里和我们最像的一家；"通知制 vs 每步推送"是两种同步哲学。

### SkyRL / SkyRL-Agent（Berkeley NovaSky）
模块化 agentic RL；**异步 dispatcher**（比 naive await 快 1.55×）、tool loop、沙箱体系；
训出 SA-SWE-32B（SWE-bench 24.4%→39.4%，成本减半以上）。→ agentic 方向代表；
"长尾 rollout 的调度"是我们 straggler 问题的放大版。

### FlashREINFORCE（NVIDIA，2026-09；OpenRLHF 作者 Jian Hu 领衔）
**无 critic、单 rollout、异步**；One-Batch REINFORCE + Sequence Trust Region +
Sample-Mean Optimization；宣传 rollout 成本减半。攻击的正是 GRPO group sampling 的两个
系统代价：固定预算下压 prompt 覆盖 + 组同步屏障。→ 给我们的 group 设计（E6 全家）一个
反命题；C 报告"前沿"一节主角。

### 其他值得提名
ROLL（阿里，actor-learner 架构）；StaleFlow（全局 staleness 上限的异步系统）；
A-3PO（ICLR 2026，staleness 感知 PPO 近似）；DORA（美团，>3× vs 同步）；
LlamaRL（Meta，DDMA 直连）；ProRL Agent（MSR，rollout-as-a-service）；
Tunix（Google）、AgentGym-RL、AgentRL（多轮多任务）。
（"用 LLM agent 自动化 ML 实验"另有一支，不在 RL infra 主体内，略。）

## 3. 我们 rig 的位置（对照表）

| 维度 | 我们 | OpenRLHF hybrid | slime | AReaL |
|---|---|---|---|---|
| Rollout | Server（HTTP） | Engine（共居） | Server（RPC） | 全异步池 |
| 权重同步 | NCCL 全量/步 | NCCL + CUDA IPC + pause/resume 重叠 | 显式 update_weight | 版本化按需 |
| 显存 | 两卡各常驻 | sleep 时分复用 | 分离 | 分离 |
| 调度 | 自研队列 | Ray | Ray | 自研 + 负载均衡 |
| 多轮 agent | — | 有 | 有 | — |

**结论**：最朴素的一角，但有两个别人没有的东西——① **失效模式的一手记录**（框架论文
只写性能，没人写"怎么静默地坏"）；② 完全可控的 1B 小 rig（受控实验便宜）。RL-infra 报告把
这两点做足就有独立价值。

## 4. 对 RL-infra 报告的直接用途

1. §7 相关实践 = 本文压缩版（五轴 + 卡片 + 位置表）；
2. §5.3 按纠错 ①② 改写（CUDA IPC / sleep mode level 2）；
3. E7 的 staleness 有了产业坐标（AReaL 版本化 / FlashREINFORCE 单 rollout）；
4. 一致性佐证：生成占比 71%（我们）vs 80–90%（业界）——同一结构性事实；
5. cheap 实验清单见 [`../rl_infra_planning.md`](../rl_infra_planning.md) §5。

## 5. 来源

- [Anatomy of RL Frameworks](https://www.hanifleo.com/anatomy-of-rl-frameworks)（五轴骨架与各家同步机制）
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) · [Hybrid Engine 文档](https://openrlhf.readthedocs.io/en/latest/hybrid_engine.html) · [vLLM Blog: Accelerating RLHF with OpenRLHF](https://vllm.ai/blog/2025-04-23-openrlhf-vllm)
- [AReaL（arXiv 2505.24298，NeurIPS 2025）](https://arxiv.org/abs/2505.24298)
- [slime（LMSYS blog）](https://www.lmsys.org/blog/2025-07-09-slime) · [slime 文档](https://thudm.github.io/slime)
- [SkyRL-Agent（arXiv 2511.16108）](https://arxiv.org/abs/2511.16108)
- [FlashREINFORCE（GitHub / PDF）](https://github.com/yifanzhang-pro/FlashREINFORCE)
- [vLLM Sleep Mode 官方文档](https://docs.vllm.ai/en/latest/features/sleep_mode.html)
- [ProRL Agent（arXiv 2603.18815）](https://arxiv.org/abs/2603.18815) · [AgentRL（arXiv 2510.04206）](https://arxiv.org/abs/2510.04206) · [A-3PO（arXiv 2512.06547）](https://arxiv.org/abs/2512.06547)
- [Cameron Wolfe: Agentic RL — Frameworks and Best Practices](https://cameronrwolfe.substack.com/p/agentic-rl) · [Google Tunix blog](https://developers.googleblog.com/scaling-agentic-rl-high-throughput-agentic-training-with-tunix) · [ROLL（阿里）](https://github.com/alibaba/ROLL)
