# LLM From Scratch

从零实现并系统评测整条语言模型 stack:tokenizer、Transformer 预训练、GPU 系统优化、
预训练数据、后训练(post-training),每个部分配一份英文技术报告。所有核心组件均由
基础原语手写(被测实现中不含 `nn.Transformer`、`flash_attn`、DeepSpeed、TRL),
报告中的每个数字都由脚本从原始 run 记录重新生成,而非手工誊抄。

**English: [README.md](README.md)**

## 项目一览

| 项目 | 内容 | 主要结果 | 报告 |
| --- | --- | --- | --- |
| [transformer-lm](transformer-lm/) | Byte-level BPE tokenizer、decoder-only Transformer(RMSNorm、RoPE、SwiGLU、causal attention)、AdamW + warmup/cosine 调度、训练循环、采样解码 | BPE 分词器与参考实现逐 bit 一致(TinyStories 上 4.18 bytes/token);22.7M 参数模型 1.5 GPU-hour 达到 1.57 验证损失;学习率扫描与四组消融,包括"1000× 过大的学习率最终 loss 仍有限 —— 余弦尾部把发散掩盖了"这一发现 | [PDF](transformer-lm/report/tech_report.pdf) |
| [training-systems](training-systems/) | Profiling 与 kernel 归因(Nsight Systems)、混合精度、梯度 checkpointing、Triton FlashAttention-2(forward + backward)、三种 DDP 变体与通信重叠、ZeRO-1、FSDP、并行策略解析上限 | 64K 上下文下 attention 加速 5.7×(bf16),也是唯一能在 FP32 64K 跑通的实现;重叠把 4 卡梯度通信占比从 69% 降到 59%;自写 FSDP 稳态显存 −43%,与 PyTorch 官方差距 14% 以内 | [PDF](training-systems/report/tech_report.pdf) |
| [data-pipeline](data-pipeline/) | Common Crawl WET → 预训练语料:正文抽取、语言识别、PII 掩码、有害内容分类器、Gopher 质量规则、fastText 质量分类器、精确 + MinHash/LSH 去重、样本级审计 | 214 万文档 → 保留 9.3% → 3.4 亿 GPT-2 tokens;配对 4000 步消融:C4-100 验证损失降低 0.27 nats(困惑度 282 vs 367);逐条审计各过滤器的失效模式,并给出"何时数据消融测的其实是记忆"的规模研究 | [PDF](data-pipeline/report/tech_report.pdf) |
| [post-training](post-training/) | 从零实现 GRPO 做推理 RL;SFT + DPO 偏好优化;验证器质量研究;与工业界 RL 框架的工程对照笔记 | GRPO 报告:1B 规模下 prompt 才是一阶变量、估计器变体全在种子噪声内;off-policy 下 clipping 是关键;验证器 10% 翻转率下 RL 保留约 90–105% 收益而 test-time selection 只剩 57%(附 affine-invariance 机制解释)。SFT/DPO 报告:指令微调的 MMLU 增益大部分是答案格式;同一个模型 AlpacaEval 胜率随裁判从 3.8% 波动到 33.2% | [PDF](post-training/report/tech_report.pdf)、[PDF](post-training/report_track3/tech_report.pdf) |

## 报告

五份报告同时以 PDF 形式挂在 [v1.0 release](https://github.com/Helios-YQH/llm-from-scratch/releases/tag/v1.0)。

1. **A Transformer Language Model From Scratch: Implementation and Controlled Experiments** — [PDF](transformer-lm/report/tech_report.pdf)
2. **Where the Time and Memory Go: Systems Optimizations for Transformer Training** — [PDF](training-systems/report/tech_report.pdf)
3. **From Crawl to Corpus: Building, Auditing, and Evaluating a Common Crawl Filtering Pipeline** — [PDF](data-pipeline/report/tech_report.pdf)
4. **Estimator Choices and Reward Quality in GRPO: A Controlled Study on the GSM8K Benchmark** — [PDF](post-training/report/tech_report.pdf)
5. **Instruction Tuning and Preference Optimization: Capability Trade-offs and Judge Dependence** — [PDF](post-training/report_track3/tech_report.pdf)

每份报告的图和正文引用的数字都由脚本从原始 run 记录生成(`report/make_figures.py`
及配套脚本),表格不会与其依据的数据脱节。

## 运行

每个项目是独立的 `uv` 工程,`uv.lock` 锁定依赖,无 workspace:

```bash
cd transformer-lm   # 或 training-systems / data-pipeline / post-training
uv sync
uv run pytest
```

说明:

- CPU 测试子集到处能跑;GPU 路径(Triton kernel、分布式训练、vLLM rollout)需要
  CUDA 机器,各项目 README 里写明了哪些是哪些。
- 大规模训练在同组共享的 6×A6000 节点上完成,复现无需同等规模。

## 定位与出处

这些项目是"实现 + 复现"研究:所复现的方法均为已发表工作,报告中有引用;每份报告的
scope acknowledgement 明确写清了哪些部分遵循已发表设计、哪些是本人的;所有数字均由
脚本从仓库内随附的 run 记录生成,没有手工誊抄。

## License

- 本仓库原创代码为 MIT 协议 —— 见 [LICENSE](LICENSE)。
- `training-systems/lm-basics/` 是为系统基准固定版本的 vendored 基线模型;
  各项目目录下的 `LICENSE`(MIT)覆盖其自带的第三方脚手架与参考测试。

## 作者

Yi Hou — 中国科学院大学,北京 — houyi25@mails.ucas.ac.cn
