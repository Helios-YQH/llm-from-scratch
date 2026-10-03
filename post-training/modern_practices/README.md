# 工业界写法对照

同一个功能,成熟框架怎么做,用来对照我们自己 from-scratch 的实现。

**收录规则:只在两边有实质差别时写。** 已经是现代写法的就不写——写出来是噪音。
差别大的单独成文,差别小的在这里简短收录。

---

## 单独成文

| 文件 | 主题 | 核心差别 |
|---|---|---|
| [`tokenization_masking.md`](tokenization_masking.md) | response mask | 我们用布尔 mask,工业界用 `-100` 哨兵值 + `CrossEntropyLoss` 的 `ignore_index` |
| [`logging_and_tracking.md`](logging_and_tracking.md) | 训练日志记什么 | 骨架一致;我们缺**撞长度上限比例**和**生成/训练策略 log 概率差**两个诊断量 |
| [`gradient_accumulation.md`](gradient_accumulation.md) | 梯度累积与混合精度 | `accelerate` 的两处调用;以及我们**优化器状态是 bf16**这个没说出口的取舍 |
| [`rl_infra_systems.md`](rl_infra_systems.md) | RL 系统的设计空间（RL-infra 调查） | 五条轴 + 9 张系统卡片 + 我们的位置;**两处纠错**（同机 colocate 走 CUDA IPC 而非"不能用 NCCL";vLLM sleep mode 就是为 RLHF 权重重载设计的） |

## 跳过:已经是现代写法

| 组件 | 为什么跳过 |
|---|---|
| `checkpoint.py` 加载模型 | `from_pretrained` + bf16 + `attn_implementation`,就是标准做法 |
| `vllm_utils.py` 推理与权重同步 | vLLM + NCCL 传权重就是标准;starter 里的 `pkill` 那段见下 |
| `dpo.py` 的损失公式 | 标准 DPO 形式,和 TRL 一致(差别在工程封装,见下) |
| `evaluation.py` 的正则解析 | 任务简单,手写正则就是合理做法 |

---

## 简要收录

### DPO 的工程封装

我们的 `compute_per_instance_dpo_loss` 接收两个模型对象。工业界(TRL `DPOTrainer`)不需要两份权重:
**参考模型就是"禁用适配器的同一个模型"**——用 peft 的上下文管理器拿基座的前向:

```python
with model_ref.disable_adapter():      # 同一份显存,临时关掉 LoRA
    ref_logits = model_ref(input_ids)
```

这是 LoRA 做 DPO 的标准姿势,省掉一整个模型的显存。另外参考模型的 log 概率是**常量**,
可以在训练前对整个数据集算一次缓存起来,训练循环里就不用再跑参考前向了。

### SFT 数据打包

我们手工把文档拼成一条长流再切块。`transformers` 有现成的打包 collator
(`DataCollatorWithFlattening`,配合 `padding="max_length"` 的变体),TRL 也有
`packing=True` 的开关。手写的价值在于**能精确对齐 fixture**(我们正是靠这个反推出了
`.strip()` 和全局右移两条规则);生产代码直接用现成的更省事。

### 推理服务的启动与清理

starter 的 `kill_existing_vllm_server` 会 `pkill` 掉同端口的进程。**单机开发没问题,
共用集群上是灾难**:两个 run 都用默认端口时,后启动的会静默杀掉前一个的 vLLM,
而前一个要等到下一次请求失败才发现——几个小时已经过去了。

我们改成了**端口被占就报错,绝不 kill**。类似的判据在很多集群工具里都有:
"清理残留"和"抢占别人的资源"在行为上无法区分,所以成熟的工具默认选择不清理。

另外我们让 vLLM 返回**生成时的逐 token log 概率**(请求里加 `logprobs`),用来做
`sampling_logprob_difference` 检查——这是唯一能发现"推理引擎权重是旧的"的指标。

### 评测框架

`evaluation.py` 里的 MMLU/GSM8K 解析手写就够了,但如果要做**跨任务的标准评测**,
工业界用 `lm-evaluation-harness`(EleutherAI)或 `lighteval`(HuggingFace)。它们的价值不在
解析本身,而在于:统一的答案提取协议、few-shot 模板管理、跨任务结果聚合,以及
**避免自己写解析器时引入的系统性偏差**(比如 `\boxed{}` 提取规则不同会让两次实验不可比)。

### 实验调度

`run_experiments.py` 是一张手写的"变体 → 超参"映射表。规模上去之后的替代品:

- **Hydra** —— 配置组合(`+variant=dr_grpo seed=0`),支持配置继承和命令行覆盖,
  解决的是"变体多了以后 flag 组合爆炸"
- **wandb sweep / Optuna** —— 超参搜索,尤其是需要按验证指标**自适应**选下一组配置时
  (我们这种固定网格用不上,但调学习率时有用)

手写映射表的优势是**可读**——一眼能看出 RFT 就是 `baseline="none"` 加剪枝。
配置库的优势是**不会漏同步**:加了新参数不用记得往每个变体里填。
