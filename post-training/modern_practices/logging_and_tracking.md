# 训练日志:成熟框架记录什么

来源:verl `verl/trainer/ppo/metric_utils.py`、TRL `trl/trainer/grpo_trainer.py`、
OpenRLHF `openrlhf/trainer/ppo_trainer.py`,均读自 2026-09-29 的 `main` 分支。

## 我们缺的两个关键指标

### 1. `completions/clipped_ratio` —— 撞到生成长度上限的比例

TRL 记 `completions/clipped_ratio`,verl 记 `response_length/clip_ratio`。
两边都有,**我们没有**。

**为什么重要**:被截断的 response 不可能输出 `</answer>`,所以奖励必然是 0 ——
但这个 0 和"模型不会做"完全不是一回事。如果 30% 的 rollout 撞上限,训练奖励曲线会
平白低一截,而你会以为是模型能力问题,去调学习率。

verl 还额外记一套 `response_length_non_aborted/*`,注释写明是为了排除零长度样本
对均值的扭曲。同一个道理:**长度统计里混着"因为外部限制而非策略选择"的样本时,
均值会骗人**。

### 2. `sampling/sampling_logp_difference` —— 生成策略与训练策略的 log 概率差

TRL 在启用 vLLM 重要性修正时记:

```
sampling/sampling_logp_difference/{mean,max}
sampling/importance_sampling_ratio/{min,mean,max}
```

**这是"生成用的权重和训练用的权重不一致"的探测器**——正是我们刚修掉的那类 bug
(重构时漏掉 `sync_policy_weights`,rollout 全部来自旧权重)。

**为什么我们自己的指标抓不到**:我们记的 `importance_ratio_mean` 是**批内**的
(chunk 0 恒等于 1.0,因为 `old_log_probs` 用当前策略算)。TRL 这个量比的是
**生成时策略**与**训练时策略**——那才是真正该监控的偏移。漏同步时它会立刻偏离 1。

值得照抄:**每个 rollout 步,用生成那一刻的策略重新算一遍 log 概率,和训练策略对比**。
代价是一次额外前向,收益是这类静默 bug 再也藏不住。

## 我们已有的(对照确认没漏)

| 我们 | TRL / verl 对应 | 评价 |
|---|---|---|
| `n_sequences_kept` | TRL `frac_reward_zero_std`(整组奖励相同的比例) | 同一个诊断。TRL 的命名更直接(它就是"无梯度的组占比"),我们的是剪枝后的计数 |
| `clip_fraction` | TRL `clip_ratio/{low,high,region}_mean` | **我们只有一个均值**;TRL 拆成 low/high/region 三种,还记 `clip_ratio/low_min` / `high_max` 这两个**单序列最坏值**。均值分不清"普遍轻微裁剪"和"个别序列裁剪到爆" |
| `token_entropy` | `entropy` | ✅ |
| `advantage_mean/std`, `reward_max/min` | `critic/advantages/*`, `reward_std` | ✅ |
| `time_generate_s` / `time_grade_s` / `time_train_s` | `timing_s/{gen,ref,values,adv,update_actor}` | ✅ 思路一致 |
| `train_response_len_mean` | `response_length/{mean,max,min}` | 我们只有 mean,缺 max/min |

## 值得借鉴的其他做法

**per-token 计时**(verl `timing_per_token_ms/*`)。他们每个阶段都同时记绝对耗时
**和**每 token 毫秒,`gen` 用 response token 数做分母、其余用 prompt+response。
理由:生成时间随 token 数线性增长,绝对耗时无法区分"这一步慢"和"这一步长"。

**吞吐**(verl `perf/throughput` = tokens/sec/GPU)。我们没有。

**把 rollout 存成 parquet**(TRL 写 `output_dir/completions/completions_{step:05d}.parquet`,
每个记录批次都落盘,与 wandb 无关)。我们现在每 40 步写 txt。parquet 的好处是以后能
按奖励筛选、和别的表 join,而不是靠肉眼读文本。

**崩溃安全的淘汰顺序**:verl 在 `max_ckpt_to_keep=1` 时**先写新的成功后才删旧的**。
我们的顺序也是先存后删 ✅。OpenRLHF 还会在保存成功**之后**才写 `metric.json`,
避免崩溃留下孤儿文件。

**NaN 隔离**(TRL):`logging_steps > 1` 时,一次 NaN 批次会污染整个平均,所以他们在
求平均前过滤。我们每步都记,不受影响。

**训练阶段作为一等指标**(SkyRL 的 `TrainingPhaseGauge`):把当前阶段
(`wait_for_generation_buffer` / `run_training` / `sync_weights` …)发布成 Prometheus
gauge,这样能和 GPU 利用率 join 起来,直接回答"为什么 GPU 利用率只有 30%"。
我们靠计时分解近似达到这个目的。

**字段名排序后再跨 rank gather**(TRL):`sorted(set(keys))` 之后才 `gather_object`,
因为 dict 插入顺序在不同 rank 上可能不同,会导致数值错配。我们单卡用不上,但是个
值得知道的分布式陷阱。

## 结论

我们的骨架和成熟框架是一致的(run 目录、配置持久化、断点续跑、计时分解都对得上)。
**实质差距在"能诊断什么"上,具体是两条**:

1. 撞长度上限的比例 —— 防止把"被截断"误读成"不会做"
2. 生成/训练策略的 log 概率差 —— 防止静默的权重不同步

两条都是**便宜且高价值**的,建议补上。
