# 梯度累积与混合精度:`accelerate` 帮着做了什么

## 我们的实现

`grpo.py` 的 `grpo_train_step` 手写累积:

```python
for start in range(0, len(keep), microbatch_size):
    ...
    if loss_normalization == "sequence":
        microbatch_loss = microbatch_loss * (len(indices) / n_sequences)
    microbatch_loss.backward()

grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
optimizer.step()
optimizer.zero_grad()
```

## 工业界做法

`accelerate` 把这几件事收进两个调用:

```python
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=32, mixed_precision="bf16")
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    with accelerator.accumulate(model):        # 只在最后一个 microbatch 同步/step
        loss = compute_loss(batch)
        accelerator.backward(loss)              # 内部已经除以 accumulation_steps
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
```

**注意 `accelerator.backward(loss)` 会自己除以 `gradient_accumulation_steps`**。
我们手写的 `* (len(indices) / n_sequences)` 是同一个东西的另一种表达——区别在于
**我们的缩放因子依赖归一化方式**(sequence 归一要按序列数加权,constant 归一不用),
而 `accelerate` 假设的是"每个 microbatch 等权",对 constant 归一那种"按全局常数除"
的语义并不直接适用。所以在 GRPO 这种自定义归一里,手写反而更清楚。

## 真正值得注意的差别:优化器状态精度

这是我们**没有处理**的问题。

我们加载模型时就转成了 bf16:

```python
AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
```

`torch.optim.AdamW` 建状态时用 `torch.zeros_like(param)`,**所以 Adam 的两个动量也是 bf16**。
后果:

- **更新精度只有约 8 位尾数**。`m / (sqrt(v) + eps)` 里如果动量本身很小,相对误差会很明显。
- **checkpoint 因此小一些**:2 个动量 × 1.48B 参数 × 2 字节 ≈ 5.9GB。若用 fp32 状态会到 11.8GB。

`accelerate` 的 `mixed_precision="bf16"` 走的是另一条路:**参数保持 fp32,前向用 autocast**。
优化器因此拿到 fp32 状态,更新精度完整;代价是显存翻倍。

两条路在真实项目里都有人用(大模型训练为了省显存常用 bf16 优化器状态),所以这不是
"我们写错了",而是**一个我们没有显式做过的取舍**。值得知道的是:

- 如果 RL 训练出现"loss 不降也不炸、就是不动",**优化器状态精度**是一个该查的方向,
  而它不会在任何日志里体现出来。
- 想换的话,最小改动是加载后 `model.to(torch.float32)` 再让 optimizer 建状态,前向用
  `torch.amp.autocast('cuda', dtype=torch.bfloat16)`,backward 放在 autocast 外面。

## 另外两件 `accelerate` 顺手做的事

**梯度裁剪的时机**。它保证 `clip_grad_norm_` 在所有 microbatch 都 backward 完之后、
`step` 之前调用——顺序错了裁剪就没有意义。我们的顺序是对的(`backward` 循环全部结束后
才 clip),但这是手写版本里容易写反的地方。

**多卡下的 `no_sync`**。`accelerator.accumulate()` 在非最后的 microbatch 上会跳过梯度
all-reduce,省掉大量通信。我们单卡用不上,但**如果以后要多卡,手写累积会默认每次
backward 都通信一遍**,那会是数倍的浪费。
