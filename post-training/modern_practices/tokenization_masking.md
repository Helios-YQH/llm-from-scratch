# Response masking:布尔 mask vs `-100` 哨兵值

## 我们的实现

`lm_alignment/tokenization.py` 手工建一个布尔张量,和 `labels` 逐位置对齐:

```python
input_ids     = padded[:-1]
labels        = padded[1:]
response_mask = zeros(..., dtype=torch.bool)
response_mask[i, len(prompt) - 1 : len(ids) - 1] = True
```

loss 端再拿这个 mask 去乘:

```python
masked_loss = per_token_loss * response_mask
```

## 工业界做法

不建 mask,而是把**不该算 loss 的位置的 label 改成 `-100`**,然后直接交给
`CrossEntropyLoss` —— 它的 `ignore_index` 默认就是 `-100`:

```python
labels = full_ids[1:].clone()
labels[: len(prompt) - 1] = -100          # prompt 与 padding 位置

logits = model(input_ids).logits
# reduction="none" 拿到逐 token loss;它内部就是 -log_softmax 后 gather
per_token_loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
).view(labels.shape)
```

这就是 HF 全家上下(HF `Trainer`、TRL、`accelerate` 例子)训练语言模型的标准写法。

## 差别不只是风格

**`-100` 版本更省显存**。`cross_entropy` 有融合 kernel,不需要把整个 `(B, L, V)` 的
log_softmax 落下来——显存瓶颈在 logits 本身而不是概率张量。我们为了拿 entropy 才用了
`log_softmax + gather`(见 `grpo.py`),代价是 microbatch=8 / 512 token / 50k 词表时
要多占约 824MB 的 fp32 中间量。

**但 mask 版本能复用**。`response_mask` 除了算 loss,还要用来做 sequence normalization
的按长度平均、GSPO 的"只对 response token 求 log 比值平均"。`-100` 只能表达"忽略",
要恢复位置信息得再 `labels != -100` 反算一次。GRPO 里两者都需要,所以我们的选择在这个
场景下是合理的。

**真实项目里两个都有人用**。纯 SFT 用 `-100` 更省事;RL 里因为 mask 要复用,反而常见
显式 mask。所以这条**不算我们写错了**,但值得知道 `-100` 的存在——读别人的训练代码时
看到满屏 `-100` 不用困惑。

## 顺带一个容易错的点

`labels` 是 `full_ids[1:]` 而不是 `full_ids`,所以**mask 的起点是 `len(prompt) - 1`**:

```
full:    [p0 p1 p2 p3 | r0 r1 r2]
labels:  [p1 p2 p3 r0 | r1 r2]        ← 右移一位
mask:    [ 0  0  0  1 |  1  1]        ← r0 落在下标 len(prompt)-1
```

`-100` 写法的对应操作是 `labels[:len(prompt)-1] = -100`(注意也是 `-1`,不是 `len(prompt)`)。
写成 `labels[:len(prompt)]` 会**多吃掉一个 response token**,而且不报错——这是本节最值得
记住的坑。
