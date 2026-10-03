"""Verify 2.1.5(b): dtypes of each tensor under torch.autocast with FP16.

Runs ToyModel forward + backward inside autocast and prints the actual dtype
of parameters, intermediate activations, logits, loss, and gradients.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def main():
    torch.manual_seed(0)
    model = ToyModel(10, 10).cuda()
    # register hooks to print each module's output dtype
    for name, m in model.named_modules():
        m.register_forward_hook(lambda mod, inp, out, n=name: print(f"  {n}.output: {out.dtype}"))

    x = torch.randn(4, 10, device="cuda")
    y_true = torch.randint(0, 10, (4,), device="cuda")

    print("=== forward under autocast(fp16) ===")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x)
        loss = F.cross_entropy(y, y_true)

    print(f"  logits (fc2 output): {y.dtype}")
    print(f"  loss: {loss.dtype}")
    print("=== parameters (never cast by autocast) ===")
    print(f"  fc1.weight: {model.fc1.weight.dtype}")
    print(f"  ln.weight:  {model.ln.weight.dtype}")
    print(f"  fc2.weight: {model.fc2.weight.dtype}")

    print("=== backward: gradient dtype ===")
    loss.backward()
    print(f"  fc1.weight.grad: {model.fc1.weight.grad.dtype}")
    print(f"  ln.weight.grad:  {model.ln.weight.grad.dtype}")
    print(f"  fc2.weight.grad: {model.fc2.weight.grad.dtype}")

    # Same under BF16 for comparison
    print("\n=== forward under autocast(bf16) ===")
    model2 = ToyModel(10, 10).cuda()
    x2 = torch.randn(4, 10, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y2 = model2(x2)
        loss2 = F.cross_entropy(y2, y_true)
    print(f"  logits (bf16): {y2.dtype}")
    print(f"  loss (bf16): {loss2.dtype}")


if __name__ == "__main__":
    main()
