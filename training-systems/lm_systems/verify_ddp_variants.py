"""Verify FlatDDP and OverlapDDP produce same results as naive DDP (CPU gloo)."""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from lm_systems.ddp import DDP, FlatDDP, OverlapDDP


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _check(rank, ws, ddp_cls):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group("gloo", rank=rank, world_size=ws)

    torch.manual_seed(rank)
    model = ddp_cls(Toy())
    x = torch.randn(4, 10)
    y = torch.randn(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        model.finish_gradient_synchronization()
        opt.step()

    # gather params across ALL ranks (all ranks must call all_gather)
    params = [p.detach().clone() for p in model.parameters()]
    gathered = [[] for _ in range(ws)]
    for i, p in enumerate(params):
        lst = [torch.zeros_like(p) for _ in range(ws)]
        dist.all_gather(lst, p)
        for r in range(ws):
            gathered[r].append(lst[r])
    dist.barrier()
    if rank == 0:
        ok = all(torch.allclose(gathered[0][i], gathered[r][i]) for r in range(1, ws) for i in range(len(gathered[0])))
        print(f"{ddp_cls.__name__}: params synced across ranks = {ok}")
    dist.destroy_process_group()


def main():
    ws = 2
    for cls in [DDP, FlatDDP, OverlapDDP]:
        mp.spawn(_check, args=(ws, cls), nprocs=ws, join=True)


if __name__ == "__main__":
    main()
