"""Trains a model.

Usage examples:

    $ PYTHONPATH=. python modeling/gpt2/train.py
    $ PYTHONPATH=. torchrun --nproc_per_node=8 modeling/gpt2/train.py
"""

import datetime
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from modeling import device_util
from modeling.gpt2.data_loader import DataLoader
from modeling.gpt2.model import GPT, GPTConfig

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

torch.set_float32_matmul_precision("high")

MICRO = False

device, device_type = device_util.get_device()

use_ddp = os.getenv("RANK") is not None
if use_ddp:
    assert device in {"cpu", "cuda"}
    init_process_group(backend=("gloo" if device == "cpu" else "nccl"))
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    is_master_process = ddp_rank == 0
    device = f"{device}:{ddp_local_rank}"
    if device_type == "cuda":
        torch.cuda.set_device(device)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    is_master_process = True

# Create model.
# Note, 50304 is slightly larger than the GPT-2 vocab size and is divisible by 128.
cfg = (
    GPTConfig(max_seq_len=64, n_layer=2, n_head=2, n_embd=4)
    if MICRO
    else GPTConfig(vocab_size=50304)
)
model = GPT(cfg)
model.to(device)
if device != "mps":
    model = torch.compile(model)
if use_ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
assert isinstance(model, nn.Module)

# Batch size and sequence length based on GPT-3 Small.
total_batch_size, micro_batch_size, seq_len = (
    (2**9, 2, 32) if MICRO or device_type != "cuda" else (2**19, 64, 1024)
)
assert total_batch_size % (micro_batch_size * seq_len * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (micro_batch_size * seq_len * ddp_world_size)

max_steps = 50 if MICRO else 19073  # 10B tokens, batch size 2**19 tokens
val_steps = 200
ckpt_steps = 2000
assert ckpt_steps % val_steps == 0

# Learning rate schedule based on GPT-3 Small.
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715  # 375M tokens, batch size 2**19 tokens
min_lr_steps = max_steps


def _get_lr(step: int):
    # Linear warmup.
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > min_lr_steps:
        return min_lr
    # Cosine decay between `warmup_steps` and `min_lr_steps`.
    decay_ratio = (step - warmup_steps) / (min_lr_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    assert 0 <= coeff <= 1
    return min_lr + coeff * (max_lr - min_lr)


# Create optimizer.
params = [p for p in model.parameters() if p.requires_grad]
params_decay = [p for p in params if p.dim() >= 2]
params_no_decay = [p for p in params if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": params_decay, "weight_decay": 0.1},
        {"params": params_no_decay, "weight_decay": 0.0},
    ],
    lr=max_lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    fused=(device_type == "cuda"),
)

train_dl = DataLoader(micro_batch_size, seq_len, ddp_rank, ddp_world_size, "train")
val_dl = DataLoader(micro_batch_size, seq_len, ddp_rank, ddp_world_size, "val")

run_dir = os.path.join(
    os.getcwd(), ".runs", datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
)
os.makedirs(run_dir)
log_file = os.path.join(run_dir, "log.txt")
open(log_file, "w").close()  # touch

# Training loop.
for step in range(max_steps):
    is_last_step = step == max_steps - 1

    # Perform one optimization step.
    t0 = time.time()
    model.train()
    optimizer.zero_grad()

    train_loss_accum = torch.zeros(1).to(device)  # average loss over the full batch
    for micro_step in range(grad_accum_steps):
        x, y = train_dl.next_batch()
        x, y = x.to(device), y.to(device)
        if use_ddp:
            assert isinstance(model, DDP)
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        with torch.autocast(device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss /= grad_accum_steps
        train_loss_accum += loss.detach()
        loss.backward()
    if use_ddp:
        dist.all_reduce(train_loss_accum, op=dist.ReduceOp.AVG)

    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = _get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0
    if is_master_process:
        s = f"{step=} train_loss={train_loss_accum.item():.6f}"
        print(
            f"{s} {lr=:.4e} {norm=:.4f} "
            f"dt={(dt * 1000):.2f}ms tok/sec={(total_batch_size / dt):.2f}"
        )
        with open(log_file, "a") as f:
            f.write(f"{s}\n")

    # Occasionally measure validation loss and save checkpoint.
    if step % val_steps == 0 or is_last_step:
        model.eval()
        val_dl.reset()
        with torch.no_grad():
            val_loss_accum_steps = 20
            val_loss_accum = torch.zeros(1).to(device)
            for _ in range(val_loss_accum_steps):
                x, y = val_dl.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss /= val_loss_accum_steps
                val_loss_accum += loss.detach()
        if use_ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if is_master_process:
            s = f"{step=} val_loss={val_loss_accum.item():.6f}"
            print(s)
            with open(log_file, "a") as f:
                f.write(f"{s}\n")
            if step > 0 and (step % ckpt_steps == 0 or is_last_step):
                torch.save(
                    {
                        "model_cfg": cfg,
                        "model_sd": model.state_dict(),
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                    },
                    os.path.join(run_dir, f"model_{step:06d}.pt"),
                )

if use_ddp:
    destroy_process_group()
