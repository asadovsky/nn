"""Trains a model.

Usage examples:

    $ PYTHONPATH=. python modeling/gpt2/train.py
    $ PYTHONPATH=. torchrun --standalone --nproc_per_node=8 modeling/gpt2/train.py
"""

import argparse
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

MINI_RUN = False

# Batch size based on GPT-3 Small.
TOTAL_BATCH_SIZE, MICRO_BATCH_SIZE, SEQ_LEN = (
    (2**9, 2, 32) if MINI_RUN or not torch.cuda.is_available() else (2**19, 64, 1024)
)

MAX_STEPS = 50 if MINI_RUN else 19073  # 10B tokens, batch size 2**19 tokens
VAL_STEPS = 200
CKPT_STEPS = 2000
assert CKPT_STEPS % VAL_STEPS == 0

MAX_LR = 6e-4
WARMUP_STEPS = 715  # 375M tokens, batch size 2**19 tokens


def get_lr(step: int):
    # Learning rate schedule based on GPT-3 Small.
    min_lr = MAX_LR * 0.1
    min_lr_steps = MAX_STEPS
    # Linear warmup.
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step > min_lr_steps:
        return min_lr
    # Cosine decay from `WARMUP_STEPS` to `min_lr_steps`.
    decay_ratio = (step - WARMUP_STEPS) / (min_lr_steps - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    assert 0 <= coeff <= 1
    return min_lr + coeff * (MAX_LR - min_lr)


def get_model(ckpt: dict | None) -> tuple[GPTConfig, GPT]:
    # Note, 50304 is slightly larger than the GPT-2 vocab size and is divisible by 128.
    cfg = (
        GPTConfig(max_seq_len=64, n_layer=2, n_head=2, n_embd=4)
        if MINI_RUN
        else GPTConfig(vocab_size=50304)
    )
    model = GPT(cfg)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_sd"])
    return cfg, model


def get_optimizer(ckpt: dict | None, model: GPT, fused: bool) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    params_decay = [p for p in params if p.dim() >= 2]
    params_no_decay = [p for p in params if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": params_decay, "weight_decay": 0.1},
            {"params": params_no_decay, "weight_decay": 0.0},
        ],
        lr=MAX_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=fused,
    )
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer_sd"])
    return optimizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", type=str)
    args = parser.parse_args()

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

    ckpt = torch.load(args.ckpt) if args.ckpt else None
    cfg, model = get_model(ckpt)
    optimizer = get_optimizer(ckpt, model, device_type == "cuda")

    model.to(device)
    if device != "mps":
        model = torch.compile(model)
    if use_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    assert isinstance(model, nn.Module)

    assert TOTAL_BATCH_SIZE % (MICRO_BATCH_SIZE * SEQ_LEN * ddp_world_size) == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // (MICRO_BATCH_SIZE * SEQ_LEN * ddp_world_size)

    # TODO: Load `train_dl` state from `ckpt`.
    train_dl = DataLoader(MICRO_BATCH_SIZE, SEQ_LEN, ddp_rank, ddp_world_size, "train")
    val_dl = DataLoader(MICRO_BATCH_SIZE, SEQ_LEN, ddp_rank, ddp_world_size, "val")

    run_dir, log_file = "", ""
    if is_master_process:
        run_dir = os.path.join(
            os.getcwd(), ".runs", datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        )
        os.makedirs(run_dir)
        log_file = os.path.join(run_dir, "log.txt")
        open(log_file, "w").close()  # touch

    # Training loop.
    for step in range(0 if ckpt is None else ckpt["step"] + 1, MAX_STEPS):
        is_last_step = step == MAX_STEPS - 1

        # Perform one optimization step.
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        train_loss_accum = torch.zeros(1).to(device)  # average loss over full batch
        for micro_step in range(grad_accum_steps):
            x, y = train_dl.next_batch()
            x, y = x.to(device), y.to(device)
            if use_ddp:
                assert isinstance(model, DDP)
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            with torch.autocast(device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss /= grad_accum_steps
            train_loss_accum += loss.detach()
            loss.backward()
        if use_ddp:
            dist.all_reduce(train_loss_accum, op=dist.ReduceOp.AVG)

        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
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
                f"dt={(dt * 1000):.2f}ms tok/sec={(TOTAL_BATCH_SIZE / dt):.2f}"
            )
            with open(log_file, "a") as f:
                f.write(f"{s}\n")

        # Occasionally measure validation loss and save checkpoint.
        if step % VAL_STEPS == 0 or is_last_step:
            model.eval()
            val_dl.reset()
            with torch.no_grad():
                val_loss_accum_steps = 20
                val_loss_accum = torch.zeros(1).to(device)
                for _ in range(val_loss_accum_steps):
                    x, y = val_dl.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    loss /= val_loss_accum_steps
                    val_loss_accum += loss.detach()
            if use_ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            if is_master_process:
                s = f"{step=} val_loss={val_loss_accum.item():.6f}"
                print(s)
                with open(log_file, "a") as f:
                    f.write(f"{s}\n")
                if step > 0 and (step % CKPT_STEPS == 0 or is_last_step):
                    torch.save(
                        {
                            "model_cfg": cfg,
                            "model_sd": model.state_dict(),
                            "optimizer_sd": optimizer.state_dict(),
                            "step": step,
                            "val_loss": val_loss_accum.item(),
                        },
                        os.path.join(run_dir, f"model_{step:06d}.pt"),
                    )

    if use_ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
