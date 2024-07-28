"""Trains a model.

Usage examples:

    $ PYTHONPATH=. python modeling/gpt2/train_torch.py
    $ PYTHONPATH=. torchrun --standalone --nproc_per_node=8 modeling/gpt2/train_torch.py
"""

import argparse
import dataclasses
import datetime
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from modeling import device_util
from modeling.gpt2.data_loader import GPTDataLoader
from modeling.gpt2.model_torch import GPT, GPTConfig

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

torch.set_float32_matmul_precision("high")


@dataclass(slots=True)
class Config:
    ckpt: str = ""
    # If set, various params below will be overridden.
    test_run: bool = False
    run_dir: str = ""
    # Batch size based on GPT-3 Small.
    total_batch_toks: int = 2**19
    micro_batch_size: int = 64  # max size for NVIDIA A100 80GB
    seq_len: int = 1024
    max_steps: int = 19073  # 10B tokens, batch size 2**19 tokens
    val_steps: int = 200
    ckpt_steps: int = 2000
    max_lr: float = 6e-4
    warmup_steps: int = 715  # 375M tokens, batch size 2**19 tokens
    device: str = ""


def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def get_lr(cfg: Config, step: int):
    # Learning rate schedule based on GPT-3 Small.
    min_lr = cfg.max_lr * 0.1
    min_lr_steps = cfg.max_steps
    # Linear warmup.
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    if step > min_lr_steps:
        return min_lr
    # Cosine decay from `warmup_steps` to `min_lr_steps`.
    decay_ratio = (step - cfg.warmup_steps) / (min_lr_steps - cfg.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    assert 0 <= coeff <= 1
    return min_lr + coeff * (cfg.max_lr - min_lr)


def get_model(
    cfg: Config, state: dict | None, device: str, use_ddp: bool, ddp_local_rank: int
) -> tuple[nn.Module, GPTConfig]:
    # Note, 50304 is slightly larger than the GPT-2 vocab size and is divisible by 128.
    model_cfg = (
        GPTConfig(max_seq_len=64, n_layer=2, n_head=2, n_embd=4)
        if cfg.test_run
        else GPTConfig(vocab_size=50304)
    )
    model = GPT(model_cfg)
    if state is not None:
        model.load_state_dict_supporting_compile_and_ddp(state)
    model.to(device)
    if device != "mps":
        model = torch.compile(model)
    if use_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    assert isinstance(model, nn.Module)
    return model, model_cfg


def get_optimizer(
    cfg: Config, state: dict | None, model: nn.Module, fused: bool
) -> torch.optim.Optimizer:  # pyright: ignore [reportPrivateImportUsage]
    params = [p for p in model.parameters() if p.requires_grad]
    params_decay = [p for p in params if p.dim() >= 2]
    params_no_decay = [p for p in params if p.dim() < 2]
    optimizer = torch.optim.AdamW(  # pyright: ignore [reportPrivateImportUsage]
        [
            {"params": params_decay, "weight_decay": 0.1},
            {"params": params_no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=fused,
    )
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def get_data_loader(
    cfg: Config, state: dict | None, ddp_rank: int, ddp_world_size: int, split: str
) -> GPTDataLoader:
    dl = GPTDataLoader(
        cfg.micro_batch_size, cfg.seq_len, ddp_rank, ddp_world_size, split
    )
    if state is not None:
        dl.load_state_dict(state)
    return dl


def run(cfg: Config) -> None:
    if cfg.test_run or not torch.cuda.is_available():
        cfg.total_batch_toks = 2**7
        cfg.micro_batch_size = 2
        cfg.seq_len = 8
    if cfg.test_run:
        cfg.max_steps = 10
        cfg.val_steps = 2
        cfg.ckpt_steps = 4
    assert cfg.ckpt_steps % cfg.val_steps == 0

    device, device_type = device_util.get_device(cfg.device)
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

    micro_batch_toks = cfg.micro_batch_size * cfg.seq_len * ddp_world_size
    assert cfg.total_batch_toks % micro_batch_toks == 0
    grad_accum_steps = cfg.total_batch_toks // micro_batch_toks

    ckpt = torch.load(cfg.ckpt, weights_only=True) if cfg.ckpt else {}
    model, model_cfg = get_model(
        cfg, ckpt.get("model_sd"), device, use_ddp, ddp_local_rank
    )
    optimizer = get_optimizer(
        cfg, ckpt.get("optimizer_sd"), model, device_type == "cuda"
    )
    train_dl = get_data_loader(
        cfg, ckpt.get("train_dl_sd"), ddp_rank, ddp_world_size, "train"
    )

    run_dir, log_file = "", ""
    if is_master_process:
        if cfg.run_dir:
            assert not any(os.scandir(cfg.run_dir))
            run_dir = cfg.run_dir
        else:
            run_dir = os.path.join(
                os.getcwd(), ".runs", datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            )
            os.makedirs(run_dir)
        log_file = os.path.join(run_dir, "log.txt")
        open(log_file, "w").close()  # touch

    # Training loop.
    for step in range(ckpt["step"] + 1 if ckpt else 0, cfg.max_steps):
        is_last_step = step == cfg.max_steps - 1

        # Perform one optimization step.
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        train_loss_accum = torch.zeros(1).to(device)  # average loss over full batch
        for micro_step in range(grad_accum_steps):
            x, y = next(train_dl)
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
        lr = get_lr(cfg, step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        if is_master_process:
            s = f"{now_str()} {step=} train_loss={train_loss_accum.item():.6f}"
            print(
                f"{s} {lr=:.4e} {norm=:.4f} "
                f"dt={(dt * 1000):.2f}ms tok/sec={(cfg.total_batch_toks / dt):.2f}"
            )
            with open(log_file, "a") as f:
                f.write(f"{s}\n")

        # Occasionally measure validation loss and save checkpoint.
        if step % cfg.val_steps == 0 or is_last_step:
            model.eval()
            val_dl = get_data_loader(cfg, None, ddp_rank, ddp_world_size, "val")
            with torch.no_grad():
                val_loss_accum_steps = 20
                val_loss_accum = torch.zeros(1).to(device)
                for _ in range(val_loss_accum_steps):
                    x, y = next(val_dl)
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    loss /= val_loss_accum_steps
                    val_loss_accum += loss.detach()
            if use_ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            if is_master_process:
                s = f"{now_str()} {step=} val_loss={val_loss_accum.item():.6f}"
                print(s)
                with open(log_file, "a") as f:
                    f.write(f"{s}\n")
                if step > 0 and (step % cfg.ckpt_steps == 0 or is_last_step):
                    torch.save(
                        {
                            "model_cfg": dataclasses.asdict(model_cfg),
                            "model_sd": model.state_dict(),
                            "optimizer_sd": optimizer.state_dict(),
                            "train_dl_sd": train_dl.state_dict(),
                            "step": step,
                            "val_loss": val_loss_accum.item(),
                        },
                        os.path.join(run_dir, f"model_{step:06d}.pt"),
                    )

    if use_ddp:
        destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", type=str)
    parser.add_argument("-t", "--test-run", action="store_true")
    args = parser.parse_args()

    cfg = Config(ckpt=args.ckpt, test_run=args.test_run)
    run(cfg)


if __name__ == "__main__":
    main()
