import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel


@dataclass(slots=True)
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    n_layer: int = 12  # num layers
    n_head: int = 12  # num attention heads
    n_embd: int = 768  # embedding dimensionality


def _init_weight(module: nn.Linear | nn.Embedding, n_layer: int | None = None) -> None:
    std = 0.02
    if n_layer is not None:
        std *= (2 * n_layer) ** -0.5
    torch.nn.init.normal_(module.weight, std=std)
    if isinstance(module, nn.Linear) and module.bias is not None:  # pyright: ignore
        torch.nn.init.zeros_(module.bias)


def _causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    if True:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    T = q.shape[-2]
    attn = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn = attn.masked_fill(torch.tril(torch.ones((T, T))) == 0, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return attn @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self._cfg: GPTConfig = cfg
        assert cfg.n_embd % cfg.n_head == 0
        # QKV projections for all attention heads.
        self.c_attn: nn.Linear = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        # Output projection.
        self.c_proj: nn.Linear = nn.Linear(cfg.n_embd, cfg.n_embd)
        _init_weight(self.c_attn)
        _init_weight(self.c_proj, cfg.n_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch size, seq len, embedding dimensionality (n_embd).
        B, T, C = x.shape
        assert C == self._cfg.n_embd
        # Calculate QKVs for all heads.
        nh = self._cfg.n_head
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self._cfg.n_embd, dim=2)
        # nh is "num heads", hs is "head size", C is "num channels" = n_embd = nh * hs.
        q = q.view(B, T, nh, C // nh).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, C // nh).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, nh, C // nh).transpose(1, 2)  # (B, nh, T, hs)
        x = _causal_attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc: nn.Linear = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu: nn.GELU = nn.GELU(approximate="tanh")
        self.c_proj: nn.Linear = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        _init_weight(self.c_fc)
        _init_weight(self.c_proj, cfg.n_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1: nn.LayerNorm = nn.LayerNorm(cfg.n_embd)
        self.attn: CausalSelfAttention = CausalSelfAttention(cfg)
        self.ln_2: nn.LayerNorm = nn.LayerNorm(cfg.n_embd)
        self.mlp: MLP = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self._cfg: GPTConfig = cfg
        self.transformer: nn.ModuleDict = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.max_seq_len, cfg.n_embd),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd),
            )
        )
        self.lm_head: nn.Linear = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        _init_weight(self.transformer.wte)
        _init_weight(self.transformer.wpe)

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        T = inputs.shape[-1]
        assert T <= self._cfg.max_seq_len
        # Token and position embeddings.
        tok_emb = self.transformer.wte(inputs)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=inputs.device)  # (T,)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = tok_emb + pos_emb
        # Transformer blocks.
        for block in self.transformer.h:
            x = block(x)
        # Final layers.
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # Calculate the average loss over the entire batch.
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_name: str) -> "GPT":
        """Loads model weights from GPT2LMHeadModel."""
        # https://openai.com/index/gpt-2-1-5b-release/
        cfg_dict = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 355M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1.5B params
        }[model_name]
        cfg_dict["vocab_size"] = 50257
        cfg_dict["max_seq_len"] = 1024
        model = GPT(GPTConfig(**cfg_dict))
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained(model_name)
        assert isinstance(model_hf, nn.Module)
        sd_hf = model_hf.state_dict()
        assert len(sd.keys()) == len(sd_hf.keys())

        # Convert transformers.pytorch_utils.Conv1D to nn.Linear by transposing.
        transpose = [
            ".attn.c_attn.weight",
            ".attn.c_proj.weight",
            ".mlp.c_fc.weight",
            ".mlp.c_proj.weight",
        ]
        for k in sd_hf.keys():
            if any(k.endswith(x) for x in transpose):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def load_state_dict_supporting_compile_and_ddp(self, sd: Mapping[str, Any]):
        if self.state_dict().keys() == sd.keys():
            return super().load_state_dict(sd)
        super().load_state_dict(
            {
                # DDP adds "module.", torch.compile adds "_orig_mod.". Here we assume
                # that torch.compile was done first.
                k.removeprefix("module.").removeprefix("_orig_mod."): v
                for k, v in sd.items()
            }
        )
