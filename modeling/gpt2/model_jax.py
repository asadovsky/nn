import re
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import torch.nn as tnn
from flax.linen.initializers import normal, zeros
from transformers import GPT2LMHeadModel


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    n_layer: int = 12  # num layers
    n_head: int = 12  # num attention heads
    n_embd: int = 768  # embedding dimensionality


def _weight_init(n_layer: int | None = None):
    std = 0.02
    if n_layer is not None:
        std *= (2 * n_layer) ** -0.5
    return normal(std)


# TODO: Check correctness.
# TODO: Use flash attention if possible.
def _causal_attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    T = q.shape[-2]
    att = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * 1.0 / jnp.sqrt(q.shape[-1])
    att = jnp.where(jnp.tril(jnp.ones((T, T))) == 0, float("-inf"), att)
    att = jax.nn.softmax(att, axis=-1)
    return jnp.matmul(att, v)


class CausalSelfAttention(nn.Module):
    _cfg: GPTConfig

    def setup(self):
        # QKV projections for all attention heads.
        self.c_attn: nn.Dense = nn.Dense(
            3 * self._cfg.n_embd, kernel_init=_weight_init(), bias_init=zeros
        )
        # Output projection.
        self.c_proj: nn.Dense = nn.Dense(
            self._cfg.n_embd,
            kernel_init=_weight_init(self._cfg.n_layer),
            bias_init=zeros,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Batch size, seq len, embedding dimensionality (n_embd).
        B, T, C = x.shape
        assert C == self._cfg.n_embd
        # Calculate QKVs for all heads.
        nh = self._cfg.n_head
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # nh is "num heads", hs is "head size", C is "num channels" = n_embd = nh * hs.
        q = jnp.reshape(q, (B, T, nh, C // nh)).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        k = jnp.reshape(k, (B, T, nh, C // nh)).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = jnp.reshape(v, (B, T, nh, C // nh)).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        x = _causal_attention(q, k, v)
        x = x.transpose(0, 2, 1, 3).reshape(B, T, C)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    _cfg: GPTConfig

    def setup(self):
        self.c_fc: nn.Dense = nn.Dense(
            4 * self._cfg.n_embd, kernel_init=_weight_init(), bias_init=zeros
        )
        self.c_proj: nn.Dense = nn.Dense(
            self._cfg.n_embd,
            kernel_init=_weight_init(self._cfg.n_layer),
            bias_init=zeros,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    _cfg: GPTConfig

    def setup(self):
        self.ln_1: nn.LayerNorm = nn.LayerNorm()
        self.attn: CausalSelfAttention = CausalSelfAttention(self._cfg)
        self.ln_2: nn.LayerNorm = nn.LayerNorm()
        self.mlp: MLP = MLP(self._cfg)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    _cfg: GPTConfig

    def setup(self):
        self.shared_weight = self.param(
            "shared_weight", _weight_init(), (self._cfg.vocab_size, self._cfg.n_embd)
        )
        self.wte: nn.Embed = nn.Embed(
            self._cfg.vocab_size,
            self._cfg.n_embd,
            embedding_init=lambda *_: self.shared_weight,
        )
        self.wpe: nn.Embed = nn.Embed(
            self._cfg.max_seq_len, self._cfg.n_embd, embedding_init=_weight_init()
        )
        self.h = [Block(self._cfg) for _ in range(self._cfg.n_layer)]
        self.ln_f: nn.LayerNorm = nn.LayerNorm()
        self.lm_head: nn.Dense = nn.Dense(
            self._cfg.vocab_size,
            use_bias=False,
            kernel_init=lambda *_: self.shared_weight.T,
        )

    def __call__(
        self, inputs: jax.Array, targets: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array | None]:
        T = inputs.shape[-1]
        assert T <= self._cfg.max_seq_len
        # Token and position embeddings.
        tok_emb = self.wte(inputs)  # (B, T, n_embd)
        pos = jnp.arange(0, T)  # (T,)
        pos_emb = self.wpe(pos)  # (T, n_embd)
        x = tok_emb + pos_emb
        # Transformer blocks.
        for block in self.h:
            x = block(x)
        # Final layers.
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # TODO: Check correctness.
            loss = jnp.mean(
                optax.softmax_cross_entropy(
                    logits, jax.nn.one_hot(targets, self._cfg.vocab_size)
                )
            )
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_name: str) -> tuple[nn.Module, dict]:
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
        cfg = GPTConfig(**cfg_dict)
        model = GPT(cfg)
        # TODO: Drop batch dimension if possible.
        params = model.init(
            jax.random.key(0),
            jnp.ones((1, cfg.max_seq_len), dtype=jnp.int32),
            jnp.ones((1, 1), dtype=jnp.int32),
        )
        assert isinstance(params, dict)

        model_hf = GPT2LMHeadModel.from_pretrained(model_name)
        assert isinstance(model_hf, tnn.Module)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()

        def copy_params_from_hf(params: dict, parent_path: str = "") -> int:
            num_copied = 0
            for k, v in params.items():
                path = f"{parent_path}.{k}" if parent_path else k
                if isinstance(v, dict):
                    num_copied += copy_params_from_hf(v, path)
                else:
                    if path in {"params.shared_weight", "params.lm_head.kernel"}:
                        continue
                    path = re.sub(r"h_(\d)", r"h.\1", path)
                    path = path.replace("params", "transformer")
                    path = path.replace("embedding", "weight")
                    path = path.replace("kernel", "weight")
                    path = path.replace("scale", "weight")
                    assert sd_hf[path].shape == params[k].shape
                    params[k] = jnp.array(sd_hf[path].numpy())
                    num_copied += 1
            return num_copied

        num_copied = copy_params_from_hf(params)
        assert num_copied == len(sd_keys_hf) - 1
        return model, params
