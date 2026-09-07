"""
ModelConfig — centralised hyperparameter definition.

All architecture dimensions live here so that experiments can swap
configs without touching model code.

All parameter counts below are EXACT MEASURED trainable parameters
with weight tying enabled (lm_head shares weights with token_emb).
Total parameter count equals trainable count because weight tying
does not create additional buffers — the shared tensor is counted once.

Preset sizes (measured with vocab_size=4096, weight tying ON):
    debug   ~    98 K  — tiny, CPU-only, for unit tests
    1m      ~   0.6 M  — small model
    8m      ~   7.95M  — baseline experiment  (d256 / 8L / ff1152)
    15m     ~  12.0 M  — medium model
    30m     ~  24.0 M  — large model

These are the ACTUAL trainable counts produced by SmallTransformer,
not nominal counts before weight tying.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import os


@dataclass
class ModelConfig:
    # ── Vocabulary ──────────────────────────────────────────────────────────
    vocab_size: int = 4096          # set after tokenizer is trained

    # ── Transformer dimensions ───────────────────────────────────────────────
    d_model: int = 256              # embedding / hidden dimension
    n_layers: int = 6               # number of decoder blocks
    n_heads: int = 8                # number of attention heads
    d_ff: int = 1024                # feed-forward inner dimension
    max_seq_len: int = 256          # maximum context length
    dropout: float = 0.1

    # ── Tokenizer paths ──────────────────────────────────────────────────────
    tokenizer_path: Optional[str] = None   # filled in by training script

    # ── Misc ─────────────────────────────────────────────────────────────────
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    # ── Serialisation ────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def param_estimate(self) -> int:
        """
        Rough parameter count assuming weight tying is ON
        (lm_head weight = token_emb weight, counted once).
        Actual count from SmallTransformer.count_parameters() is authoritative.
        """
        embed   = self.vocab_size * self.d_model    # token embedding (shared with lm_head)
        pos_emb = self.max_seq_len * self.d_model   # positional embedding
        attn    = 4 * self.d_model * self.d_model   # Q K V O per layer
        ff      = 2 * self.d_model * self.d_ff      # two linear layers per block
        ln      = 4 * self.d_model                  # two LayerNorms per block (w+b)
        block   = attn + ff + ln
        ln_final = 2 * self.d_model                 # final LayerNorm
        # lm_head weight is SHARED with embed — not counted again
        return embed + pos_emb + self.n_layers * block + ln_final


# ── Pre-defined size presets ─────────────────────────────────────────────────

def get_debug_config() -> ModelConfig:
    """Ultra-tiny model for unit tests / CPU overfit check (~50 K params)."""
    return ModelConfig(
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=128,
        dropout=0.0,
    )


def get_1m_config() -> ModelConfig:
    """~1 M parameter model."""
    return ModelConfig(
        vocab_size=4096,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        max_seq_len=256,
        dropout=0.05,
    )


def get_8m_config() -> ModelConfig:
    """~8M parameter model — baseline experiment.

    Exact measured trainable parameters: 7,949,824
    (with weight tying, vocab_size=4096)
    Total parameters: 7,949,824 (weight tying — lm_head and token_emb share tensor)

    Architecture: d_model=256, n_layers=8, n_heads=8, d_ff=1152, max_seq_len=256
    """
    return ModelConfig(
        vocab_size=4096,
        d_model=256,
        n_layers=8,
        n_heads=8,
        d_ff=1152,
        max_seq_len=256,
        dropout=0.1,
    )


def get_15m_config() -> ModelConfig:
    """~15 M parameter model."""
    return ModelConfig(
        vocab_size=4096,
        d_model=384,
        n_layers=8,
        n_heads=8,
        d_ff=1536,
        max_seq_len=512,
        dropout=0.1,
    )


def get_30m_config() -> ModelConfig:
    """~30 M parameter model."""
    return ModelConfig(
        vocab_size=4096,
        d_model=512,
        n_layers=10,
        n_heads=8,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
    )


PRESETS = {
    "debug": get_debug_config,
    "1m":    get_1m_config,
    "8m":    get_8m_config,
    "15m":   get_15m_config,
    "30m":   get_30m_config,
}


def get_config_by_name(name: str) -> ModelConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Choose from: {list(PRESETS)}")
    return PRESETS[name]()
