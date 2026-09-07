"""
TrainConfig — all hyperparameters for the training pipeline.

Colab resource presets:
    debug        — tiny model + 200 examples, CPU/GPU, for sanity checks
    colab_small  — 7.95M-param model (8m preset) + 20k examples
    colab_medium — 7.95M-param model (8m preset) + 50k examples
"""

from dataclasses import dataclass, asdict, field
from typing import Optional
import json
import os


@dataclass
class TrainConfig:
    # ── Experiment identity ───────────────────────────────────────────────────
    experiment_name: str = "exp_baseline"
    output_dir:      str = "experiments/results/exp_baseline"
    seed:            int = 42

    # ── Model ─────────────────────────────────────────────────────────────────
    model_size: str = "8m"          # key into model.config.PRESETS

    # ── Data ──────────────────────────────────────────────────────────────────
    n_train:        int = 20_000
    n_val:          int = 2_000
    n_test:         int = 1_000
    difficulty:     str = "medium"  # "easy" | "medium" | "hard"
    use_reasoning:  bool = True     # True = include chain-of-thought steps
    data_seed:      int = 42
    tokenizer_vocab_size: int = 4096

    # ── Training loop ─────────────────────────────────────────────────────────
    max_epochs:             int   = 20
    max_steps:              int   = -1      # -1 = run to max_epochs
    batch_size:             int   = 64
    grad_accumulation_steps: int  = 1       # effective batch = batch_size * accum
    eval_every_n_steps:     int   = 500
    save_every_n_steps:     int   = 1000
    log_every_n_steps:      int   = 50
    eval_n_problems:        int   = 500     # problems for reasoning accuracy eval
    n_generalization_levels: int  = 5       # levels in generalization benchmark

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer:      str   = "adamw"
    learning_rate:  float = 3e-4
    weight_decay:   float = 0.1
    beta1:          float = 0.9
    beta2:          float = 0.95
    grad_clip:      float = 1.0

    # ── LR schedule ───────────────────────────────────────────────────────────
    scheduler:          str   = "cosine"   # "cosine" | "linear" | "constant"
    warmup_steps:       int   = 100
    min_lr_ratio:       float = 0.1        # min LR = lr * min_lr_ratio

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp: bool = True    # will be disabled automatically if no CUDA

    # ── Sequence ──────────────────────────────────────────────────────────────
    max_seq_len: int = 256   # must match ModelConfig.max_seq_len

    # ── Generation (for mid-training evals) ──────────────────────────────────
    gen_max_new_tokens: int   = 64
    gen_temperature:    float = 0.0   # 0 = greedy

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_from: Optional[str] = None   # path to checkpoint to resume from

    # ── Paths (populated automatically) ──────────────────────────────────────
    checkpoint_dir:     str = ""
    tokenizer_dir:      str = ""
    metrics_file:       str = ""
    model_config_file:  str = ""
    train_config_file:  str = ""
    eval_file:          str = ""
    samples_file:       str = ""

    def __post_init__(self):
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if not self.tokenizer_dir:
            self.tokenizer_dir = os.path.join(self.output_dir, "tokenizer")
        if not self.metrics_file:
            self.metrics_file = os.path.join(self.output_dir, "metrics.jsonl")
        if not self.model_config_file:
            self.model_config_file = os.path.join(self.output_dir, "model_config.json")
        if not self.train_config_file:
            self.train_config_file = os.path.join(self.output_dir, "train_config.json")
        if not self.eval_file:
            self.eval_file = os.path.join(self.output_dir, "evaluation.json")
        if not self.samples_file:
            self.samples_file = os.path.join(self.output_dir, "samples.json")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str = "") -> None:
        p = path or self.train_config_file
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ── Colab resource presets ────────────────────────────────────────────────────

def get_debug_train_config() -> TrainConfig:
    """Ultra-tiny config for CPU-only unit testing and overfit checks."""
    return TrainConfig(
        experiment_name         = "debug",
        output_dir              = "experiments/results/debug",
        model_size              = "debug",
        n_train                 = 200,
        n_val                   = 50,
        n_test                  = 50,
        difficulty              = "easy",
        use_reasoning           = True,
        tokenizer_vocab_size    = 512,
        max_epochs              = 10,
        batch_size              = 16,
        eval_every_n_steps      = 20,
        save_every_n_steps      = 50,
        log_every_n_steps       = 10,
        eval_n_problems         = 50,
        learning_rate           = 1e-3,
        warmup_steps            = 10,
        max_seq_len             = 128,
        gen_max_new_tokens      = 32,
        use_amp                 = False,
    )


def get_colab_small_train_config() -> TrainConfig:
    """~8M model, ~20k examples — baseline experiment for Colab T4."""
    return TrainConfig(
        experiment_name         = "colab_small_baseline",
        output_dir              = "experiments/results/colab_small_baseline",
        model_size              = "8m",
        n_train                 = 20_000,
        n_val                   = 2_000,
        n_test                  = 1_000,
        difficulty              = "medium",
        use_reasoning           = True,
        tokenizer_vocab_size    = 4096,
        max_epochs              = 30,
        batch_size              = 64,
        grad_accumulation_steps = 2,
        eval_every_n_steps      = 500,
        save_every_n_steps      = 1000,
        log_every_n_steps       = 50,
        eval_n_problems         = 500,
        learning_rate           = 3e-4,
        warmup_steps            = 200,
        max_seq_len             = 256,
        gen_max_new_tokens      = 64,
        use_amp                 = True,
    )


def get_colab_medium_train_config() -> TrainConfig:
    """~8M model, ~50k examples — longer training run."""
    return TrainConfig(
        experiment_name         = "colab_medium",
        output_dir              = "experiments/results/colab_medium",
        model_size              = "8m",
        n_train                 = 50_000,
        n_val                   = 5_000,
        n_test                  = 2_000,
        difficulty              = "medium",
        use_reasoning           = True,
        tokenizer_vocab_size    = 4096,
        max_epochs              = 50,
        batch_size              = 64,
        grad_accumulation_steps = 4,
        eval_every_n_steps      = 1000,
        save_every_n_steps      = 2000,
        log_every_n_steps       = 100,
        eval_n_problems         = 1000,
        learning_rate           = 3e-4,
        warmup_steps            = 400,
        max_seq_len             = 256,
        gen_max_new_tokens      = 64,
        use_amp                 = True,
    )


TRAIN_PRESETS = {
    "debug":        get_debug_train_config,
    "colab_small":  get_colab_small_train_config,
    "colab_medium": get_colab_medium_train_config,
}


def get_train_config_by_name(name: str) -> TrainConfig:
    if name not in TRAIN_PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Choose from: {list(TRAIN_PRESETS)}")
    return TRAIN_PRESETS[name]()
