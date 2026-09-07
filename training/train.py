"""
Training pipeline for Small Reasoning LLM Lab.

Usage:
    # Run with a preset:
    python training/train.py --preset debug
    python training/train.py --preset colab_small

    # Run with a custom config JSON:
    python training/train.py --config experiments/configs/my_experiment.json

    # Resume training:
    python training/train.py --preset colab_small \
        --resume experiments/results/colab_small_baseline/checkpoints/step_5000.pt

Pipeline:
    1. Load config
    2. Set random seed
    3. Detect device (CUDA / MPS / CPU)
    4. Generate dataset (train / val / test)
    5. Train BPE tokenizer on train corpus
    6. Save tokenizer
    7. Build DataLoaders
    8. Instantiate model from random init
    9. Training loop:
       a. Forward pass + loss
       b. Backward pass
       c. Gradient clipping
       d. Optimizer step
       e. LR schedule step
       f. Periodic validation
       g. Periodic reasoning evaluation
       h. Checkpoint save
    10. Final evaluation
    11. Save all artifacts
"""

import os
import sys
import json
import math
import time
import argparse
import random
import numpy as np

# Make sure the project root is on the Python path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from model.config import ModelConfig, get_config_by_name
from model.transformer import SmallTransformer
from model.tokenizer import BPETokenizer

from data.generators.arithmetic import ArithmeticGenerator, build_corpus

from training.config import TrainConfig, get_train_config_by_name
from training.dataset import make_dataloader, ReasoningDataset
from training.checkpoint import (
    save_checkpoint, load_checkpoint,
    find_latest_checkpoint, save_metrics, load_metrics
)

from evaluation.arithmetic import verify
from evaluation.benchmark import run_benchmark, run_generalization_benchmark


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] CUDA: {name} ({mem:.1f} GB)")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device] Apple MPS")
        return "mps"
    print("[Device] CPU (no GPU detected)")
    return "cpu"


def build_lr_scheduler(optimizer, train_config: TrainConfig, total_steps: int):
    """Return a torch LR scheduler based on train_config."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LambdaLR

    warmup = train_config.warmup_steps
    min_lr = train_config.learning_rate * train_config.min_lr_ratio

    # Warmup scheduler: linear ramp from 0 to 1.0 over warmup_steps
    warmup_sched = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup)

    remaining = max(1, total_steps - warmup)

    if train_config.scheduler == "cosine":
        main_sched = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=min_lr)
    elif train_config.scheduler == "linear":
        main_sched = LinearLR(optimizer, start_factor=1.0,
                              end_factor=train_config.min_lr_ratio, total_iters=remaining)
    else:  # constant
        main_sched = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, main_sched],
        milestones=[warmup],
    )
    return scheduler


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_val_loss(model, val_loader, device: str, amp: bool = False) -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    ctx = torch.autocast(device_type=device, dtype=torch.float16) if (amp and device == "cuda") \
          else torch.no_grad()

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(amp and device == "cuda")):
            loss, _ = model(x, y)
        total_loss += loss.item()
        n_batches  += 1

    model.train()
    return total_loss / max(n_batches, 1)


# ── Main training function ────────────────────────────────────────────────────

def train(train_config: TrainConfig) -> dict:
    """
    Run the full training pipeline.

    Returns:
        dict with final metrics
    """
    set_seed(train_config.seed)
    os.makedirs(train_config.output_dir, exist_ok=True)

    # ── 1. Device detection ───────────────────────────────────────────────────
    device = detect_device()
    use_amp = train_config.use_amp and (device == "cuda")

    # ── 2. Generate dataset ───────────────────────────────────────────────────
    print(f"\n[Data] Generating dataset (seed={train_config.data_seed}, "
          f"difficulty={train_config.difficulty})")
    generator = ArithmeticGenerator(
        seed=train_config.data_seed,
        difficulty=train_config.difficulty,
    )
    # Use generate_all_splits() to guarantee zero cross-split problem overlap.
    train_examples, val_examples, test_examples = generator.generate_all_splits(
        n_train=train_config.n_train,
        n_val=train_config.n_val,
        n_test=train_config.n_test,
    )

    print(f"  Train: {len(train_examples)} | Val: {len(val_examples)} | Test: {len(test_examples)}")

    # ── 3. Train tokenizer ────────────────────────────────────────────────────
    tokenizer_path = os.path.join(train_config.output_dir, "tokenizer")
    tok_exists = os.path.exists(os.path.join(tokenizer_path, "tokenizer.json"))

    if tok_exists and train_config.resume_from:
        print(f"[Tokenizer] Loading existing tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print(f"[Tokenizer] Training BPE tokenizer (vocab_size={train_config.tokenizer_vocab_size})")
        train_corpus = build_corpus(train_examples, use_reasoning=train_config.use_reasoning)
        tokenizer = BPETokenizer()
        tokenizer.train(
            texts=train_corpus,
            vocab_size=train_config.tokenizer_vocab_size,
            show_progress=True,
        )
        tokenizer.save(tokenizer_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── 4. Build DataLoaders ──────────────────────────────────────────────────
    print("\n[Data] Building DataLoaders")
    train_loader = make_dataloader(
        train_examples, tokenizer,
        max_seq_len=train_config.max_seq_len,
        batch_size=train_config.batch_size,
        use_reasoning=train_config.use_reasoning,
        shuffle=True,
    )
    val_loader = make_dataloader(
        val_examples, tokenizer,
        max_seq_len=train_config.max_seq_len,
        batch_size=train_config.batch_size,
        use_reasoning=train_config.use_reasoning,
        shuffle=False,
    )

    # ── 5. Model ──────────────────────────────────────────────────────────────
    model_config = get_config_by_name(train_config.model_size)
    model_config.vocab_size    = tokenizer.vocab_size
    model_config.max_seq_len   = train_config.max_seq_len
    model_config.pad_token_id  = tokenizer.pad_id
    model_config.bos_token_id  = tokenizer.bos_id
    model_config.eos_token_id  = tokenizer.eos_id
    model_config.tokenizer_path = tokenizer_path

    model = SmallTransformer(model_config).to(device)
    n_params = model.count_parameters()
    print(f"\n[Model] {train_config.model_size} | {n_params:,} parameters")
    print(f"  Config: d_model={model_config.d_model}, n_layers={model_config.n_layers}, "
          f"n_heads={model_config.n_heads}, d_ff={model_config.d_ff}")

    # ── 6. Optimizer + Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / train_config.grad_accumulation_steps)
    total_steps = (
        train_config.max_steps if train_config.max_steps > 0
        else steps_per_epoch * train_config.max_epochs
    )
    scheduler = build_lr_scheduler(optimizer, train_config, total_steps)
    scaler    = GradScaler(enabled=use_amp)

    # ── 7. Resume ─────────────────────────────────────────────────────────────
    start_epoch   = 0
    global_step   = 0
    best_val_loss = float("inf")

    resume_path = train_config.resume_from or find_latest_checkpoint(train_config.checkpoint_dir)
    if resume_path and os.path.exists(resume_path):
        print(f"\n[Resume] Loading checkpoint: {resume_path}")
        ck = load_checkpoint(resume_path, model, optimizer, scheduler, device=device)
        start_epoch   = ck.get("epoch", 0)
        global_step   = ck.get("global_step", 0)
        best_val_loss = ck.get("best_val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch}, step {global_step}")
    else:
        # Save initial configs
        model_config.to_json(train_config.model_config_file)
        train_config.to_json()
        print(f"\n[Config] Saved to {train_config.output_dir}")

    # ── 8. Training loop ──────────────────────────────────────────────────────
    print(f"\n[Train] Starting training: {total_steps} total optimizer steps")
    model.train()
    t_start = time.time()

    train_stats = {
        "n_params":    n_params,
        "train_steps": 0,
        "best_val_loss": float("inf"),
        "final_val_loss": float("inf"),
        "final_train_loss": float("inf"),
        "reasoning_accuracy": 0.0,
    }

    # ── Progression checkpoint schedule ─────────────────────────────────────
    # Save init, early (~10%), mid (~50%), and final regardless of normal schedule.
    # These allow post-hoc analysis of learning progression.
    early_step = max(1, total_steps // 10)
    mid_step   = max(1, total_steps // 2)
    progression_checkpoints = {
        0:          "init",
        early_step: "early",
        mid_step:   "mid",
    }

    epoch = start_epoch   # ensure 'epoch' is defined for the init checkpoint save

    def _save_progression_ckpt(label: str) -> None:
        path = os.path.join(train_config.checkpoint_dir, f"{label}.pt")
        save_checkpoint(
            path=path, model=model, optimizer=optimizer, epoch=epoch,
            global_step=global_step, best_val_loss=best_val_loss,
            train_config=train_config.to_dict(),
            model_config=model_config.to_dict(),
            scheduler=scheduler,
        )
        print(f"    [Ckpt] Progression checkpoint '{label}' saved at step {global_step}")

    accum_loss  = 0.0
    accum_steps = 0
    optimizer.zero_grad()

    # Save init checkpoint (step 0) before any training
    _save_progression_ckpt("init")

    for epoch in range(start_epoch, train_config.max_epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward + loss
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                 enabled=use_amp):
                loss, _ = model(x, y)
                loss    = loss / train_config.grad_accumulation_steps

            scaler.scale(loss).backward()
            accum_loss  += loss.item() * train_config.grad_accumulation_steps
            accum_steps += 1

            # Optimizer step after accumulation
            if accum_steps % train_config.grad_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = accum_loss / accum_steps
                accum_loss = 0.0
                accum_steps = 0

                # ── Progression checkpoints (init/early/mid) ──────────────────
                if global_step in progression_checkpoints:
                    _save_progression_ckpt(progression_checkpoints[global_step])

                # ── Logging ───────────────────────────────────────────────────
                if global_step % train_config.log_every_n_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - t_start
                    print(
                        f"  epoch {epoch+1:3d} | step {global_step:6d}/{total_steps} | "
                        f"loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s"
                    )
                    save_metrics({
                        "step":         global_step,
                        "epoch":        epoch + 1,
                        "train_loss":   avg_loss,
                        "lr":           lr,
                        "elapsed_s":    elapsed,
                    }, train_config.metrics_file)
                    train_stats["final_train_loss"] = avg_loss

                # ── Validation ────────────────────────────────────────────────
                if global_step % train_config.eval_every_n_steps == 0:
                    val_loss = evaluate_val_loss(model, val_loader, device, use_amp)
                    print(f"    [Val] step {global_step} | val_loss {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            path=os.path.join(train_config.checkpoint_dir, "best.pt"),
                            model=model, optimizer=optimizer, epoch=epoch,
                            global_step=global_step, best_val_loss=best_val_loss,
                            train_config=train_config.to_dict(),
                            model_config=model_config.to_dict(),
                            metrics={"val_loss": val_loss},
                            scheduler=scheduler,
                        )
                        print(f"    [Ckpt] New best model saved (val_loss={val_loss:.4f})")

                    save_metrics({
                        "step":       global_step,
                        "val_loss":   val_loss,
                        "best_val_loss": best_val_loss,
                    }, train_config.metrics_file)
                    train_stats["final_val_loss"] = val_loss
                    train_stats["best_val_loss"]  = best_val_loss

                    # Run a quick reasoning accuracy probe
                    if train_config.eval_n_problems > 0:
                        bench = run_benchmark(
                            model=model,
                            tokenizer=tokenizer,
                            generator=generator,
                            n_problems=min(100, train_config.eval_n_problems),
                            split="val",
                            max_new_tokens=train_config.gen_max_new_tokens,
                            temperature=0.0,
                            use_reasoning=train_config.use_reasoning,
                            max_samples_to_save=5,
                            device=device,
                        )
                        acc = bench.accuracy
                        print(f"    [Eval] reasoning accuracy = {acc:.1%}")
                        save_metrics({
                            "step":               global_step,
                            "reasoning_accuracy": acc,
                            "family_accuracy":    bench.family_accuracy,
                        }, train_config.metrics_file)
                        train_stats["reasoning_accuracy"] = acc
                    model.train()

                # ── Periodic checkpoint ───────────────────────────────────────
                if global_step % train_config.save_every_n_steps == 0:
                    ck_path = os.path.join(
                        train_config.checkpoint_dir, f"step_{global_step:07d}.pt"
                    )
                    save_checkpoint(
                        path=ck_path,
                        model=model, optimizer=optimizer, epoch=epoch,
                        global_step=global_step, best_val_loss=best_val_loss,
                        train_config=train_config.to_dict(),
                        model_config=model_config.to_dict(),
                        scheduler=scheduler,
                    )

                # ── Stop if max_steps reached ─────────────────────────────────
                if train_config.max_steps > 0 and global_step >= train_config.max_steps:
                    break

        if train_config.max_steps > 0 and global_step >= train_config.max_steps:
            break

    # ── 9. Final evaluation ───────────────────────────────────────────────────
    # Save 'mid' if not yet saved (e.g. training stopped early)
    if global_step < mid_step:
        _save_progression_ckpt("mid")
    # Always save a 'final' progression checkpoint
    _save_progression_ckpt("final")

    print("\n[Eval] Running final evaluation on test split...")
    # Load best checkpoint for final evaluation
    best_ck_path = os.path.join(train_config.checkpoint_dir, "best.pt")
    if os.path.exists(best_ck_path):
        load_checkpoint(best_ck_path, model, device=device)
        print(f"  Loaded best checkpoint for final eval")

    bench = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        generator=generator,
        n_problems=train_config.eval_n_problems,
        split="test",
        max_new_tokens=train_config.gen_max_new_tokens,
        temperature=0.0,
        use_reasoning=train_config.use_reasoning,
        max_samples_to_save=50,
        device=device,
    )
    print(bench.summary_str())
    bench.save(train_config.eval_file)

    # ── Generalization benchmark ──────────────────────────────────────────────
    print("\n[Eval] Running generalization benchmark...")
    # Build exclusion set from all known problems to guarantee no overlap
    all_known_problems = set(e.problem for e in train_examples + val_examples + test_examples)
    gen_results = run_generalization_benchmark(
        model=model, tokenizer=tokenizer,
        generator=generator,
        n_per_level=min(200, train_config.eval_n_problems),
        device=device,
        use_reasoning=train_config.use_reasoning,
        exclude_problems=all_known_problems,
    )
    for level, r in gen_results.items():
        print(f"  Level {level}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")

    gen_path = os.path.join(train_config.output_dir, "generalization.json")
    with open(gen_path, "w") as f:
        json.dump(gen_results, f, indent=2)

    # ── Save samples ──────────────────────────────────────────────────────────
    with open(train_config.samples_file, "w") as f:
        json.dump(bench.samples, f, indent=2)

    total_time = time.time() - t_start
    train_stats.update({
        "train_steps":        global_step,
        "total_time_s":       total_time,
        "n_train":            len(train_examples),
        "n_val":              len(val_examples),
        "n_test":             len(test_examples),
        "vocab_size":         tokenizer.vocab_size,
        "final_accuracy":     bench.accuracy,
        "family_accuracy":    bench.family_accuracy,
        "gen_level_accuracy": gen_results,
    })

    summary_path = os.path.join(train_config.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(train_stats, f, indent=2)

    print(f"\n[Done] Training complete in {total_time/60:.1f} min")
    print(f"  Final test accuracy: {bench.accuracy:.1%}")
    print(f"  Results saved to: {train_config.output_dir}")

    return train_stats


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Small Reasoning LLM")
    parser.add_argument("--preset",  type=str, default=None,
                        help="Training preset: debug | colab_small | colab_medium")
    parser.add_argument("--config",  type=str, default=None,
                        help="Path to a train_config.json file")
    parser.add_argument("--resume",  type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    if args.config:
        train_config = TrainConfig.from_json(args.config)
    elif args.preset:
        train_config = get_train_config_by_name(args.preset)
    else:
        print("No preset or config specified. Defaulting to 'debug'.")
        train_config = get_train_config_by_name("debug")

    if args.resume:
        train_config.resume_from = args.resume
    if args.output_dir:
        train_config.output_dir = args.output_dir

    train(train_config)


if __name__ == "__main__":
    main()
