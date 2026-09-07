"""
Checkpoint utilities — save and load training state.

Checkpoint format:
    {
        "epoch":          int,
        "global_step":    int,
        "model_state":    OrderedDict,
        "optimizer_state": dict,
        "scheduler_state": dict | None,
        "best_val_loss":  float,
        "train_config":   dict,
        "model_config":   dict,
        "metrics":        dict,   # latest metrics at save time
    }
"""

import os
import json
import torch
from typing import Optional


def save_checkpoint(
    path: str,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    train_config: dict,
    model_config: dict,
    metrics: Optional[dict] = None,
    scheduler=None,
) -> None:
    """
    Save a complete training checkpoint.

    Args:
        path:          file path to save to (e.g. checkpoints/step_5000.pt)
        model:         SmallTransformer instance
        optimizer:     torch optimizer
        epoch:         current epoch number
        global_step:   total optimizer steps taken
        best_val_loss: best validation loss seen so far
        train_config:  dict representation of TrainConfig
        model_config:  dict representation of ModelConfig
        metrics:       latest metrics dict (loss, accuracy, etc.)
        scheduler:     optional lr scheduler
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch":           epoch,
        "global_step":     global_step,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss":   best_val_loss,
        "train_config":    train_config,
        "model_config":    model_config,
        "metrics":         metrics or {},
    }
    # Save to a temp file then rename (atomic on most filesystems)
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    device: str = "cpu",
) -> dict:
    """
    Load a checkpoint and restore model (and optionally optimizer) state.

    Args:
        path:       path to checkpoint file
        model:      SmallTransformer (already instantiated with matching config)
        optimizer:  if provided, loads optimizer state
        scheduler:  if provided, loads scheduler state
        device:     device to map tensors to

    Returns:
        The checkpoint dict (contains epoch, global_step, metrics, etc.)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the most recent checkpoint in a directory by global_step.

    Scans for *.pt files, loads the step number embedded in the filename
    or reads the step from the checkpoint header.
    Returns the path to the latest checkpoint, or None.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    ckpts = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") and not f.endswith(".tmp")
    ]
    if not ckpts:
        return None

    # Sort by embedded step number (fall back to mtime)
    def _step(path):
        try:
            ck = torch.load(path, map_location="cpu")
            return ck.get("global_step", 0)
        except Exception:
            return os.path.getmtime(path)

    return max(ckpts, key=_step)


def save_metrics(metrics: dict, path: str) -> None:
    """Append a metrics dict entry to a JSON-lines file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def load_metrics(path: str) -> list:
    """Load all metrics entries from a JSON-lines file."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
