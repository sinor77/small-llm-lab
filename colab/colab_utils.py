"""
colab_utils.py — zero-configuration helpers for the evaluation notebook.

This module handles all the messy environment detection so the notebook
cells stay clean. Import this once at the top of the notebook and all
path/device/checkpoint logic is handled automatically.
"""

import os
import sys
import glob
import time
import subprocess


# ── Repository setup ──────────────────────────────────────────────────────────

REPO_URL  = "https://github.com/sinor77/small-llm-lab.git"
REPO_NAME = "small-llm-lab"


def setup_repo() -> str:
    """
    Clone or update the repository, change into it, and add it to sys.path.
    Returns the absolute path of the repository root.
    Works in a fresh Colab runtime or an existing one.
    """
    # Already inside the repo?
    if os.path.exists("model") and os.path.exists("training"):
        root = os.getcwd()
        print(f"[Repo] Already in repo: {root}")
        _git("pull", "origin", "main", cwd=root)
        sys.path.insert(0, root)
        return root

    # Repo cloned as a subdirectory?
    if os.path.exists(REPO_NAME):
        root = os.path.abspath(REPO_NAME)
        print(f"[Repo] Found existing clone: {root}")
        _git("pull", "origin", "main", cwd=root)
        os.chdir(root)
        sys.path.insert(0, root)
        return root

    # Fresh clone
    print(f"[Repo] Cloning {REPO_URL} ...")
    _git("clone", REPO_URL)
    root = os.path.abspath(REPO_NAME)
    os.chdir(root)
    sys.path.insert(0, root)
    print(f"[Repo] Cloned to {root}")
    return root


def _git(*args, cwd=None):
    result = subprocess.run(
        ["git"] + list(args),
        cwd=cwd or os.getcwd(),
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            print(f"  git: {line}")
    else:
        print(f"  git warning: {result.stderr.strip()[:200]}")


def install_deps():
    """Install Python dependencies if not already present."""
    try:
        import tokenizers  # noqa
        print("[Deps] tokenizers already installed.")
    except ImportError:
        print("[Deps] Installing tokenizers ...")
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "tokenizers>=0.15.0", "-q"], check=True)
        print("[Deps] Done.")


# ── Device detection ──────────────────────────────────────────────────────────

def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] CUDA: {name} ({mem:.1f} GB)")
        return "cuda"
    print("[Device] No GPU detected — using CPU (evaluation will be slow)")
    return "cpu"


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def find_best_checkpoint(repo_root: str) -> str | None:
    """
    Search for best.pt in the following order:
      1. experiments/results/*/checkpoints/best.pt  (newest experiment first)
      2. /content/*.pt and /content/**/*.pt
      3. /content/drive/MyDrive/**/*.pt  (if Drive is mounted)

    Returns the absolute path of the best checkpoint found, or None.
    """
    candidates = []

    # 1. experiments/results/ within the repo
    pattern = os.path.join(repo_root, "experiments", "results",
                           "*", "checkpoints", "best.pt")
    for p in glob.glob(pattern):
        candidates.append(p)

    # Sort by modification time — newest first
    candidates.sort(key=os.path.getmtime, reverse=True)

    if candidates:
        chosen = candidates[0]
        print(f"[Checkpoint] Found in experiments/results:")
        for c in candidates:
            marker = " <-- selected" if c == chosen else ""
            print(f"  {c}{marker}")
        return chosen

    # 2. Anywhere in /content (Colab VM root)
    print("[Checkpoint] Not found in experiments/. Searching /content ...")
    for p in glob.glob("/content/**/best.pt", recursive=True):
        candidates.append(p)
    candidates.sort(key=os.path.getmtime, reverse=True)
    if candidates:
        chosen = candidates[0]
        print(f"[Checkpoint] Found in /content: {chosen}")
        return chosen

    # 3. Google Drive
    drive_root = "/content/drive/MyDrive"
    if os.path.isdir(drive_root):
        print("[Checkpoint] Searching Google Drive ...")
        for p in glob.glob(os.path.join(drive_root, "**/best.pt"), recursive=True):
            candidates.append(p)
        candidates.sort(key=os.path.getmtime, reverse=True)
        if candidates:
            chosen = candidates[0]
            print(f"[Checkpoint] Found on Drive: {chosen}")
            return chosen

    return None


def get_experiment_dir(checkpoint_path: str) -> str:
    """
    Given experiments/results/exp_name/checkpoints/best.pt,
    return experiments/results/exp_name/.
    Works for any checkpoint depth.
    """
    ck_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    # Walk up until we find the tokenizer/ sibling
    candidate = ck_dir
    for _ in range(5):
        if os.path.isdir(os.path.join(candidate, "tokenizer")):
            return candidate
        candidate = os.path.dirname(candidate)
    # Fallback: parent of checkpoints/
    return os.path.dirname(ck_dir)


# ── Missing-checkpoint recovery ───────────────────────────────────────────────

def handle_missing_checkpoint(repo_root: str) -> str | None:
    """
    Called when no checkpoint is found anywhere.
    Tries Google Drive first, then offers to train.
    Returns the checkpoint path if recovery succeeds, or None.
    """
    print()
    print("=" * 60)
    print("  NO CHECKPOINT FOUND")
    print("=" * 60)
    print()
    print("  The trained model (best.pt) is not present in this")
    print("  Colab session. This happens when:")
    print("    - The Colab runtime was reset after training")
    print("    - You opened a fresh runtime without restoring files")
    print()
    print("  Trying to find it on Google Drive...")

    # Attempt Drive mount
    try:
        from google.colab import drive  # noqa
        print("  Mounting Google Drive...")
        drive.mount("/content/drive", force_remount=False)
        drive_root = "/content/drive/MyDrive"
        hits = glob.glob(os.path.join(drive_root, "**/best.pt"), recursive=True)
        hits.sort(key=os.path.getmtime, reverse=True)
        if hits:
            print(f"  Found on Drive: {hits[0]}")
            return hits[0]
        else:
            print("  best.pt not found on Google Drive either.")
    except Exception as e:
        print(f"  Could not mount Drive: {e}")

    # Offer to train
    print()
    print("=" * 60)
    print("  OPTION: Run training now")
    print("=" * 60)
    print()
    print("  No saved checkpoint is available. The notebook will now")
    print("  run a training job to create one.")
    print("  Estimated time: 10-30 minutes on T4 GPU.")
    print()

    try:
        answer = input("  Train the model now? [y/N]: ").strip().lower()
    except EOFError:
        answer = "y"   # non-interactive (run-all) → auto-train

    if answer in ("y", "yes", ""):
        print()
        print("[Training] Starting training pipeline...")
        ck_path = _run_training(repo_root)
        return ck_path
    else:
        print()
        print("  Training skipped. Cannot proceed without a checkpoint.")
        print("  To restore a checkpoint manually, upload best.pt and its")
        print("  tokenizer/ directory to:")
        print("    experiments/results/colab_small_baseline/")
        return None


def _run_training(repo_root: str) -> str | None:
    """Run training/train.py with the colab_small preset and return the checkpoint path."""
    sys.path.insert(0, repo_root)
    from training.train import train
    from training.config import get_train_config_by_name

    train_config = get_train_config_by_name("colab_small")
    train(train_config)

    best = os.path.join(repo_root, train_config.checkpoint_dir, "best.pt")
    if os.path.exists(best):
        print(f"[Training] Complete. Checkpoint: {best}")
        return best
    else:
        print("[Training] WARNING: training finished but best.pt not found.")
        return find_best_checkpoint(repo_root)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str):
    """
    Load model, tokenizer, and checkpoint metadata from a checkpoint path.
    The tokenizer is located automatically relative to the checkpoint.
    Returns (model, tokenizer, checkpoint_dict, experiment_dir).
    """
    from inference.generate import load_model_and_tokenizer

    print(f"[Model] Loading from: {checkpoint_path}")
    model, tokenizer, ck = load_model_and_tokenizer(checkpoint_path, device=device)
    n = model.count_parameters()
    step = ck.get("global_step", "?")
    cfg  = model.config
    exp_dir = get_experiment_dir(checkpoint_path)

    print(f"  Parameters:  {n:,}")
    print(f"  Step:        {step}")
    print(f"  d{cfg.d_model}/L{cfg.n_layers}/H{cfg.n_heads}/ff{cfg.d_ff}, "
          f"vocab={cfg.vocab_size}")
    print(f"  Tokenizer:   {tokenizer.vocab_size} tokens")
    print(f"  Exp dir:     {exp_dir}")
    return model, tokenizer, ck, exp_dir
