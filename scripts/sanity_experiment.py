"""
Learning sanity experiment.

Purpose: verify that the training signal is real BEFORE the main 8M Colab run.

NOT measuring impressive accuracy. Measuring whether:
  - training loss decreases meaningfully
  - validation loss tracks training loss (not wildly diverging)
  - exact accuracy rises above zero during training

Setup:
  - Families:    single_op + multi_step only (simplest arithmetic)
  - Difficulty:  easy
  - Dataset:     2000 train / 400 val / 400 test (deduped, zero overlap)
  - Model:       debug config (~98K params) — trains on CPU in minutes
  - Format:      reasoning (full chain-of-thought)
  - Checkpoints: init (step 0), 25%, 50%, 75%, 100%

Each checkpoint is evaluated on the SAME frozen test set.
The test set is never seen during training.

Results written to: sanity_experiment_results.txt
"""

import sys, os, time, json, math, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "sanity_experiment_results.txt")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import get_debug_config
from model.transformer import SmallTransformer
from model.tokenizer import BPETokenizer
from data.generators.arithmetic import ArithmeticGenerator, build_corpus
from training.dataset import ReasoningDataset, collate_fn
from training.checkpoint import save_checkpoint, load_checkpoint
from evaluation.arithmetic import verify

lines = []
def log(msg=""): lines.append(str(msg)); print(str(msg))

# ── Config ────────────────────────────────────────────────────────────────────
SEED          = 7
FAMILIES      = ["single_op", "multi_step"]
DIFFICULTY    = "easy"
N_TRAIN       = 2000
N_VAL         = 400
N_TEST        = 400
VOCAB_SIZE    = 512
MAX_SEQ_LEN   = 128
BATCH_SIZE    = 32
LR            = 5e-3
WEIGHT_DECAY  = 0.01
MAX_EPOCHS    = 40
N_CHECKPOINTS = 5      # init + 4 evenly spaced

log("=" * 65)
log("LEARNING SANITY EXPERIMENT")
log("=" * 65)
log(f"  families={FAMILIES}, difficulty={DIFFICULTY}")
log(f"  n_train={N_TRAIN}, n_val={N_VAL}, n_test={N_TEST}")
log(f"  model=debug (~98K params), lr={LR}, epochs={MAX_EPOCHS}")
log()

# ── Data ──────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
gen = ArithmeticGenerator(seed=SEED, difficulty=DIFFICULTY, families=FAMILIES)
train_ex, val_ex, test_ex = gen.generate_all_splits(
    n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST
)

# Verify zero overlap
tp = set(e.problem for e in train_ex)
vp = set(e.problem for e in val_ex)
sp = set(e.problem for e in test_ex)
assert len(tp & vp) == 0 and len(tp & sp) == 0 and len(vp & sp) == 0, \
    "Split overlap detected!"
log(f"  Split overlap check: PASS (all 0)")
log(f"  Train: {len(train_ex)}, Val: {len(val_ex)}, Test: {len(test_ex)}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
corpus = build_corpus(train_ex, use_reasoning=True)
tok = BPETokenizer()
tok.train(corpus, vocab_size=VOCAB_SIZE, min_frequency=1, show_progress=False)
log(f"  Tokenizer: vocab_size={tok.vocab_size}")

# ── Datasets ──────────────────────────────────────────────────────────────────
train_ds = ReasoningDataset(train_ex, tok, max_seq_len=MAX_SEQ_LEN, use_reasoning=True)
val_ds   = ReasoningDataset(val_ex,   tok, max_seq_len=MAX_SEQ_LEN, use_reasoning=True)
test_ds  = ReasoningDataset(test_ex,  tok, max_seq_len=MAX_SEQ_LEN, use_reasoning=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, tok.pad_id))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=lambda b: collate_fn(b, tok.pad_id))

stats = train_ds.stats()
log(f"  Train tokens: {stats['total_tokens']:,}, avg_len={stats['avg_len']:.1f}")

# ── Model ─────────────────────────────────────────────────────────────────────
cfg = get_debug_config()
cfg.vocab_size   = tok.vocab_size
cfg.max_seq_len  = MAX_SEQ_LEN
cfg.pad_token_id = tok.pad_id
cfg.bos_token_id = tok.bos_id
cfg.eos_token_id = tok.eos_id
cfg.dropout      = 0.0   # no dropout for this small experiment

model = SmallTransformer(cfg)
n_params = model.count_parameters()
log(f"  Model: {n_params:,} trainable params  "
    f"(d={cfg.d_model}, L={cfg.n_layers}, H={cfg.n_heads})")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)

# ── Evaluation helper ─────────────────────────────────────────────────────────
@torch.no_grad()
def compute_val_loss():
    model.eval()
    total, n = 0.0, 0
    for x, y in val_loader:
        loss, _ = model(x, y)
        total += loss.item(); n += 1
    model.train()
    return total / max(n, 1)

def compute_test_accuracy(n_problems=200):
    """Greedy decode on n_problems frozen test examples, return exact accuracy."""
    model.eval()
    correct = 0
    subset = test_ex[:n_problems]
    for ex in subset:
        prompt = f"Problem: {ex.problem}\nReasoning:\n"
        ids = tok.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([ids], dtype=torch.long)
        out = inp.clone()
        for _ in range(40):
            ctx    = out[:, -MAX_SEQ_LEN:]
            logits = model(ctx)[:, -1, :]
            nxt    = logits.argmax(dim=-1, keepdim=True)
            out    = torch.cat([out, nxt], dim=1)
            if nxt.item() == tok.eos_id:
                break
        gen_ids  = out[0, len(ids):].tolist()
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        full     = prompt + gen_text
        vr = verify(full, ex.answer, ex.answer_num)
        if vr["correct"]:
            correct += 1
    model.train()
    return correct / len(subset)

# ── Checkpoint schedule ───────────────────────────────────────────────────────
steps_per_epoch = math.ceil(len(train_loader))
total_steps     = steps_per_epoch * MAX_EPOCHS
# Checkpoints at: 0%, 25%, 50%, 75%, 100%
ck_steps = [0,
            total_steps // 4,
            total_steps // 2,
            3 * total_steps // 4,
            total_steps]
ck_steps = sorted(set(ck_steps))

log(f"\n  Training: {total_steps} total steps, checkpoints at {ck_steps}")
log()

# Table header
log(f"  {'Checkpoint':<12} {'Step':>6} {'Train Loss':>12} "
    f"{'Val Loss':>10} {'Test Acc':>10} {'Time':>8}")
log(f"  {'-'*62}")

results_table = []
tmpdir = tempfile.mkdtemp()

# ── Training loop ─────────────────────────────────────────────────────────────
global_step = 0
t0 = time.time()

def record_checkpoint(label):
    train_loss_now = last_train_loss if last_train_loss is not None else float("nan")
    val_loss_now   = compute_val_loss()
    acc_now        = compute_test_accuracy(n_problems=min(200, N_TEST))
    elapsed        = time.time() - t0
    row = {
        "label":      label,
        "step":       global_step,
        "train_loss": round(train_loss_now, 4),
        "val_loss":   round(val_loss_now,   4),
        "test_acc":   round(acc_now,        4),
        "elapsed_s":  round(elapsed,        1),
    }
    results_table.append(row)
    log(f"  {label:<12} {global_step:>6} {train_loss_now:>12.4f} "
        f"{val_loss_now:>10.4f} {acc_now:>10.1%} {elapsed:>7.1f}s")
    # Save checkpoint
    ck_path = os.path.join(tmpdir, f"{label}.pt")
    save_checkpoint(path=ck_path, model=model, optimizer=optimizer,
                    epoch=0, global_step=global_step, best_val_loss=val_loss_now,
                    train_config={}, model_config=cfg.to_dict())

last_train_loss = None
ck_labels = ["init", "25pct", "50pct", "75pct", "final"]
ck_idx    = 0

model.train()

# Record init: measure actual initial loss before any training
model.eval()
with torch.no_grad():
    init_loss_total, init_n = 0.0, 0
    for x, y in train_loader:
        loss, _ = model(x, y)
        init_loss_total += loss.item()
        init_n += 1
        if init_n >= 5:   # sample 5 batches for speed
            break
last_train_loss = init_loss_total / max(init_n, 1)
model.train()

record_checkpoint(ck_labels[ck_idx]); ck_idx += 1

for epoch in range(MAX_EPOCHS):
    epoch_loss = 0.0
    n_batches  = 0
    for x, y in train_loader:
        loss, _ = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss  += loss.item()
        n_batches   += 1
        global_step += 1
        last_train_loss = epoch_loss / n_batches

        # Check if we hit a checkpoint step
        if ck_idx < len(ck_labels) and global_step >= ck_steps[ck_idx]:
            record_checkpoint(ck_labels[ck_idx])
            ck_idx += 1

# Final checkpoint if not already recorded
if ck_idx < len(ck_labels):
    record_checkpoint(ck_labels[ck_idx])

# ── Analysis ──────────────────────────────────────────────────────────────────
log()
log("=" * 65)
log("ANALYSIS")
log("=" * 65)

init_row  = results_table[0]
final_row = results_table[-1]

train_drop = init_row["train_loss"] - final_row["train_loss"]
val_drop   = init_row["val_loss"]   - final_row["val_loss"]
acc_gain   = final_row["test_acc"]  - init_row["test_acc"]

log(f"  Training loss:    {init_row['train_loss']:.4f} -> {final_row['train_loss']:.4f}  "
    f"(drop = {train_drop:.4f})")
log(f"  Validation loss:  {init_row['val_loss']:.4f} -> {final_row['val_loss']:.4f}  "
    f"(drop = {val_drop:.4f})")
log(f"  Test accuracy:    {init_row['test_acc']:.1%} -> {final_row['test_acc']:.1%}  "
    f"(gain = {acc_gain:.1%})")
log()

# Gate checks
gates_passed = True

if (not math.isnan(train_drop)) and final_row["train_loss"] < init_row["train_loss"] * 0.5:
    log("  [PASS] Training loss dropped by >50%")
elif math.isnan(init_row["train_loss"]):
    log("  [INFO] Init train loss was not measured (nan) — check script logic")
    gates_passed = False
else:
    log(f"  [FAIL] Training loss did not drop by >50% "
        f"({init_row['train_loss']:.4f} -> {final_row['train_loss']:.4f})")
    gates_passed = False

if final_row["val_loss"] < init_row["val_loss"]:
    log("  [PASS] Validation loss decreased")
else:
    log(f"  [FAIL] Validation loss did NOT decrease "
        f"({init_row['val_loss']:.4f} -> {final_row['val_loss']:.4f})")
    gates_passed = False

if final_row["test_acc"] > 0.02:
    log(f"  [PASS] Test accuracy rose above 2% (got {final_row['test_acc']:.1%})")
else:
    log(f"  [FAIL] Test accuracy still near zero ({final_row['test_acc']:.1%})")
    log("         Investigate: prompt format, answer extraction, generation length,")
    log("         tokenizer, or train/test distribution mismatch.")
    gates_passed = False

# Check val loss doesn't massively diverge from train loss
if final_row["val_loss"] < final_row["train_loss"] * 3.0:
    log("  [PASS] Val loss within 3x of train loss (no catastrophic overfitting)")
else:
    log(f"  [FAIL] Val loss ({final_row['val_loss']:.4f}) > 3x train loss "
        f"({final_row['train_loss']:.4f}) — possible overfitting")
    gates_passed = False

log()
if gates_passed:
    log("  OVERALL: PASS — learning signal confirmed.")
    log("  The training loop, data pipeline, and evaluation are all working.")
    log("  Ready to proceed to the main 8M Colab experiment.")
else:
    log("  OVERALL: FAIL — investigate before proceeding to Colab training.")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "config": {
        "families": FAMILIES, "difficulty": DIFFICULTY,
        "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
        "model_params": n_params, "lr": LR, "epochs": MAX_EPOCHS,
        "vocab_size": tok.vocab_size,
    },
    "table": results_table,
    "gates_passed": gates_passed,
}
json_path = os.path.join(ROOT, "sanity_experiment_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

log(f"\n  Results saved to: {OUT}")
log(f"  JSON saved to:    {json_path}")
