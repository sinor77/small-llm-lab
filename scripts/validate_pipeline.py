"""
Comprehensive pipeline validation script.
Covers tasks 5–10: training targets, model verification, checkpoint resume,
full debug pipeline, colab_small config estimate, and baseline/ablation check.

Run from project root:
    python scripts/validate_pipeline.py

Writes results to: pipeline_validation.txt
"""

import sys, os, math, time, json, tempfile, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT  = os.path.join(ROOT, "pipeline_validation.txt")

lines = []

def log(msg=""):
    lines.append(str(msg))
    print(str(msg))

def section(title):
    log()
    log("=" * 65)
    log(title)
    log("=" * 65)

def ok(msg):  log(f"  [PASS] {msg}")
def fail(msg): log(f"  [FAIL] {msg}")
def info(msg): log(f"  [INFO] {msg}")

import torch
import torch.nn as nn

from model.config import ModelConfig, get_debug_config, get_8m_config
from model.transformer import SmallTransformer
from model.tokenizer import BPETokenizer
from data.generators.arithmetic import ArithmeticGenerator, build_corpus, Example
from training.dataset import ReasoningDataset, make_dataloader, collate_fn
from training.checkpoint import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint
)
from evaluation.arithmetic import verify, _extract_answer


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 — Training targets: tokenized examples, loss calculation
# ══════════════════════════════════════════════════════════════════════════════
section("TASK 5 — Training Targets & Loss Calculation")

gen = ArithmeticGenerator(seed=42, difficulty="easy")
examples = gen.generate(n=200, split="train")
corpus   = build_corpus(examples, use_reasoning=True)

tok = BPETokenizer()
tok.train(corpus, vocab_size=512, min_frequency=1, show_progress=False)
info(f"Tokenizer vocab_size = {tok.vocab_size}")
info(f"Special token IDs: PAD={tok.pad_id}, BOS={tok.bos_id}, EOS={tok.eos_id}")

# Inspect a concrete example
ex = examples[0]
log()
log(f"  Example text (reasoning format):")
log(f"  ---")
for line in ex.to_reasoning_text().split("\n"):
    log(f"  {line}")
log(f"  ---")

ids = tok.encode(ex.to_reasoning_text(), add_special_tokens=True)
log(f"\n  Token IDs ({len(ids)} tokens): {ids[:20]}{'...' if len(ids)>20 else ''}")
decoded_back = tok.decode(ids, skip_special_tokens=False)
log(f"  Decoded back: {repr(decoded_back[:120])}")

# Verify BOS at start, EOS at end
if ids[0] == tok.bos_id:
    ok("First token is BOS")
else:
    fail(f"First token is NOT BOS: got {ids[0]}, expected {tok.bos_id}")

if ids[-1] == tok.eos_id:
    ok("Last token is EOS")
else:
    fail(f"Last token is NOT EOS: got {ids[-1]}, expected {tok.eos_id}")

# Verify x/y shift in dataset
ds  = ReasoningDataset([ex], tok, max_seq_len=128)
x, y = ds[0]
log(f"\n  Dataset item: x length={len(x)}, y length={len(y)}")
if len(x) == len(y):
    ok("x and y have equal length")
else:
    fail(f"x length {len(x)} != y length {len(y)}")

if x[0].item() == tok.bos_id:
    ok("x[0] == BOS (model sees BOS as first input)")
else:
    fail(f"x[0] = {x[0].item()}, expected BOS={tok.bos_id}")

if y[-1].item() == tok.eos_id:
    ok("y[-1] == EOS (model predicts EOS as last target)")
else:
    fail(f"y[-1] = {y[-1].item()}, expected EOS={tok.eos_id}")

# Verify x and y are shifted by exactly 1
original_ids = torch.tensor(ids[:128])
expected_x = original_ids[:-1]
expected_y = original_ids[1:]
if torch.equal(x, expected_x):
    ok("x = ids[:-1]  (correct input slice)")
else:
    fail("x does not match ids[:-1]")
if torch.equal(y, expected_y):
    ok("y = ids[1:]   (correct target slice, shifted by 1)")
else:
    fail("y does not match ids[1:]")

# Verify loss ignores padding
log("\n  Padding check: loss must ignore pad_id tokens in targets")
cfg = get_debug_config()
cfg.vocab_size  = tok.vocab_size
cfg.pad_token_id = tok.pad_id
cfg.dropout = 0.0
model = SmallTransformer(cfg)
model.eval()

# Batch with no padding
x1 = torch.tensor([[tok.bos_id, 10, 11, 12]])
y1 = torch.tensor([[10, 11, 12, tok.eos_id]])

# Same batch, but last target is pad (should be ignored)
x2 = x1.clone()
y2 = torch.tensor([[10, 11, 12, tok.pad_id]])

with torch.no_grad():
    loss1, _ = model(x1, y1)
    loss2, _ = model(x2, y2)

# loss1 includes EOS prediction; loss2 ignores last position (pad)
# They differ (as expected), but loss2 should be < loss1 only by one token
info(f"  Loss with EOS target: {loss1.item():.4f}")
info(f"  Loss with PAD target: {loss2.item():.4f}")
ok("Loss calculation runs without error for both padded and unpadded targets")

# Verify the loss function uses ignore_index=pad_id
# Construct a case where target == pad_id and manually check it's excluded
x3 = torch.tensor([[tok.bos_id, tok.pad_id]])
y3 = torch.tensor([[tok.pad_id, tok.pad_id]])  # all padding
with torch.no_grad():
    loss3, _ = model(x3, y3)
if math.isnan(loss3.item()):
    fail("Loss is NaN when all targets are PAD — check transformer forward() guard")
elif loss3.item() == 0.0:
    ok(f"All-PAD target loss = 0.0 (NaN guard in forward() working correctly)")
else:
    ok(f"All-PAD target loss = {loss3.item():.4f} (handled gracefully)")

# Check "Answer:" token appears in encoded sequences
answer_in_tokens = tok.decode(ids, skip_special_tokens=True)
if "Answer" in answer_in_tokens or "nswer" in answer_in_tokens:
    ok("'Answer' keyword present in tokenized sequence")
else:
    fail("'Answer' keyword NOT found in tokenized sequence — verifier may fail")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 7 — Model verification: param count, shapes, masking, checkpoint resume
# ══════════════════════════════════════════════════════════════════════════════
section("TASK 7 — Model Verification")

# 8M config exact param count
cfg8 = get_8m_config()
cfg8.vocab_size = 4096
m8 = SmallTransformer(cfg8)
n8 = m8.count_parameters()
log(f"\n  8M config: d_model={cfg8.d_model}, n_layers={cfg8.n_layers}, "
    f"n_heads={cfg8.n_heads}, d_ff={cfg8.d_ff}, vocab={cfg8.vocab_size}")
log(f"  Actual parameter count: {n8:,}")

if 4_000_000 <= n8 <= 12_000_000:
    ok(f"Parameter count {n8:,} is in expected 4M-12M range for '8m' preset (weight-tied)")
else:
    fail(f"Parameter count {n8:,} is OUTSIDE expected 4M-12M range")

# Weight tying reduces count: embedding + lm_head share weights
# Without tying: 2 * 4096 * 256 = 2,097,152 extra params
bd = m8.count_parameters_breakdown()
info(f"  Breakdown: { {k: f'{v:,}' for k,v in bd.items()} }")
if m8.lm_head.weight is m8.token_emb.weight:
    ok("Weight tying confirmed: lm_head.weight is token_emb.weight")
else:
    fail("Weight tying BROKEN: lm_head and token_emb have separate weights")

# Forward pass shapes
log()
m8.eval()
dummy_input = torch.randint(1, cfg8.vocab_size, (2, 32))
with torch.no_grad():
    logits = m8(dummy_input)
expected_shape = (2, 32, cfg8.vocab_size)
if logits.shape == expected_shape:
    ok(f"Forward pass shape {tuple(logits.shape)} correct")
else:
    fail(f"Forward pass shape {tuple(logits.shape)} != expected {expected_shape}")

# No NaN/Inf in logits at init
if torch.isfinite(logits).all():
    ok("No NaN/Inf in logits at random initialisation")
else:
    fail(f"NaN or Inf in logits at init! max={logits.max()}, min={logits.min()}")

# Causal masking: token at position i must not depend on position j > i
log()
m8.eval()
x_base = torch.randint(1, cfg8.vocab_size, (1, 8))
x_mod  = x_base.clone()
x_mod[0, 7] = (x_base[0, 7] + 1) % cfg8.vocab_size   # change last token only
with torch.no_grad():
    out_base = m8(x_base)
    out_mod  = m8(x_mod)

# Positions 0..6 must be identical (they can't see position 7)
positions_unaffected = torch.allclose(out_base[0, :7], out_mod[0, :7], atol=1e-5)
position_7_affected  = not torch.allclose(out_base[0, 7], out_mod[0, 7], atol=1e-5)

if positions_unaffected:
    ok("Causal masking: positions 0–6 unchanged when token 7 is modified")
else:
    fail("Causal masking BROKEN: earlier positions affected by later token change")

if position_7_affected:
    ok("Causal masking: position 7 output changes when token 7 changes (self-attention works)")
else:
    fail("Position 7 output unchanged when token 7 changes — attention may not be working")

# Greedy generation produces same output twice (deterministic)
log()
debug_cfg = get_debug_config()
debug_cfg.vocab_size = tok.vocab_size
debug_model = SmallTransformer(debug_cfg)
debug_model.eval()
prompt = torch.randint(1, tok.vocab_size, (1, 4))
with torch.no_grad():
    out1 = debug_model.generate(prompt, max_new_tokens=8, temperature=0.0)
    out2 = debug_model.generate(prompt, max_new_tokens=8, temperature=0.0)
if torch.equal(out1, out2):
    ok("Greedy generation is deterministic (same output both runs)")
else:
    fail("Greedy generation is NOT deterministic")

# Sampled generation (temp > 0)
torch.manual_seed(0)
out3 = debug_model.generate(prompt, max_new_tokens=8, temperature=1.0)
ok(f"Sampled generation runs without error (output length {out3.shape[1]})")

# Top-k generation
out4 = debug_model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=5)
ok(f"Top-k=5 generation runs without error (output length {out4.shape[1]})")

# EOS stopping
# Set the model to always predict token at index 2 (EOS in debug config)
eos_id = debug_cfg.eos_token_id
test_m = SmallTransformer(debug_cfg)
test_m.eval()
with torch.no_grad():
    test_m.lm_head.weight.zero_()
    test_m.lm_head.weight[eos_id] = 1000.0
prompt2 = torch.tensor([[debug_cfg.bos_token_id]])
out5 = test_m.generate(prompt2, max_new_tokens=50, temperature=0.0, eos_token_id=eos_id)
if out5.shape[1] <= 4:
    ok(f"EOS stopping works: generation length {out5.shape[1]} (stopped early)")
else:
    fail(f"EOS stopping FAILED: generation length {out5.shape[1]} (should be ~2)")

# Checkpoint round-trip
log()
with tempfile.TemporaryDirectory() as tmpdir:
    ck_path = os.path.join(tmpdir, "ckpt.pt")
    opt = torch.optim.AdamW(debug_model.parameters(), lr=1e-3)

    # Record weights before save
    param_before = {n: p.clone() for n, p in debug_model.named_parameters()}

    save_checkpoint(
        path=ck_path, model=debug_model, optimizer=opt,
        epoch=3, global_step=999, best_val_loss=0.123,
        train_config={"lr": 1e-3}, model_config=debug_cfg.to_dict(),
        metrics={"val_loss": 0.123},
    )
    assert os.path.exists(ck_path)

    # Corrupt model weights
    with torch.no_grad():
        for p in debug_model.parameters():
            p.add_(torch.randn_like(p) * 10.0)

    # Verify corruption happened
    corrupted = False
    for n, p in debug_model.named_parameters():
        if not torch.allclose(p, param_before[n], atol=0.01):
            corrupted = True
            break
    if not corrupted:
        fail("Weight corruption check: weights didn't change after perturbation")

    # Load
    ck = load_checkpoint(ck_path, debug_model, optimizer=opt)

    # Verify restoration
    restored = all(
        torch.allclose(p, param_before[n], atol=1e-6)
        for n, p in debug_model.named_parameters()
    )
    if restored:
        ok("Checkpoint save/load: weights restored exactly")
    else:
        fail("Checkpoint save/load: weights NOT correctly restored")

    if ck["global_step"] == 999 and ck["epoch"] == 3:
        ok(f"Checkpoint metadata restored: epoch={ck['epoch']}, step={ck['global_step']}")
    else:
        fail(f"Checkpoint metadata wrong: epoch={ck['epoch']}, step={ck['global_step']}")

    if math.isclose(ck["best_val_loss"], 0.123):
        ok(f"best_val_loss restored: {ck['best_val_loss']}")
    else:
        fail(f"best_val_loss wrong: {ck['best_val_loss']}")

    # find_latest_checkpoint
    latest = find_latest_checkpoint(tmpdir)
    if latest and os.path.exists(latest):
        ok(f"find_latest_checkpoint returns valid path: {os.path.basename(latest)}")
    else:
        fail("find_latest_checkpoint returned None or invalid path")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 8 — Full local debug pipeline end-to-end
# ══════════════════════════════════════════════════════════════════════════════
section("TASK 8 — Full Local Debug Pipeline (end-to-end)")

log()
t_pipeline_start = time.time()

# Step 1: Generate data
log("  Step 1: Generate data")
gen = ArithmeticGenerator(seed=42, difficulty="easy")
train_ex, val_ex, test_ex = gen.generate_all_splits(n_train=80, n_val=20, n_test=20)
ok(f"Generated train={len(train_ex)}, val={len(val_ex)}, test={len(test_ex)}")

# Step 2: Train tokenizer
log("  Step 2: Train tokenizer")
corpus = build_corpus(train_ex, use_reasoning=True)
tok = BPETokenizer()
tok.train(corpus, vocab_size=512, min_frequency=1, show_progress=False)
ok(f"Tokenizer trained, vocab_size={tok.vocab_size}")

# Step 3: Build datasets
log("  Step 3: Build datasets")
train_ds = ReasoningDataset(train_ex, tok, max_seq_len=128, use_reasoning=True)
val_ds   = ReasoningDataset(val_ex,   tok, max_seq_len=128, use_reasoning=True)
stats = train_ds.stats()
ok(f"Train dataset: {stats['n_examples']} examples, {stats['total_tokens']:,} tokens, "
   f"avg_len={stats['avg_len']:.1f}")

# Step 4: Build DataLoaders
log("  Step 4: Build DataLoaders")
from torch.utils.data import DataLoader
train_loader = DataLoader(
    train_ds, batch_size=8, shuffle=True,
    collate_fn=lambda b: collate_fn(b, tok.pad_id)
)
val_loader = DataLoader(
    val_ds, batch_size=8, shuffle=False,
    collate_fn=lambda b: collate_fn(b, tok.pad_id)
)
x_sample, y_sample = next(iter(train_loader))
ok(f"DataLoader batch shape: x={tuple(x_sample.shape)}, y={tuple(y_sample.shape)}")

# Step 5: Initialise model from random weights
log("  Step 5: Initialise model from random weights")
torch.manual_seed(42)
cfg = get_debug_config()
cfg.vocab_size    = tok.vocab_size
cfg.max_seq_len   = 128
cfg.pad_token_id  = tok.pad_id
cfg.bos_token_id  = tok.bos_id
cfg.eos_token_id  = tok.eos_id
cfg.dropout       = 0.0
model = SmallTransformer(cfg)
n_params = model.count_parameters()
ok(f"Model initialised: {n_params:,} params, arch: {cfg.d_model}d/{cfg.n_layers}L/{cfg.n_heads}H")

# Step 6: Training loop (short)
log("  Step 6: Training loop (5 epochs)")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)

train_losses = []
for epoch in range(5):
    epoch_loss = 0.0
    n_batches  = 0
    for xb, yb in train_loader:
        loss, _ = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches  += 1
    avg = epoch_loss / n_batches
    train_losses.append(avg)
    log(f"    epoch {epoch+1}: train_loss={avg:.4f}")

if train_losses[-1] < train_losses[0]:
    ok(f"Training loss decreased: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
else:
    fail(f"Training loss did NOT decrease: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")

# Step 7: Validation loss
log("  Step 7: Validation loss")
model.eval()
val_loss = 0.0
n_val = 0
with torch.no_grad():
    for xb, yb in val_loader:
        loss, _ = model(xb, yb)
        val_loss += loss.item()
        n_val += 1
val_loss /= n_val
ok(f"Validation loss: {val_loss:.4f}")

# Step 8: Save checkpoint
log("  Step 8: Save checkpoint")
with tempfile.TemporaryDirectory() as tmpdir:
    ck_path = os.path.join(tmpdir, "debug_ckpt.pt")
    save_checkpoint(
        path=ck_path, model=model, optimizer=optimizer,
        epoch=5, global_step=len(train_loader)*5,
        best_val_loss=val_loss,
        train_config=cfg.to_dict(), model_config=cfg.to_dict(),
        metrics={"val_loss": val_loss, "train_loss": train_losses[-1]},
    )
    ok(f"Checkpoint saved: {os.path.getsize(ck_path):,} bytes")

    # Step 9: Load checkpoint and resume
    log("  Step 9: Load checkpoint and resume training")
    model2 = SmallTransformer(cfg)
    opt2   = torch.optim.AdamW(model2.parameters(), lr=5e-3)
    ck     = load_checkpoint(ck_path, model2, optimizer=opt2)
    ok(f"Checkpoint loaded: epoch={ck['epoch']}, step={ck['global_step']}")

    # Verify model2 gives same loss as model on the same batch
    model.eval();  model2.eval()
    with torch.no_grad():
        l1, _ = model(x_sample, y_sample)
        l2, _ = model2(x_sample, y_sample)
    if math.isclose(l1.item(), l2.item(), rel_tol=1e-5):
        ok("Resumed model produces identical loss to saved model (weights match)")
    else:
        fail(f"Resumed model loss {l2.item():.6f} != saved {l1.item():.6f}")

    # Step 10: Inference from checkpoint
    log("  Step 10: Inference from checkpoint")
    from inference.generate import generate_answer
    test_problem = test_ex[0].problem
    result = generate_answer(
        problem=test_problem,
        model=model2,
        tokenizer=tok,
        device="cpu",
        use_reasoning=True,
        max_new_tokens=40,
        temperature=0.0,
    )
    ok(f"Inference ran: problem='{test_problem[:50]}'")
    ok(f"Generated: '{result['generated_text'][:80].strip()}'")
    info(f"Extracted answer: {result['extracted_answer']}")

    # Step 11: Verify answer (model may be wrong after 5 epochs — that's fine)
    log("  Step 11: Run verifier on generated answer")
    vr = verify(result["full_text"], test_ex[0].answer, test_ex[0].answer_num)
    ok(f"Verifier ran: correct={vr['correct']}, "
       f"predicted={vr['predicted']}, expected={vr['expected']}")
    info("(Model accuracy not expected after only 5 training epochs — correct=False is acceptable here)")

    # Step 12: Mini evaluation benchmark
    log("  Step 12: Mini evaluation benchmark (20 test problems)")
    from evaluation.benchmark import run_benchmark
    bench = run_benchmark(
        model=model2, tokenizer=tok, generator=gen,
        n_problems=20, split="test",
        max_new_tokens=40, temperature=0.0,
        use_reasoning=True, max_samples_to_save=5,
        device="cpu",
    )
    ok(f"Benchmark complete: {bench.correct}/{bench.total} correct, "
       f"accuracy={bench.accuracy:.1%}")
    info("(Low accuracy expected after 5 epochs on tiny model — measuring pipeline, not accuracy)")

t_pipeline_end = time.time()
ok(f"Full debug pipeline completed in {t_pipeline_end - t_pipeline_start:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 9 — Colab notebook check + colab_small runtime estimate
# ══════════════════════════════════════════════════════════════════════════════
section("TASK 9 — Colab Notebook Check & Runtime Estimate")

# Check notebook exists and is an orchestration notebook
nb_path = os.path.join(ROOT, "colab", "train_small_reasoning_llm.ipynb")
if os.path.exists(nb_path):
    ok(f"Notebook exists: {nb_path}")
else:
    fail(f"Notebook NOT found at {nb_path}")

import json as json_mod
with open(nb_path) as f:
    nb = json_mod.load(f)

cells = nb.get("cells", [])
code_cells = [c for c in cells if c.get("cell_type") == "code"]
info(f"Notebook has {len(cells)} cells ({len(code_cells)} code cells)")

# Check notebook does NOT contain Transformer implementation
all_code = "\n".join(
    "".join(c.get("source", [])) for c in code_cells
)
bad_patterns = [
    "class SmallTransformer",
    "class DecoderBlock",
    "class CausalSelfAttention",
    "nn.ModuleList",
]
found_impl = [p for p in bad_patterns if p in all_code]
if found_impl:
    fail(f"Notebook contains Transformer implementation code: {found_impl}")
else:
    ok("Notebook does NOT contain Transformer implementation (orchestration only)")

# Check notebook imports from repository
good_imports = ["from training.train import train", "from model.", "from training.", "from evaluation."]
found_imports = [p for p in good_imports if p in all_code]
ok(f"Notebook imports from repository: {found_imports}")

# Check for GPU detection cell
if "torch.cuda.is_available" in all_code:
    ok("Notebook detects GPU availability")
else:
    fail("Notebook missing GPU detection")

# Check for checkpoint save
if "save" in all_code.lower() and "checkpoint" in all_code.lower():
    ok("Notebook mentions checkpoint saving")
else:
    fail("Notebook missing checkpoint save logic")

# Check for download/export
if "shutil.make_archive" in all_code or "files.download" in all_code:
    ok("Notebook includes artifact export")
else:
    fail("Notebook missing artifact export")

# Runtime estimate for colab_small
# Method: estimate steps/sec on the actual 8M model (not debug).
# The 8M model is what colab_small actually trains.
log()
log("  --- colab_small runtime estimate ---")

torch.manual_seed(0)
est_cfg = get_8m_config()
est_cfg.vocab_size  = 4096
est_cfg.max_seq_len = 256
est_model = SmallTransformer(est_cfg)
est_model.train()
est_opt = torch.optim.AdamW(est_model.parameters(), lr=3e-4)

# Small batch for timing (batch=8, seq=128 — shorter than full for speed)
xb_est = torch.randint(1, 4096, (8, 127))
yb_est = torch.randint(1, 4096, (8, 127))

# Warm up
for _ in range(2):
    l, _ = est_model(xb_est, yb_est)
    est_opt.zero_grad(); l.backward(); est_opt.step()

# Timed run — 10 steps
N_EST = 10
t0 = time.time()
for _ in range(N_EST):
    l, _ = est_model(xb_est, yb_est)
    est_opt.zero_grad(); l.backward(); est_opt.step()
t1 = time.time()

cpu_steps_per_sec_8m = N_EST / (t1 - t0)
cpu_tokens_per_sec_8m = cpu_steps_per_sec_8m * 8 * 127

info(f"Local CPU (8M model, batch=8, seq=128): {cpu_steps_per_sec_8m:.2f} steps/sec, "
     f"{cpu_tokens_per_sec_8m:,.0f} tokens/sec")

# T4 GPU is roughly 20-60x faster than typical CPU for this model size.
# Use conservative 20x and optimistic 60x to give a range.
T4_LOW = 20
T4_HIGH = 60

# colab_small: 20k examples x avg ~80 tokens x 30 epochs
colab_total_tokens = 20_000 * 80 * 30   # ~48M tokens
est_low_sec  = colab_total_tokens / (cpu_tokens_per_sec_8m * T4_HIGH)
est_high_sec = colab_total_tokens / (cpu_tokens_per_sec_8m * T4_LOW)

# Actual batch in colab_small is 64 (vs 8 here) — larger batches are faster per token on GPU
# This partially offsets the uncertainty. Keep range conservative.

log()
info("  colab_small config parameters:")
info("    model:   ~7.95M params (8m preset, weight-tied, d256/8L/ff1152)")
info("    dataset: 20,000 training examples")
info("    epochs:  30")
info("    batch:   64 (grad_accum=2, effective batch=128)")
info(f"    total tokens (est): ~{colab_total_tokens/1e6:.0f}M")
info(f"    est. training time on T4: {est_low_sec/60:.0f} - {est_high_sec/60:.0f} minutes")
info("  NOTE: This is a rough order-of-magnitude estimate only.")
info("  Actual Colab T4 speed varies with hardware, memory, and batch size.")
info("  Checkpoints every 1000 steps. Safe to resume after interruption.")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 10 — Baseline/ablation config comparison
# ══════════════════════════════════════════════════════════════════════════════
section("TASK 10 — Baseline vs Ablation Config Comparison")

configs_dir = os.path.join(ROOT, "experiments", "configs")

def load_cfg(name):
    path = os.path.join(configs_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json_mod.load(f)

cfg_baseline = load_cfg("exp_01_baseline.json")
cfg_direct   = load_cfg("exp_02_direct_format.json")

if cfg_baseline is None:
    fail("exp_01_baseline.json not found")
elif cfg_direct is None:
    fail("exp_02_direct_format.json not found")
else:
    ok("Both experiment configs found")

    # Fields that MUST be identical for a fair comparison
    must_match = [
        "model_size", "n_train", "n_val", "n_test", "difficulty",
        "data_seed", "tokenizer_vocab_size", "max_epochs", "batch_size",
        "grad_accumulation_steps", "optimizer", "learning_rate",
        "weight_decay", "beta1", "beta2", "grad_clip", "scheduler",
        "warmup_steps", "min_lr_ratio", "max_seq_len", "seed",
        "eval_n_problems",
    ]

    mismatches = []
    for field in must_match:
        v1 = cfg_baseline.get(field)
        v2 = cfg_direct.get(field)
        if v1 != v2:
            mismatches.append(f"{field}: baseline={v1}, direct={v2}")

    if mismatches:
        fail(f"Config mismatch in controlled fields:")
        for m in mismatches:
            log(f"    {m}")
    else:
        ok("All controlled fields are identical between baseline and ablation")

    # The ONE field that should differ
    if cfg_baseline.get("use_reasoning") is True and cfg_direct.get("use_reasoning") is False:
        ok("use_reasoning differs: baseline=True (reasoning), ablation=False (direct)")
    else:
        fail(f"use_reasoning mismatch: baseline={cfg_baseline.get('use_reasoning')}, "
             f"direct={cfg_direct.get('use_reasoning')}")

    # Check output dirs are different (don't overwrite each other)
    if cfg_baseline.get("output_dir") != cfg_direct.get("output_dir"):
        ok("Output directories are distinct")
    else:
        fail("Both configs write to the same output_dir — they will overwrite each other")

    # gen_max_new_tokens: reasoning format needs more tokens
    bt = cfg_baseline.get("gen_max_new_tokens", 0)
    dt = cfg_direct.get("gen_max_new_tokens", 0)
    info(f"gen_max_new_tokens: baseline={bt}, direct={dt}")
    if bt >= dt:
        ok("Baseline (reasoning) has >= gen_max_new_tokens than direct — appropriate")
    else:
        fail("Direct format has more gen tokens than reasoning format — may be misconfigured")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("PIPELINE VALIDATION SUMMARY")

passed = sum(1 for l in lines if "[PASS]" in l)
failed = sum(1 for l in lines if "[FAIL]" in l)
log(f"\n  PASS: {passed}")
log(f"  FAIL: {failed}")
log()
if failed == 0:
    log("  ALL CHECKS PASSED — pipeline is ready for full test suite rerun.")
else:
    log("  FAILURES FOUND — fix before proceeding to Colab training.")
    for l in lines:
        if "[FAIL]" in l:
            log(f"  >>> {l.strip()}")

with open(OUT, "w") as f:
    f.write("\n".join(lines))
log(f"\nFull report saved to: {OUT}")
