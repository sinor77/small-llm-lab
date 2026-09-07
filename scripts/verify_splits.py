"""
Exhaustive split separation verification for exp_01_baseline datasets.

Generates the exact same datasets the experiment will use (n_train=20000,
n_val=2000, n_test=1000, difficulty=medium, seed=42) and checks:

  1. Exact problem-text overlap (train/val, train/test, val/test)
  2. Structural template overlap with documentation
  3. Generalization level definitions verified against actual examples
  4. Answer overlap reported but not treated as leakage

Writes results to: split_verification_report.txt
"""

import sys, os, re, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "split_verification_report.txt")

from data.generators.arithmetic import ArithmeticGenerator, Example

lines = []
def log(msg=""): lines.append(str(msg))
def section(t):
    log(); log("=" * 65); log(t); log("=" * 65)
def ok(m):   log(f"  [PASS] {m}")
def fail(m): log(f"  [FAIL] {m}")
def info(m): log(f"  [INFO] {m}")

# ── Parameters matching exp_01_baseline exactly ───────────────────────────────
SEED       = 42
DIFFICULTY = "medium"
N_TRAIN    = 20_000
N_VAL      = 2_000
N_TEST     = 1_000

section("GENERATING EXP_01 DATASETS")
info(f"seed={SEED}, difficulty={DIFFICULTY}")
info(f"n_train={N_TRAIN:,}, n_val={N_VAL:,}, n_test={N_TEST:,}")
info("Using generate_all_splits() — guaranteed zero cross-split overlap")

gen = ArithmeticGenerator(seed=SEED, difficulty=DIFFICULTY)
train_ex, val_ex, test_ex = gen.generate_all_splits(
    n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST
)
ok(f"Generated {len(train_ex):,} train, {len(val_ex):,} val, {len(test_ex):,} test")

# ── 1. EXACT PROBLEM-TEXT OVERLAP ────────────────────────────────────────────
section("1. EXACT PROBLEM-TEXT OVERLAP")

train_problems = set(e.problem for e in train_ex)
val_problems   = set(e.problem for e in val_ex)
test_problems  = set(e.problem for e in test_ex)

tv = train_problems & val_problems
tt = train_problems & test_problems
vt = val_problems   & test_problems

log(f"\n  train unique problems : {len(train_problems):,}")
log(f"  val   unique problems : {len(val_problems):,}")
log(f"  test  unique problems : {len(test_problems):,}")
log()
log(f"  train ∩ val  overlap  : {len(tv)}")
log(f"  train ∩ test overlap  : {len(tt)}")
log(f"  val   ∩ test overlap  : {len(vt)}")

if len(tv) == 0:
    ok("train ∩ val  = 0  (no exact text leakage)")
else:
    fail(f"train ∩ val  = {len(tv)}  LEAKAGE DETECTED")
    for p in list(tv)[:5]:
        log(f"    example: {p}")

if len(tt) == 0:
    ok("train ∩ test = 0  (no exact text leakage)")
else:
    fail(f"train ∩ test = {len(tt)}  LEAKAGE DETECTED")
    for p in list(tt)[:5]:
        log(f"    example: {p}")

if len(vt) == 0:
    ok("val   ∩ test = 0  (no exact text leakage)")
else:
    fail(f"val   ∩ test = {len(vt)}  LEAKAGE DETECTED")

# ── 2. ANSWER OVERLAP (expected — not leakage) ────────────────────────────────
section("2. ANSWER OVERLAP (numeric coincidence — not leakage)")

train_answers = collections.Counter(e.answer for e in train_ex)
val_answers   = set(e.answer for e in val_ex)
test_answers  = set(e.answer for e in test_ex)

shared_tv = set(train_answers.keys()) & val_answers
shared_tt = set(train_answers.keys()) & test_answers
info(f"Unique answers in train: {len(train_answers)}")
info(f"Train answers also in val:  {len(shared_tv)} / {len(val_answers)}")
info(f"Train answers also in test: {len(shared_tt)} / {len(test_answers)}")
info("Answer overlap is EXPECTED for arithmetic (many problems share the same number).")
info("This does NOT constitute data leakage.")

top_answers = train_answers.most_common(5)
info(f"Most frequent train answers: {top_answers}")

# ── 3. STRUCTURAL TEMPLATE OVERLAP ────────────────────────────────────────────
section("3. STRUCTURAL TEMPLATE OVERLAP")

log("""
  DEFINITION: Two problems share a structural template if they belong to
  the same problem family and use the same arithmetic operation pattern,
  but with different numerical values.

  Example of SAME template (NOT duplicates):
      "What is 37 + 58?"   and   "What is 12 + 91?"
      Both are single_op/addition with different numbers.

  Example of DIFFERENT templates:
      "What is 37 + 58?"   and   "What is 9 * 4?"
      Different operation, different template.

  Structural overlap is EXPECTED and INTENTIONAL:
  The model must learn to apply the same procedure to unseen numbers.
  We test generalization (new values, familiar templates) at Level 1.

  Different random seeds do NOT prove structural generalization.
  The test set shares operation templates with the training set.
  Generalization is measured empirically by accuracy on the test set.
""")

# Count family distribution across splits
def family_counts(examples):
    c = collections.Counter(e.family for e in examples)
    return dict(sorted(c.items()))

train_fam = family_counts(train_ex)
val_fam   = family_counts(val_ex)
test_fam  = family_counts(test_ex)

log("  Family distribution:")
log(f"  {'Family':<25} {'Train':>8} {'Val':>6} {'Test':>6}")
log(f"  {'-'*47}")
for fam in sorted(set(list(train_fam) + list(val_fam) + list(test_fam))):
    log(f"  {fam:<25} {train_fam.get(fam,0):>8,} {val_fam.get(fam,0):>6,} {test_fam.get(fam,0):>6,}")

# Verify proportions are roughly equal (all families present in all splits)
all_fams = set(train_fam.keys())
val_missing  = all_fams - set(val_fam.keys())
test_missing = all_fams - set(test_fam.keys())
if not val_missing:
    ok("All 8 families present in validation split")
else:
    fail(f"Missing families in val: {val_missing}")
if not test_missing:
    ok("All 8 families present in test split")
else:
    fail(f"Missing families in test: {test_missing}")

# ── 4. GENERALIZATION LEVEL DEFINITIONS ──────────────────────────────────────
section("4. GENERALIZATION LEVEL DEFINITIONS AND VERIFICATION")

log("""
  The 5-level generalization benchmark uses a completely separate seed space
  (seed + 9_000_000 + level*100_000) so no example overlaps with train/val/test.

  Level definitions:
  ─────────────────────────────────────────────────────────────────────────────
  Level 1 — New values, familiar templates (same difficulty: medium)
    What this tests: Can the model apply a known operation to unseen numbers?
    Example: trained on "37 + 58", tested on "64 + 27"
    Generator: same families, same difficulty, different seed
    This is the minimum bar. A model that only memorizes specific problems fails.

  Level 2 — New numerical ranges (difficulty: hard)
    What this tests: Does the model generalize to larger numbers it hasn't seen?
    Example: trained on 1-100 range, tested on 1-999 range
    Generator: same families, hard difficulty
    A model that learned the procedure should still work; one that
    memorized number patterns will fail.

  Level 3 — Mixed families, medium difficulty
    What this tests: Can the model switch between different problem types
    within a single evaluation pass?
    Generator: all families randomly mixed, medium difficulty
    This is essentially the same as the standard test but with a
    different seed — confirms consistency.

  Level 4 — Longer reasoning chains (hard, limited to multi_step + algebra)
    What this tests: Can the model maintain coherent reasoning over more steps?
    Generator: only multi_step and algebra_linear, hard difficulty
    These families produce the longest reasoning chains at hard difficulty.

  Level 5 — Hard difficulty, full family mix
    What this tests: Combined stress test of harder numbers and mixed families.
    Generator: all families, hard difficulty
    This is the hardest evaluation tier.
  ─────────────────────────────────────────────────────────────────────────────

  WHAT THIS BENCHMARK DOES NOT TEST:
  - Genuinely compositional reasoning (combining two distinct problem types
    in a single problem, e.g., "solve the ratio then find the percentage")
  - Out-of-distribution operators (e.g., exponentiation if not in training)
  - Problems with adversarial formatting
  These are future work items.
""")

# Verify generalization set has no overlap with train/test
# Pass all known problems as exclusion set so the gen set is guaranteed clean
all_known = train_problems | val_problems | test_problems
gset = gen.generate_generalization_set(n_per_level=200, exclude=all_known)
gen_problems = set()
for level, exs in gset.items():
    gen_problems.update(e.problem for e in exs)

gen_train_overlap = gen_problems & train_problems
gen_test_overlap  = gen_problems & test_problems

log(f"  Generalization set size: {len(gen_problems)} unique problems")
if len(gen_train_overlap) == 0:
    ok("Generalization set ∩ train = 0 (no exact overlap)")
else:
    fail(f"Generalization set ∩ train = {len(gen_train_overlap)}")
    for p in list(gen_train_overlap)[:3]:
        log(f"    {p}")

if len(gen_test_overlap) == 0:
    ok("Generalization set ∩ test  = 0 (no exact overlap)")
else:
    fail(f"Generalization set ∩ test = {len(gen_test_overlap)}")

# Show a few real examples per level
log()
log("  Sample problems per generalization level:")
for level in sorted(gset.keys()):
    exs = gset[level]
    log(f"\n  --- Level {level} ---")
    for ex in exs[:3]:
        log(f"    [{ex.family}/{ex.difficulty}] {ex.problem}  => {ex.answer}")

# ── 5. DIFFICULTY DISTRIBUTION WITHIN EACH SPLIT ─────────────────────────────
section("5. DIFFICULTY UNIFORMITY CHECK")
info("All splits use 'medium' difficulty. Confirming no accidental mixing.")
for split_name, exs in [("train", train_ex), ("val", val_ex), ("test", test_ex)]:
    diffs = set(e.difficulty for e in exs)
    if diffs == {"medium"}:
        ok(f"{split_name}: all examples are 'medium' difficulty")
    else:
        fail(f"{split_name}: mixed difficulties found: {diffs}")

# ── 6. REPRODUCIBILITY CHECK ─────────────────────────────────────────────────
section("6. REPRODUCIBILITY")
gen2 = ArithmeticGenerator(seed=SEED, difficulty=DIFFICULTY)
train2, _, _ = gen2.generate_all_splits(n_train=100, n_val=20, n_test=10)
match = all(a.problem == b.problem and a.answer == b.answer
            for a, b in zip(train_ex[:100], train2))
if match:
    ok("Same seed produces identical problems (deterministic)")
else:
    fail("Same seed produces different problems — non-deterministic!")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
section("SPLIT VERIFICATION SUMMARY")
passed = sum(1 for l in lines if "[PASS]" in l)
failed = sum(1 for l in lines if "[FAIL]" in l)
log(f"\n  PASS: {passed}")
log(f"  FAIL: {failed}")
if failed == 0:
    log("\n  All split checks PASSED.")
    log("  Exact train/val/test problem-text overlap is ZERO.")
    log("  Generalization set does not overlap with any training or test data.")
else:
    log("\n  FAILURES FOUND. Fix before proceeding.")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Report written to: {OUT}")
for l in lines:
    print(l)
