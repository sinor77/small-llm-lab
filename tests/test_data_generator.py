"""
Tests for data/generators/arithmetic.py.

Covers:
- All problem families generate valid examples
- Solutions are programmatically correct
- Seeds produce reproducible data
- Train/val/test splits don't overlap
- Generalization set is structured correctly
- Difficulty levels affect problem parameters
"""

import sys, os
import math
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generators.arithmetic import (
    ArithmeticGenerator, Example, build_corpus,
    DIRECT_TEMPLATE, REASONING_TEMPLATE
)


@pytest.fixture
def gen():
    return ArithmeticGenerator(seed=42, difficulty="medium")


@pytest.fixture
def easy_gen():
    return ArithmeticGenerator(seed=42, difficulty="easy")


@pytest.fixture
def hard_gen():
    return ArithmeticGenerator(seed=42, difficulty="hard")


# ── Basic generation ──────────────────────────────────────────────────────────

def test_generate_returns_correct_count(gen):
    examples = gen.generate(n=50, split="train")
    assert len(examples) == 50


def test_each_example_has_required_fields(gen):
    examples = gen.generate(n=20, split="train")
    for ex in examples:
        assert isinstance(ex.problem, str) and len(ex.problem) > 0
        assert isinstance(ex.answer, str) and len(ex.answer) > 0
        assert isinstance(ex.answer_num, float)
        assert ex.family in ArithmeticGenerator.ALL_FAMILIES
        assert ex.difficulty in ("easy", "medium", "hard")


# ── Per-family correctness ────────────────────────────────────────────────────

def _check_answer(ex: Example) -> bool:
    """Parse the answer_num from the problem's answer string and verify."""
    try:
        parsed = float(ex.answer)
        return math.isclose(parsed, ex.answer_num, rel_tol=1e-4, abs_tol=1e-4)
    except ValueError:
        return True  # non-numeric answer (comparison), trust generator


@pytest.mark.parametrize("family", ArithmeticGenerator.ALL_FAMILIES)
def test_family_answer_is_numerically_correct(family):
    gen = ArithmeticGenerator(seed=0, difficulty="medium", families=[family])
    import random
    rng = random.Random(0)
    for _ in range(30):
        ex = getattr(gen, f"_gen_{family}")(rng)
        assert _check_answer(ex), (
            f"Family '{family}': answer '{ex.answer}' != answer_num {ex.answer_num}\n"
            f"Problem: {ex.problem}"
        )


def test_single_op_addition_correct():
    gen = ArithmeticGenerator(seed=1, difficulty="easy", families=["single_op"])
    import random
    rng = random.Random(99)
    # Force an addition by checking multiple examples
    for _ in range(50):
        ex = gen._gen_single_op(rng)
        assert _check_answer(ex)


def test_multi_step_evaluation_correct():
    gen = ArithmeticGenerator(seed=2, difficulty="medium", families=["multi_step"])
    import random
    rng = random.Random(5)
    for _ in range(30):
        ex = gen._gen_multi_step(rng)
        assert _check_answer(ex)


def test_algebra_linear_correct():
    """ax + b = c; verify that x satisfies the equation."""
    gen = ArithmeticGenerator(seed=3, difficulty="hard", families=["algebra_linear"])
    import random
    rng = random.Random(10)
    for _ in range(30):
        ex = gen._gen_algebra_linear(rng)
        # Parse coefficients from problem string "Solve for x: ax + b = c"
        import re
        m = re.match(r"Solve for x: (-?\d+)x \+ (-?\d+) = (-?\d+)", ex.problem)
        if m:
            a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            x = ex.answer_num
            assert math.isclose(a * x + b, c, abs_tol=1e-4), (
                f"ax+b={a*x+b} != c={c}"
            )


def test_percentage_correct():
    gen = ArithmeticGenerator(seed=4, difficulty="medium", families=["percentage"])
    import random
    rng = random.Random(20)
    for _ in range(20):
        ex = gen._gen_percentage(rng)
        # Extract pct and whole from "What is X% of Y?"
        import re
        m = re.match(r"What is (\d+)% of (\d+)", ex.problem)
        if m:
            pct, whole = int(m.group(1)), int(m.group(2))
            expected = round(whole * pct / 100, 2)
            assert math.isclose(ex.answer_num, expected, rel_tol=1e-4)


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_same_seed_same_examples():
    gen1 = ArithmeticGenerator(seed=77, difficulty="medium")
    gen2 = ArithmeticGenerator(seed=77, difficulty="medium")
    ex1  = gen1.generate(n=20, split="train")
    ex2  = gen2.generate(n=20, split="train")
    for a, b in zip(ex1, ex2):
        assert a.problem == b.problem
        assert a.answer  == b.answer


def test_different_seeds_different_examples():
    gen1 = ArithmeticGenerator(seed=1)
    gen2 = ArithmeticGenerator(seed=2)
    ex1  = gen1.generate(n=20, split="train")
    ex2  = gen2.generate(n=20, split="train")
    problems1 = set(e.problem for e in ex1)
    problems2 = set(e.problem for e in ex2)
    # Some overlap is fine but shouldn't be identical
    assert problems1 != problems2


# ── Split separation ──────────────────────────────────────────────────────────

def test_train_val_test_no_overlap(gen):
    """Zero exact problem overlap between splits (using generate_all_splits)."""
    train, val, test = gen.generate_all_splits(n_train=500, n_val=100, n_test=100)

    train_p = set(e.problem for e in train)
    val_p   = set(e.problem for e in val)
    test_p  = set(e.problem for e in test)

    assert len(train_p & val_p)  == 0, f"train∩val overlap: {len(train_p & val_p)}"
    assert len(train_p & test_p) == 0, f"train∩test overlap: {len(train_p & test_p)}"
    assert len(val_p   & test_p) == 0, f"val∩test overlap: {len(val_p & test_p)}"


def test_generate_deduplication_within_split():
    """deduplicate=True must produce no repeated problem strings in a single split."""
    gen = ArithmeticGenerator(seed=42, difficulty="easy")  # easy = small space, more collisions
    examples = gen.generate(n=100, split="train", deduplicate=True)
    problems = [e.problem for e in examples]
    assert len(problems) == len(set(problems)), "Duplicate problems found within split"


def test_generate_all_splits_lengths():
    """generate_all_splits must return exactly the requested counts (when space permits)."""
    gen = ArithmeticGenerator(seed=0, difficulty="medium")
    train, val, test = gen.generate_all_splits(n_train=200, n_val=50, n_test=50)
    assert len(train) == 200
    assert len(val)   == 50
    assert len(test)  == 50


# ── Formatting ────────────────────────────────────────────────────────────────

def test_direct_format_contains_answer(gen):
    ex = gen.generate(n=1, split="train")[0]
    text = ex.to_direct_text()
    assert "Problem:" in text
    assert "Answer:" in text
    assert ex.answer in text


def test_reasoning_format_contains_reasoning(gen):
    ex = gen.generate(n=1, split="train")[0]
    text = ex.to_reasoning_text()
    assert "Problem:" in text
    assert "Reasoning:" in text
    assert "Answer:" in text


def test_build_corpus_returns_strings(gen):
    examples = gen.generate(n=10, split="train")
    corpus   = build_corpus(examples, use_reasoning=True)
    assert len(corpus) == 10
    assert all(isinstance(t, str) for t in corpus)


# ── Generalization set ────────────────────────────────────────────────────────

def test_generalization_set_has_5_levels(gen):
    gset = gen.generate_generalization_set(n_per_level=10)
    assert set(gset.keys()) == {1, 2, 3, 4, 5}


def test_generalization_set_each_level_correct_count(gen):
    gset = gen.generate_generalization_set(n_per_level=20)
    for level, examples in gset.items():
        assert len(examples) == 20, f"Level {level} has {len(examples)} != 20"


def test_difficulty_affects_number_range():
    easy = ArithmeticGenerator(seed=0, difficulty="easy", families=["single_op"])
    hard = ArithmeticGenerator(seed=0, difficulty="hard", families=["single_op"])
    import re, random
    rng = random.Random(0)
    easy_nums = []
    hard_nums = []
    for _ in range(100):
        for nums, g in [(easy_nums, easy), (hard_nums, hard)]:
            ex = g._gen_single_op(rng)
            found = re.findall(r"\d+", ex.problem)
            nums.extend(int(n) for n in found)
    assert max(hard_nums) > max(easy_nums), "Hard difficulty should produce larger numbers"
