"""
ArithmeticGenerator — procedural synthetic arithmetic problem generator.

Design principles:
- Every problem has a programmatically calculated exact answer.
- Problems are generated deterministically given a seed.
- Difficulty is configurable: easy / medium / hard.
- Train, validation, and test splits use non-overlapping seeds so there
  is no leakage between splits.
- The test generator can produce problems that are structurally different
  from the exact training examples (longer chains, new combinations),
  supporting genuine generalization tests.

Problem families supported:
    1. single_op         — single arithmetic operation  (e.g., 37 + 58)
    2. multi_step        — chained operations           (e.g., (3+4)*5-2)
    3. comparison        — which is larger?
    4. percentage        — what is X% of Y?
    5. ratio             — ratio and proportion
    6. algebra_linear    — solve for x in ax + b = c
    7. word_problem      — simple word-problem wrapper around arithmetic
    8. number_sequence   — arithmetic / geometric sequence, find next term

Two output formats are supported (configurable per dataset):
    direct:    Problem → Answer  (no reasoning steps)
    reasoning: Problem → Reasoning steps → Answer

The exact format strings are centralised here so they can be changed
without touching training or evaluation code.
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from fractions import Fraction

# ── Format templates (single source of truth) ────────────────────────────────

DIRECT_TEMPLATE = "Problem: {problem}\nAnswer: {answer}"

REASONING_TEMPLATE = (
    "Problem: {problem}\n"
    "Reasoning:\n{reasoning}\n"
    "Answer: {answer}"
)

ANSWER_PREFIX = "Answer:"    # used by verifier to extract answer from generated text
REASONING_PREFIX = "Reasoning:"
PROBLEM_PREFIX = "Problem:"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Example:
    """A single training / evaluation example."""
    problem:     str                      # the question text
    answer:      str                      # canonical answer as string
    answer_num:  float                    # numeric value (for verifier)
    reasoning:   str = ""                 # step-by-step reasoning (may be empty)
    family:      str = ""                 # problem family name
    difficulty:  str = "medium"           # easy / medium / hard
    metadata:    dict = field(default_factory=dict)  # extra provenance info

    def to_direct_text(self) -> str:
        """Format as Problem → Answer (no reasoning)."""
        return DIRECT_TEMPLATE.format(problem=self.problem, answer=self.answer)

    def to_reasoning_text(self) -> str:
        """Format as Problem → Reasoning → Answer."""
        return REASONING_TEMPLATE.format(
            problem=self.problem,
            reasoning=self.reasoning if self.reasoning else "Direct computation.",
            answer=self.answer,
        )

    def to_dict(self) -> dict:
        return {
            "problem":    self.problem,
            "answer":     self.answer,
            "answer_num": self.answer_num,
            "reasoning":  self.reasoning,
            "family":     self.family,
            "difficulty": self.difficulty,
            "metadata":   self.metadata,
        }


# ── Difficulty parameter table ────────────────────────────────────────────────

DIFFICULTY_PARAMS = {
    "easy": {
        "int_range":    (1, 20),
        "chain_len":    (1, 2),
        "seq_len":      (3, 5),
        "pct_vals":     [10, 20, 25, 50, 75],
        "coef_range":   (1, 5),
    },
    "medium": {
        "int_range":    (1, 100),
        "chain_len":    (2, 4),
        "seq_len":      (4, 7),
        "pct_vals":     list(range(5, 100, 5)),
        "coef_range":   (1, 20),
    },
    "hard": {
        "int_range":    (1, 999),
        "chain_len":    (4, 6),
        "seq_len":      (5, 9),
        "pct_vals":     list(range(1, 100)),
        "coef_range":   (1, 50),
    },
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_num(x: float) -> str:
    """Format a number cleanly: integer if whole, 2dp otherwise."""
    if x == int(x):
        return str(int(x))
    return f"{x:.2f}"


def _fmt_frac(x: float) -> str:
    """Try to represent as a fraction for cleaner reasoning text."""
    try:
        f = Fraction(x).limit_denominator(1000)
        if f.denominator == 1:
            return str(f.numerator)
        return f"{f.numerator}/{f.denominator}"
    except Exception:
        return _fmt_num(x)


# ── Individual problem generators ────────────────────────────────────────────

class ArithmeticGenerator:
    """
    Generate synthetic arithmetic problems with exact solutions.

    Args:
        seed:       base random seed (train/val/test splits use offsets)
        difficulty: "easy" | "medium" | "hard"
        families:   which problem families to include (None = all)
    """

    ALL_FAMILIES = [
        "single_op",
        "multi_step",
        "comparison",
        "percentage",
        "ratio",
        "algebra_linear",
        "word_problem",
        "number_sequence",
    ]

    def __init__(
        self,
        seed: int = 42,
        difficulty: str = "medium",
        families: Optional[List[str]] = None,
    ):
        self.seed       = seed
        self.difficulty = difficulty
        self.families   = families or self.ALL_FAMILIES
        self.params     = DIFFICULTY_PARAMS[difficulty]

        # Validate
        unknown = set(self.families) - set(self.ALL_FAMILIES)
        if unknown:
            raise ValueError(f"Unknown families: {unknown}")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        n: int,
        split: str = "train",
        deduplicate: bool = True,
        exclude: Optional[set] = None,
    ) -> List[Example]:
        """
        Generate n UNIQUE examples for a given split.

        Split seed offsets give each split an independent RNG stream:
            train: seed + 0
            val:   seed + 1_000_000
            test:  seed + 2_000_000

        Args:
            n:           number of examples to generate
            split:       "train" | "val" | "test"
            deduplicate: if True (default), discard duplicate problem strings
                         within this split. The generator will draw extra
                         samples as needed until n unique problems are found.
            exclude:     optional set of problem strings to also exclude
                         (used for cross-split deduplication)

        Returns:
            List of Example objects with unique problem strings.
        """
        offsets = {"train": 0, "val": 1_000_000, "test": 2_000_000}
        if split not in offsets:
            raise ValueError(f"split must be train/val/test, got '{split}'")

        rng = random.Random(self.seed + offsets[split])
        examples: List[Example] = []
        seen: set = set(exclude) if exclude else set()
        max_attempts = n * 20   # safety ceiling: abort if too many collisions

        attempts = 0
        while len(examples) < n and attempts < max_attempts:
            family = rng.choice(self.families)
            gen_fn = getattr(self, f"_gen_{family}")
            ex = gen_fn(rng)
            ex.difficulty = self.difficulty
            attempts += 1

            if deduplicate:
                if ex.problem in seen:
                    continue
                seen.add(ex.problem)

            examples.append(ex)

        if len(examples) < n:
            import warnings
            warnings.warn(
                f"Could only generate {len(examples)}/{n} unique problems for "
                f"split='{split}' after {max_attempts} attempts. "
                "The problem space may be exhausted for this difficulty/family combination. "
                "Consider reducing n or increasing difficulty.",
                UserWarning,
                stacklevel=2,
            )

        return examples

    def generate_all_splits(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
    ) -> tuple:
        """
        Generate train, val, and test splits with GUARANTEED zero cross-split overlap.

        Generates train first, then val excluding all train problems, then test
        excluding all train+val problems. This is the correct way to use the
        generator when exact non-overlap is required.

        Returns:
            (train_examples, val_examples, test_examples)
        """
        train_ex = self.generate(n=n_train, split="train", deduplicate=True)
        train_problems = set(e.problem for e in train_ex)

        val_ex = self.generate(n=n_val, split="val", deduplicate=True,
                               exclude=train_problems)
        val_problems = set(e.problem for e in val_ex)

        test_ex = self.generate(n=n_test, split="test", deduplicate=True,
                                exclude=train_problems | val_problems)

        return train_ex, val_ex, test_ex

    def generate_generalization_set(
        self,
        n_per_level: int = 100,
        exclude: Optional[set] = None,
    ) -> dict:
        """
        Generate a structured generalization test set with 5 difficulty levels.

        Args:
            n_per_level: number of examples per level
            exclude:     set of problem strings to exclude (pass the union of
                         train+val+test problem strings for strict separation)

        Level 1: new values, familiar templates (medium difficulty)
        Level 2: new numerical ranges (hard difficulty)
        Level 3: mixed families, medium difficulty
        Level 4: longer chains (hard, multi_step + algebra only)
        Level 5: hard difficulty, full family mix
        """
        global_exclude: set = set(exclude) if exclude else set()

        result = {}
        levels = {
            1: ("medium", self.families, 1),
            2: ("hard",   self.families, 2),
            3: ("medium", self.families, 3),
            4: ("hard",   ["multi_step", "algebra_linear"], 4),
            5: ("hard",   self.families, 5),
        }
        for level, (diff, fams, offset_mult) in levels.items():
            gen = ArithmeticGenerator(
                seed=self.seed + 9_000_000 + offset_mult * 100_000,
                difficulty=diff,
                families=fams,
            )
            rng = random.Random(self.seed + 9_000_000 + offset_mult * 100_000)
            examples = []
            # Each level excludes global set PLUS problems already in this level
            level_seen: set = set(global_exclude)
            max_attempts = n_per_level * 20
            attempts = 0
            while len(examples) < n_per_level and attempts < max_attempts:
                family = rng.choice(fams)
                ex = getattr(gen, f"_gen_{family}")(rng)
                ex.difficulty = diff
                ex.metadata["gen_level"] = level
                attempts += 1
                if ex.problem not in level_seen:
                    level_seen.add(ex.problem)
                    examples.append(ex)
            result[level] = examples
        return result

    # ── Single operation ──────────────────────────────────────────────────────

    def _gen_single_op(self, rng: random.Random) -> Example:
        lo, hi = self.params["int_range"]
        ops = ["+", "-", "*"]
        if hi >= 10:
            ops.append("//")   # integer division

        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        op = rng.choice(ops)

        if op == "+" :
            answer_num = a + b
            problem = f"What is {a} + {b}?"
            reasoning = (f"Step 1: Add {a} and {b}.\n"
                         f"Step 2: {a} + {b} = {answer_num}.")
        elif op == "-":
            a, b = max(a, b), min(a, b)   # keep positive
            answer_num = a - b
            problem = f"What is {a} - {b}?"
            reasoning = (f"Step 1: Subtract {b} from {a}.\n"
                         f"Step 2: {a} - {b} = {answer_num}.")
        elif op == "*":
            answer_num = a * b
            problem = f"What is {a} * {b}?"
            reasoning = (f"Step 1: Multiply {a} by {b}.\n"
                         f"Step 2: {a} * {b} = {answer_num}.")
        else:   # integer division
            if b == 0:
                b = 1
            answer_num = a // b
            problem = f"What is {a} divided by {b}, rounded down to a whole number?"
            reasoning = (f"Step 1: Divide {a} by {b} using integer division.\n"
                         f"Step 2: {a} // {b} = {answer_num}.")

        return Example(
            problem=problem,
            answer=_fmt_num(answer_num),
            answer_num=float(answer_num),
            reasoning=reasoning,
            family="single_op",
        )

    # ── Multi-step chained arithmetic ─────────────────────────────────────────

    def _gen_multi_step(self, rng: random.Random) -> Example:
        lo, hi = self.params["int_range"]
        min_len, max_len = self.params["chain_len"]
        chain_len = rng.randint(min_len, max_len)

        ops    = ["+", "-", "*"]
        nums   = [rng.randint(lo, max(lo, hi // 2)) for _ in range(chain_len + 1)]
        chosen = [rng.choice(ops) for _ in range(chain_len)]

        # Build expression and evaluate step-by-step
        expr      = str(nums[0])
        steps     = []
        result    = nums[0]
        for i, (op, num) in enumerate(zip(chosen, nums[1:])):
            prev   = result
            if op == "+":
                result += num
            elif op == "-":
                result -= num
            else:
                result *= num
            expr += f" {op} {num}"
            steps.append(f"Step {i+1}: {prev} {op} {num} = {result}")

        problem   = f"Calculate: {expr}"
        reasoning = "\n".join(steps)

        return Example(
            problem=problem,
            answer=_fmt_num(result),
            answer_num=float(result),
            reasoning=reasoning,
            family="multi_step",
        )

    # ── Comparison ────────────────────────────────────────────────────────────

    def _gen_comparison(self, rng: random.Random) -> Example:
        lo, hi = self.params["int_range"]
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        while a == b:
            b = rng.randint(lo, hi)

        # Sometimes compare expressions instead of bare numbers
        use_expr = rng.random() < 0.4 and hi >= 10
        if use_expr:
            c = rng.randint(lo, hi // 2)
            d = rng.randint(lo, hi // 2)
            val_a = a + c
            val_b = b * d if b * d < 10 * hi else b + d
            expr_a = f"{a} + {c}"
            expr_b = f"{b} * {d}" if b * d < 10 * hi else f"{b} + {d}"
        else:
            val_a, val_b = a, b
            expr_a, expr_b = str(a), str(b)

        if val_a > val_b:
            answer_text = expr_a
            answer_num  = float(val_a)
        elif val_b > val_a:
            answer_text = expr_b
            answer_num  = float(val_b)
        else:
            answer_text = "equal"
            answer_num  = float(val_a)

        problem = f"Which is greater: {expr_a} or {expr_b}?"
        reasoning = (
            f"Step 1: Evaluate {expr_a} = {val_a}.\n"
            f"Step 2: Evaluate {expr_b} = {val_b}.\n"
            f"Step 3: Compare {val_a} and {val_b}: "
            + ("the first is greater." if val_a > val_b
               else "the second is greater." if val_b > val_a
               else "they are equal.")
        )

        return Example(
            problem=problem,
            answer=answer_text,
            answer_num=answer_num,
            reasoning=reasoning,
            family="comparison",
        )

    # ── Percentage ────────────────────────────────────────────────────────────

    def _gen_percentage(self, rng: random.Random) -> Example:
        lo, hi  = self.params["int_range"]
        pct     = rng.choice(self.params["pct_vals"])
        whole   = rng.randint(max(lo, 10), hi)
        result  = round(whole * pct / 100, 2)

        problem = f"What is {pct}% of {whole}?"
        reasoning = (
            f"Step 1: Convert {pct}% to decimal: {pct} / 100 = {pct/100}.\n"
            f"Step 2: Multiply {pct/100} × {whole} = {result}."
        )

        return Example(
            problem=problem,
            answer=_fmt_num(result),
            answer_num=float(result),
            reasoning=reasoning,
            family="percentage",
        )

    # ── Ratio ─────────────────────────────────────────────────────────────────

    def _gen_ratio(self, rng: random.Random) -> Example:
        lo, hi  = self.params["int_range"]
        # a:b ratio, total n, find share for a
        a = rng.randint(1, 10)
        b = rng.randint(1, 10)
        n = rng.randint(max(lo, a + b), min(hi, (a + b) * 10))
        # Make n divisible by (a+b) for clean answers
        total_parts = a + b
        n = (n // total_parts) * total_parts
        if n == 0:
            n = total_parts

        share_a = n * a // total_parts

        problem = (
            f"Two quantities are in the ratio {a}:{b}. "
            f"If the total is {n}, what is the first quantity?"
        )
        reasoning = (
            f"Step 1: Total parts = {a} + {b} = {total_parts}.\n"
            f"Step 2: Value of each part = {n} / {total_parts} = {n // total_parts}.\n"
            f"Step 3: First quantity = {a} × {n // total_parts} = {share_a}."
        )

        return Example(
            problem=problem,
            answer=str(share_a),
            answer_num=float(share_a),
            reasoning=reasoning,
            family="ratio",
        )

    # ── Linear algebra  (ax + b = c, solve for x) ────────────────────────────

    def _gen_algebra_linear(self, rng: random.Random) -> Example:
        lo, hi  = self.params["coef_range"]
        a = rng.randint(1, hi)           # coefficient (non-zero)
        x = rng.randint(-hi, hi)         # true answer
        b = rng.randint(-hi, hi)
        c = a * x + b                    # so ax + b = c exactly

        problem = f"Solve for x: {a}x + {b} = {c}"
        reasoning = (
            f"Step 1: Subtract {b} from both sides: {a}x = {c} - {b} = {c - b}.\n"
            f"Step 2: Divide both sides by {a}: x = {c - b} / {a} = {x}."
        )

        return Example(
            problem=problem,
            answer=str(x),
            answer_num=float(x),
            reasoning=reasoning,
            family="algebra_linear",
        )

    # ── Word problem ──────────────────────────────────────────────────────────

    _WORD_TEMPLATES = [
        (
            "Alice has {a} apples. She buys {b} more. How many apples does she have now?",
            lambda a, b: a + b,
            lambda a, b, r: (
                f"Step 1: Start with {a} apples.\n"
                f"Step 2: Add {b} more: {a} + {b} = {r}."
            ),
        ),
        (
            "A box contains {a} items. {b} items are removed. How many remain?",
            lambda a, b: max(a, b) - min(a, b),
            lambda a, b, r: (
                f"Step 1: Start with {max(a,b)} items.\n"
                f"Step 2: Remove {min(a,b)}: {max(a,b)} - {min(a,b)} = {r}."
            ),
        ),
        (
            "There are {a} rows of seats with {b} seats each. How many seats in total?",
            lambda a, b: a * b,
            lambda a, b, r: (
                f"Step 1: Multiply rows by seats per row.\n"
                f"Step 2: {a} × {b} = {r}."
            ),
        ),
        (
            "{a} students are split equally into {b} groups. How many students per group?",
            lambda a, b: a // b if b > 0 else 0,
            lambda a, b, r: (
                f"Step 1: Divide total students by number of groups.\n"
                f"Step 2: {a} ÷ {b} = {r}."
            ),
        ),
        (
            "A store sells items at ${a} each. A customer buys {b} items. "
            "What is the total cost?",
            lambda a, b: a * b,
            lambda a, b, r: (
                f"Step 1: Multiply price by quantity.\n"
                f"Step 2: ${a} × {b} = ${r}."
            ),
        ),
    ]

    def _gen_word_problem(self, rng: random.Random) -> Example:
        lo, hi    = self.params["int_range"]
        template, fn, reason_fn = rng.choice(self._WORD_TEMPLATES)
        a = rng.randint(max(lo, 2), hi)
        b = rng.randint(max(lo, 1), max(a, hi))
        result = fn(a, b)

        problem   = template.format(a=a, b=b)
        reasoning = reason_fn(a, b, result)

        return Example(
            problem=problem,
            answer=str(result),
            answer_num=float(result),
            reasoning=reasoning,
            family="word_problem",
        )

    # ── Number sequence ───────────────────────────────────────────────────────

    def _gen_number_sequence(self, rng: random.Random) -> Example:
        lo, hi   = self.params["int_range"]
        min_len, max_len = self.params["seq_len"]
        seq_len  = rng.randint(min_len, max_len)

        # Choose: arithmetic (constant diff) or geometric (constant ratio)
        kind  = rng.choice(["arithmetic", "geometric"])

        if kind == "arithmetic":
            start = rng.randint(lo, hi)
            diff  = rng.randint(1, max(2, hi // 10))
            seq   = [start + i * diff for i in range(seq_len + 1)]
            next_val = seq[-1]
            seq_str  = ", ".join(str(s) for s in seq[:-1])
            problem  = f"What is the next number in the sequence: {seq_str}, ...?"
            reasoning = (
                f"Step 1: Find the common difference: {seq[1]} - {seq[0]} = {diff}.\n"
                f"Step 2: Add the difference to the last term: {seq[-2]} + {diff} = {next_val}."
            )
        else:
            start = rng.randint(max(lo, 1), max(2, hi // 4))
            ratio = rng.randint(2, 4)
            seq   = [start * (ratio ** i) for i in range(seq_len + 1)]
            next_val = seq[-1]
            seq_str  = ", ".join(str(s) for s in seq[:-1])
            problem  = f"What is the next number in the sequence: {seq_str}, ...?"
            reasoning = (
                f"Step 1: Find the common ratio: {seq[1]} / {seq[0]} = {ratio}.\n"
                f"Step 2: Multiply the last term by the ratio: {seq[-2]} × {ratio} = {next_val}."
            )

        return Example(
            problem=problem,
            answer=str(next_val),
            answer_num=float(next_val),
            reasoning=reasoning,
            family="number_sequence",
        )


# ── Corpus builder ────────────────────────────────────────────────────────────

def build_corpus(examples: List[Example], use_reasoning: bool = True) -> List[str]:
    """
    Convert a list of Examples into text strings for training / tokenizer fitting.

    Args:
        examples:      list of Example objects
        use_reasoning: if True, use the full reasoning format
    Returns:
        list of formatted strings
    """
    texts = []
    for ex in examples:
        if use_reasoning:
            texts.append(ex.to_reasoning_text())
        else:
            texts.append(ex.to_direct_text())
    return texts
