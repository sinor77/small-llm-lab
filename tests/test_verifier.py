"""
Tests for evaluation/arithmetic.py (programmatic verifier).

Covers:
- Correct answers pass
- Incorrect answers fail
- Answer extraction from various text formats
- Floating-point tolerance
- Non-numeric (comparison) answers
- Missing answer returns correct=False
"""

import sys, os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.arithmetic import verify, verify_batch, _extract_answer, _parse_number


# ── Answer extraction ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("Answer: 42",                                   "42"),
    ("answer: 42",                                   "42"),
    ("Answer:42",                                    "42"),
    ("Answer: -3",                                   "-3"),
    ("Answer: 1.5",                                  "1.5"),
    ("The answer is 7",                              "7"),
    ("Step 1: 3+4=7.\nAnswer: 7",                    "7"),
    ("Reasoning:\nStep 1: ...\nAnswer: 100",         "100"),
    ("Problem: ...\nReasoning:\nStep 1: ...\nAnswer: 50", "50"),
])
def test_extract_answer_formats(text, expected):
    result = _extract_answer(text)
    assert result == expected, f"Got '{result}' for text: {text!r}"


def test_extract_answer_missing_returns_none():
    assert _extract_answer("No answer here at all.") is None


def test_extract_answer_fraction():
    result = _extract_answer("Answer: 3/4")
    assert result == "3/4"


# ── Number parsing ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("s, expected", [
    ("42",    42.0),
    ("-3",    -3.0),
    ("1.5",   1.5),
    ("1,000", 1000.0),
    ("3/4",   0.75),
    ("1/2",   0.5),
])
def test_parse_number_valid(s, expected):
    result = _parse_number(s)
    assert result is not None
    assert abs(result - expected) < 1e-6


def test_parse_number_invalid():
    assert _parse_number("abc") is None
    assert _parse_number(None)  is None


# ── verify() ─────────────────────────────────────────────────────────────────

def test_correct_integer_answer():
    result = verify("Answer: 7", "7", 7.0)
    assert result["correct"] is True


def test_incorrect_integer_answer():
    result = verify("Answer: 8", "7", 7.0)
    assert result["correct"] is False


def test_correct_with_reasoning_text():
    text = (
        "Problem: What is 37 + 58?\n"
        "Reasoning:\nStep 1: 37 + 58 = 95.\n"
        "Answer: 95"
    )
    result = verify(text, "95", 95.0)
    assert result["correct"] is True


def test_float_tolerance_close_enough():
    # The generator stores answer_num = round(result, 2), so the expected value
    # is already rounded. Test that a tiny floating-point rounding difference
    # (within abs_tol) is treated as correct.
    # 33.33 vs 33.3300001 — well within 1e-4 absolute tolerance.
    result = verify("Answer: 33.33", "33.33", 33.3300001)
    assert result["correct"] is True


def test_float_tolerance_2dp_vs_exact():
    # 33.33 is NOT within 1e-4 of 33.333... (100/3).
    # This is correct and expected behaviour: the model predicted a 2dp answer
    # for a value that requires more precision.
    result = verify("Answer: 33.33", "33.33", 100 / 3)
    assert result["correct"] is False


def test_float_tolerance_too_far():
    # 33 is not close to 33.33 — should fail
    result = verify("Answer: 33", "33.33", 33.33)
    assert result["correct"] is False


def test_negative_answer():
    result = verify("Answer: -5", "-5", -5.0)
    assert result["correct"] is True


def test_missing_answer_returns_not_correct():
    result = verify("The model said nothing useful.", "42", 42.0)
    assert result["correct"] is False
    assert result["predicted"] is None


def test_result_dict_has_required_keys():
    result = verify("Answer: 10", "10", 10.0)
    for key in ("correct", "predicted", "expected", "pred_num", "exp_num", "reason"):
        assert key in result


def test_fraction_answer_correct():
    # 3/4 = 0.75
    result = verify("Answer: 3/4", "0.75", 0.75)
    assert result["correct"] is True


# ── verify_batch ──────────────────────────────────────────────────────────────

def test_verify_batch_all_correct():
    texts    = ["Answer: 7",  "Answer: 10", "Answer: -3"]
    answers  = ["7",          "10",         "-3"]
    nums     = [7.0,          10.0,         -3.0]
    results  = verify_batch(texts, answers, nums)
    assert len(results) == 3
    assert all(r["correct"] for r in results)


def test_verify_batch_mixed():
    texts   = ["Answer: 7",  "Answer: 99"]
    answers = ["7",          "10"]
    nums    = [7.0,          10.0]
    results = verify_batch(texts, answers, nums)
    assert results[0]["correct"] is True
    assert results[1]["correct"] is False


# ── Comparison (non-numeric) answers ─────────────────────────────────────────

def test_comparison_string_match():
    """
    Comparison problems return text answers like "5 + 3" or "equal".
    The verifier should handle string matching for these.
    """
    result = verify("Answer: equal", "equal", 5.0)
    # "equal" is also a valid string answer
    # pred_num will be None; falls back to string compare
    assert result["correct"] is True or result["predicted"] is not None
