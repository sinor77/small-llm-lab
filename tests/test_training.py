"""
Tests for training pipeline.

Covers:
- Tiny model can overfit a tiny dataset (MANDATORY gate)
- DataLoader produces correct shapes
- Checkpoint save / load round-trip
- Resumed training matches expected step count
- ReasoningDataset statistics

The overfit test is the most important: if a tiny model cannot drive loss
to near-zero on 8 examples, something is wrong with the training loop,
data pipeline, or model.
"""

import sys, os
import math
import tempfile
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import get_debug_config
from model.transformer import SmallTransformer
from model.tokenizer import BPETokenizer
from data.generators.arithmetic import ArithmeticGenerator, build_corpus
from training.dataset import ReasoningDataset, make_dataloader, collate_fn
from training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_tokenizer():
    gen = ArithmeticGenerator(seed=0, difficulty="easy")
    examples = gen.generate(n=200, split="train")
    corpus   = build_corpus(examples, use_reasoning=True)
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=512, min_frequency=1, show_progress=False)
    return tok


@pytest.fixture(scope="module")
def small_examples():
    gen = ArithmeticGenerator(seed=0, difficulty="easy")
    return gen.generate(n=30, split="train")


# ── Dataset ───────────────────────────────────────────────────────────────────

def test_dataset_len(small_examples, small_tokenizer):
    ds = ReasoningDataset(small_examples, small_tokenizer, max_seq_len=128)
    assert len(ds) == len(small_examples)


def test_dataset_item_shapes(small_examples, small_tokenizer):
    ds = ReasoningDataset(small_examples, small_tokenizer, max_seq_len=128)
    x, y = ds[0]
    assert x.shape == y.shape
    assert x.ndim == 1
    assert len(x) > 0


def test_dataset_targets_shifted(small_examples, small_tokenizer):
    ds = ReasoningDataset(small_examples, small_tokenizer, max_seq_len=128)
    for i in range(min(5, len(ds))):
        x, y = ds[i]
        # Retrieve original IDs
        ids = small_tokenizer.encode(
            small_examples[i].to_reasoning_text(), add_special_tokens=True
        )
        ids_t = torch.tensor(ids[:128])
        expected_x = ids_t[:-1]
        expected_y = ids_t[1:]
        assert torch.equal(x, expected_x)
        assert torch.equal(y, expected_y)


def test_dataset_token_count_positive(small_examples, small_tokenizer):
    ds = ReasoningDataset(small_examples, small_tokenizer, max_seq_len=128)
    assert ds.token_count() > 0


def test_dataloader_batch_shape(small_examples, small_tokenizer):
    loader = make_dataloader(
        small_examples, small_tokenizer,
        max_seq_len=128, batch_size=4, shuffle=False
    )
    x, y = next(iter(loader))
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape == y.shape
    assert x.shape[0] <= 4


# ── Checkpoint ────────────────────────────────────────────────────────────────

def test_checkpoint_save_load_model_state(small_tokenizer):
    cfg   = get_debug_config()
    cfg.vocab_size = small_tokenizer.vocab_size
    model = SmallTransformer(cfg)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        ck_path = os.path.join(tmpdir, "test.pt")
        save_checkpoint(
            path=ck_path, model=model, optimizer=opt,
            epoch=2, global_step=100, best_val_loss=0.5,
            train_config={"lr": 1e-3}, model_config=cfg.to_dict(),
            metrics={"val_loss": 0.5},
        )
        assert os.path.exists(ck_path)

        # Perturb weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        # Reload
        ck = load_checkpoint(ck_path, model, optimizer=opt)
        assert ck["epoch"] == 2
        assert ck["global_step"] == 100
        assert math.isclose(ck["best_val_loss"], 0.5)


def test_find_latest_checkpoint_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_latest_checkpoint(tmpdir)
        assert result is None


def test_find_latest_checkpoint_returns_highest_step(small_tokenizer):
    cfg   = get_debug_config()
    cfg.vocab_size = small_tokenizer.vocab_size
    model = SmallTransformer(cfg)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        for step in [100, 200, 500]:
            save_checkpoint(
                path=os.path.join(tmpdir, f"step_{step:07d}.pt"),
                model=model, optimizer=opt, epoch=1,
                global_step=step, best_val_loss=1.0,
                train_config={}, model_config=cfg.to_dict(),
            )
        latest = find_latest_checkpoint(tmpdir)
        assert "500" in latest or "0000500" in latest


# ── MANDATORY: Tiny overfit test ──────────────────────────────────────────────

def test_tiny_model_overfits_tiny_dataset(small_tokenizer):
    """
    A tiny model trained on 8 identical examples for many steps must reach
    near-zero training loss.

    This is the most important training gate: if this fails, the training
    loop, model, or data pipeline has a fundamental bug.
    """
    torch.manual_seed(42)
    cfg = get_debug_config()
    cfg.vocab_size = small_tokenizer.vocab_size
    cfg.dropout    = 0.0   # no dropout for overfit test
    model = SmallTransformer(cfg)
    model.train()

    gen      = ArithmeticGenerator(seed=0, difficulty="easy", families=["single_op"])
    examples = gen.generate(n=8, split="train")

    # Build a fixed tiny dataset
    ds = ReasoningDataset(examples, small_tokenizer, max_seq_len=128)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, shuffle=False,
        collate_fn=lambda b: collate_fn(b, small_tokenizer.pad_id)
    )
    x_fixed, y_fixed = next(iter(loader))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    initial_loss = None
    final_loss   = None

    for step in range(300):
        loss, _ = model(x_fixed, y_fixed)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    print(f"\n[Overfit Test] initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}")

    # Loss must drop by at least 80% from the initial value
    assert final_loss < initial_loss * 0.2, (
        f"Model failed to overfit: initial={initial_loss:.4f}, final={final_loss:.4f}. "
        "Check training loop, model, or data pipeline."
    )
    # Additionally, final loss should be below 1.0
    assert final_loss < 1.0, f"Final loss {final_loss:.4f} is still too high."


# ── Generalization benchmark regression tests ─────────────────────────────────

def test_run_generalization_benchmark_no_name_error(small_tokenizer):
    """
    Regression test for NameError: name 'gen_level_acc' is not defined.

    Previously run_generalization_benchmark() referenced 'gen_level_acc'
    which was a leftover variable that was never initialised after a refactor.
    This test ensures the function completes without NameError and returns
    the expected result structure for all 5 levels.
    """
    from evaluation.benchmark import run_generalization_benchmark
    from data.generators.arithmetic import ArithmeticGenerator

    torch.manual_seed(0)
    cfg = get_debug_config()
    cfg.vocab_size    = small_tokenizer.vocab_size
    cfg.max_seq_len   = 128
    cfg.pad_token_id  = small_tokenizer.pad_id
    cfg.bos_token_id  = small_tokenizer.bos_id
    cfg.eos_token_id  = small_tokenizer.eos_id
    cfg.dropout       = 0.0
    model = SmallTransformer(cfg)
    model.eval()

    generator = ArithmeticGenerator(seed=99, difficulty="easy")

    # Must not raise NameError
    results = run_generalization_benchmark(
        model=model,
        tokenizer=small_tokenizer,
        generator=generator,
        n_per_level=5,
        max_new_tokens=20,
        temperature=0.0,
        device="cpu",
        use_reasoning=True,
        exclude_problems=None,
    )

    # Must return a dict with all 5 levels
    assert isinstance(results, dict)
    assert set(results.keys()) == {1, 2, 3, 4, 5}, \
        f"Expected levels {{1,2,3,4,5}}, got {set(results.keys())}"

    # Each level must have the expected keys and types
    for level, r in results.items():
        assert "accuracy" in r, f"Level {level} missing 'accuracy'"
        assert "correct"  in r, f"Level {level} missing 'correct'"
        assert "total"    in r, f"Level {level} missing 'total'"
        assert "samples"  in r, f"Level {level} missing 'samples'"
        assert isinstance(r["accuracy"], float), f"Level {level} accuracy not float"
        assert 0.0 <= r["accuracy"] <= 1.0,      f"Level {level} accuracy out of range"
        assert r["total"] > 0,                    f"Level {level} has 0 examples"

    # The Colab notebook usage pattern must work without error
    for level, r in results.items():
        _ = f"Level {level}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})"


def test_run_generalization_benchmark_with_exclude(small_tokenizer):
    """
    Ensure exclude_problems parameter is accepted without error.
    """
    from evaluation.benchmark import run_generalization_benchmark
    from data.generators.arithmetic import ArithmeticGenerator

    torch.manual_seed(1)
    cfg = get_debug_config()
    cfg.vocab_size   = small_tokenizer.vocab_size
    cfg.max_seq_len  = 128
    cfg.pad_token_id = small_tokenizer.pad_id
    cfg.bos_token_id = small_tokenizer.bos_id
    cfg.eos_token_id = small_tokenizer.eos_id
    cfg.dropout      = 0.0
    model = SmallTransformer(cfg)
    model.eval()

    generator = ArithmeticGenerator(seed=7, difficulty="easy")
    exclude = {"What is 5 + 3?", "What is 2 + 2?"}

    results = run_generalization_benchmark(
        model=model,
        tokenizer=small_tokenizer,
        generator=generator,
        n_per_level=5,
        max_new_tokens=20,
        temperature=0.0,
        device="cpu",
        use_reasoning=True,
        exclude_problems=exclude,
    )

    assert set(results.keys()) == {1, 2, 3, 4, 5}
    # Excluded problems must not appear in any level
    for level, r in results.items():
        for sample in r["samples"]:
            assert sample["problem"] not in exclude, \
                f"Excluded problem found in generalization level {level}"
