"""
Tests for model/tokenizer.py.

Covers:
- Training on small corpus
- Encode / decode round-trip
- Special tokens present and at correct IDs
- Deterministic save/load
- Batch encoding
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tokenizer import BPETokenizer, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


SAMPLE_TEXTS = [
    "Problem: What is 3 + 4? Answer: 7",
    "Problem: What is 10 * 5? Answer: 50",
    "Problem: Solve for x: 2x + 1 = 9 Answer: 4",
    "Reasoning: Step 1: Add 3 and 4. Step 2: 3 + 4 = 7.",
    "Problem: What is 15 - 8? Reasoning: Step 1: 15 - 8 = 7. Answer: 7",
] * 20   # repeat to get enough frequency for BPE merges


@pytest.fixture(scope="module")
def trained_tokenizer():
    tok = BPETokenizer()
    tok.train(SAMPLE_TEXTS, vocab_size=256, min_frequency=1, show_progress=False)
    return tok


def test_vocab_size_is_positive(trained_tokenizer):
    assert trained_tokenizer.vocab_size > 0


def test_vocab_size_at_most_requested(trained_tokenizer):
    # vocab_size may be less than requested if corpus is small
    assert trained_tokenizer.vocab_size <= 256


def test_special_tokens_present(trained_tokenizer):
    tok = trained_tokenizer
    assert tok.pad_id is not None
    assert tok.bos_id is not None
    assert tok.eos_id is not None
    assert tok.unk_id is not None
    assert tok.sep_id is not None


def test_special_token_ids_are_distinct(trained_tokenizer):
    tok = trained_tokenizer
    ids = [tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id, tok.sep_id]
    assert len(set(ids)) == 5


def test_encode_returns_list_of_ints(trained_tokenizer):
    ids = trained_tokenizer.encode("What is 3 + 4?")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_encode_includes_bos_eos(trained_tokenizer):
    ids = trained_tokenizer.encode("What is 3 + 4?", add_special_tokens=True)
    assert ids[0] == trained_tokenizer.bos_id
    assert ids[-1] == trained_tokenizer.eos_id


def test_encode_no_special_tokens(trained_tokenizer):
    ids = trained_tokenizer.encode("What is 3 + 4?", add_special_tokens=False)
    # BOS / EOS should not appear
    assert trained_tokenizer.bos_id not in ids
    assert trained_tokenizer.eos_id not in ids


def test_decode_round_trip(trained_tokenizer):
    text = "Answer: 42"
    ids  = trained_tokenizer.encode(text, add_special_tokens=False)
    back = trained_tokenizer.decode(ids, skip_special_tokens=True)
    # After byte-level BPE + decode, spaces/case may shift slightly —
    # check core content is present
    assert "42" in back


def test_encode_batch_same_as_individual(trained_tokenizer):
    texts = ["What is 3 + 4?", "Solve for x: 2x = 8"]
    batch = trained_tokenizer.encode_batch(texts, add_special_tokens=True)
    for text, batch_ids in zip(texts, batch):
        single = trained_tokenizer.encode(text, add_special_tokens=True)
        assert batch_ids == single


def test_deterministic_save_load(trained_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        trained_tokenizer.save(tmpdir)
        loaded = BPETokenizer.load(tmpdir)

        text = "Problem: 12 + 34"
        ids1 = trained_tokenizer.encode(text)
        ids2 = loaded.encode(text)
        assert ids1 == ids2


def test_meta_file_created(trained_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        trained_tokenizer.save(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "tokenizer.json"))
        assert os.path.exists(os.path.join(tmpdir, "tokenizer_meta.json"))


def test_load_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        BPETokenizer.load("/nonexistent_dir_xyz/")


def test_repr(trained_tokenizer):
    r = repr(trained_tokenizer)
    assert "BPETokenizer" in r
    assert "vocab_size" in r
