"""
Tests for model/transformer.py and model/attention.py.

Covers:
- Forward pass tensor shapes
- Causal masking correctness
- Loss computation
- Parameter counting
- Generation (greedy)
- Weight tying
"""

import pytest
import torch
import math

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig, get_debug_config
from model.transformer import SmallTransformer, DecoderBlock, FeedForward
from model.attention import CausalSelfAttention


@pytest.fixture
def cfg():
    return get_debug_config()


@pytest.fixture
def model(cfg):
    m = SmallTransformer(cfg)
    m.eval()
    return m


# ── Config ────────────────────────────────────────────────────────────────────

def test_config_d_model_divisible_by_n_heads():
    cfg = get_debug_config()
    assert cfg.d_model % cfg.n_heads == 0


def test_config_bad_head_count_raises():
    with pytest.raises(AssertionError):
        ModelConfig(d_model=64, n_heads=3)  # 64 % 3 != 0


def test_config_param_estimate_positive(cfg):
    est = cfg.param_estimate()
    assert est > 0


def test_config_to_from_dict(cfg):
    d    = cfg.to_dict()
    cfg2 = ModelConfig.from_dict(d)
    assert cfg2.d_model == cfg.d_model
    assert cfg2.n_layers == cfg.n_layers


# ── Attention ─────────────────────────────────────────────────────────────────

def test_attention_output_shape(cfg):
    attn = CausalSelfAttention(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        max_seq_len=cfg.max_seq_len,
        dropout=0.0,
    )
    x = torch.randn(2, 16, cfg.d_model)
    out = attn(x)
    assert out.shape == (2, 16, cfg.d_model)


def test_causal_mask_is_lower_triangular(cfg):
    attn = CausalSelfAttention(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        max_seq_len=cfg.max_seq_len,
    )
    mask = attn.causal_mask[0, 0]  # (max_seq_len, max_seq_len)
    T = mask.shape[0]
    for i in range(T):
        for j in range(T):
            expected = (j <= i)
            assert mask[i, j].item() == expected, f"Mask wrong at ({i},{j})"


def test_attention_future_positions_different(cfg):
    """
    Token at position 0 should not depend on token at position 1 (causal).
    We verify by checking that zeroing out a future token changes its own
    position's output but NOT earlier positions.
    """
    attn = CausalSelfAttention(
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        max_seq_len=cfg.max_seq_len, dropout=0.0,
    )
    attn.eval()
    x1 = torch.randn(1, 4, cfg.d_model)
    x2 = x1.clone()
    x2[0, 3] = 0.0   # zero out last token

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # First token output must be identical (does not see future)
    assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)
    # Last token output must differ (it sees itself)
    assert not torch.allclose(out1[0, 3], out2[0, 3], atol=1e-6)


# ── Transformer forward ───────────────────────────────────────────────────────

def test_forward_logit_shape(model, cfg):
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_forward_with_targets_returns_loss(model, cfg):
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    loss, logits = model(x, y)
    assert loss.ndim == 0            # scalar
    assert loss.item() > 0
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_forward_all_pad_targets_no_nan(model, cfg):
    """All-pad targets must not produce NaN (NaN guard in forward())."""
    x = torch.randint(1, cfg.vocab_size, (2, 8))
    y = torch.full((2, 8), cfg.pad_token_id, dtype=torch.long)
    loss, _ = model(x, y)
    assert not torch.isnan(loss), "Loss is NaN when all targets are PAD"
    assert loss.item() == 0.0 or loss.item() >= 0.0  # 0.0 from guard, or normal


def test_forward_seq_too_long_raises(model, cfg):
    x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len + 1))
    with pytest.raises(AssertionError):
        model(x)


def test_loss_decreases_on_single_batch(cfg):
    """Model should be able to reduce loss on a single repeated batch."""
    torch.manual_seed(0)
    model = SmallTransformer(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(1, cfg.vocab_size, (4, 20))
    y = torch.randint(1, cfg.vocab_size, (4, 20))

    losses = []
    for _ in range(30):
        loss, _ = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "Loss did not decrease"


def test_weight_tying(model):
    """Token embedding and LM head should share the same weight tensor."""
    assert model.token_emb.weight is model.lm_head.weight


def test_parameter_count_positive(model):
    n = model.count_parameters()
    assert n > 0
    # Debug model should be well below 1M
    assert n < 1_000_000


def test_parameter_count_breakdown(model):
    bd = model.count_parameters_breakdown()
    assert "total" in bd
    assert bd["total"] > 0


def test_8m_config_exact_parameter_count():
    """
    The 8m preset must produce exactly 7,949,824 trainable parameters.
    This test will fail if the architecture is accidentally changed,
    catching any unintended model size drift.
    """
    from model.config import get_8m_config
    cfg = get_8m_config()
    cfg.vocab_size = 4096
    m = SmallTransformer(cfg)
    n = m.count_parameters()
    assert n == 7_949_824, (
        f"8m preset parameter count changed: got {n:,}, expected 7,949,824. "
        "Update the preset docstring and README if this was intentional."
    )
    # Weight tying: total == trainable (no additional buffers for shared weight)
    total = sum(p.numel() for p in m.parameters())
    assert total == n, "Total params != trainable params — weight tying may be broken"


# ── Generation ────────────────────────────────────────────────────────────────

def test_generate_output_longer_than_prompt(model, cfg):
    prompt = torch.randint(1, cfg.vocab_size, (1, 5))
    out = model.generate(prompt, max_new_tokens=10)
    assert out.shape[1] > 5


def test_generate_greedy_deterministic(model, cfg):
    prompt = torch.randint(1, cfg.vocab_size, (1, 5))
    out1 = model.generate(prompt, max_new_tokens=10, temperature=0.0, top_k=1)
    out2 = model.generate(prompt, max_new_tokens=10, temperature=0.0, top_k=1)
    assert torch.equal(out1, out2)


def test_generate_eos_stops_early(cfg):
    """Generation should stop when EOS token is produced."""
    model = SmallTransformer(cfg)
    model.eval()
    # Bias the LM head so token cfg.eos_token_id is always predicted
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[cfg.eos_token_id] = 100.0

    prompt = torch.tensor([[cfg.bos_token_id]])
    out = model.generate(
        prompt,
        max_new_tokens=50,
        temperature=0.0,
        top_k=1,
        eos_token_id=cfg.eos_token_id,
    )
    # Should stop very quickly
    assert out.shape[1] <= 5
