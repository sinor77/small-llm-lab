"""
Causal multi-head self-attention.

Design notes:
- Standard scaled dot-product attention with a causal mask.
- We pre-register the mask as a buffer so it moves with .to(device) and
  does not need to be recreated on every forward pass.
- Projection weights are kept separate (not merged) so the implementation
  stays readable and inspectable.
- Uses torch.nn.functional.scaled_dot_product_attention when available
  (PyTorch >= 2.0) for memory-efficient flash-attention paths; falls back
  to an explicit implementation otherwise.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention block.

    Args:
        d_model:   model embedding dimension
        n_heads:   number of attention heads (d_model must be divisible by n_heads)
        max_seq_len: maximum sequence length (used to pre-build the causal mask)
        dropout:   attention dropout probability (applied during training)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.dropout = dropout

        # Linear projections — kept as separate modules for clarity
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask: upper-triangular (excluding diagonal) filled with -inf.
        # Shape: (1, 1, max_seq_len, max_seq_len)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        # Register as buffer — moves to correct device automatically
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

        # Detect flash attention availability (PyTorch >= 2.0)
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.shape

        # Project to queries, keys, values
        q = self.q_proj(x)   # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if self._use_flash:
            # Flash attention handles the causal mask internally
            attn_drop_p = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=attn_drop_p,
                is_causal=True,
            )
        else:
            # Manual implementation
            scale = 1.0 / math.sqrt(self.d_head)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

            # Apply causal mask — positions that should be masked get -inf
            mask = self.causal_mask[:, :, :T, :T]   # trim to current seq length
            scores = scores.masked_fill(~mask, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_drop(attn_weights)
            y = torch.matmul(attn_weights, v)        # (B, H, T, d_head)

        # Re-assemble heads: (B, H, T, d_head) → (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + residual dropout
        return self.resid_drop(self.o_proj(y))
