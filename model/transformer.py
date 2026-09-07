"""
SmallTransformer — decoder-only Transformer language model.

Architecture (per decoder block):
    x → LayerNorm → CausalSelfAttention → residual add
      → LayerNorm → FeedForward         → residual add

Final output:
    x → LayerNorm → LM head (linear projection to vocab)

Notes:
- Pre-norm (LayerNorm before each sub-layer) is used: more stable for
  small models trained from scratch.
- Learned absolute positional embeddings. Sinusoidal or RoPE can be
  swapped in without changing the rest of the architecture.
- All weights are randomly initialised — no pretrained checkpoint is
  loaded anywhere in this file.
- Weight tying between token embedding and LM head is supported and
  enabled by default (reduces parameter count, often improves quality).
"""

import math
import torch
import torch.nn as nn

from .config import ModelConfig
from .attention import CausalSelfAttention


# ── Feed-Forward Network ──────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise FFN:  Linear → GELU → Linear (+ dropout).

    Using GELU rather than ReLU following standard modern practice.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Single Decoder Block ─────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """One transformer decoder block (pre-norm)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff  = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.ln1(x))
        # Feed-forward with residual
        x = x + self.ff(self.ln2(x))
        return x


# ── Full Model ───────────────────────────────────────────────────────────────

class SmallTransformer(nn.Module):
    """
    Decoder-only Transformer language model trained from random initialisation.

    Args:
        config: ModelConfig instance

    Usage:
        model = SmallTransformer(config)
        logits = model(input_ids)            # (B, T, vocab_size)
        loss   = model(input_ids, targets)   # returns scalar loss
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model,
                                       padding_idx=config.pad_token_id)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop  = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head  = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying — token embedding and LM head share the same matrix.
        # This is standard practice and cuts ~vocab_size * d_model parameters.
        self.lm_head.weight = self.token_emb.weight

        # Initialise weights
        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Initialise weights following GPT-2 style:
        - Linear layers: normal(0, 0.02)
        - Embeddings:    normal(0, 0.02)
        - LayerNorm:     weight=1, bias=0
        - Residual projection scaling by 1/sqrt(n_layers) for stability.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Scale output projections of attention and FFN by 1/sqrt(2 * n_layers)
        # (GPT-2 trick) to keep residual stream variance bounded at init.
        scale = (2 * self.config.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("net.3.weight"):
                p.data.mul_(scale)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (B, T) token indices
            targets:   (B, T) token indices, shifted by 1 from input_ids
                       (i.e. targets[t] = input_ids[t+1]). If provided,
                       computes cross-entropy loss.

        Returns:
            logits:    (B, T, vocab_size)   if targets is None
            (loss, logits) if targets is provided
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # Build position indices and embed
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(pos))

        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x)

        # Final norm + project to vocabulary
        x      = self.ln_final(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        if targets is None:
            return logits

        # Cross-entropy loss (ignore padding tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
            ignore_index=self.config.pad_token_id,
        )
        # Guard: if every target token was padding, cross_entropy returns NaN.
        # This should not happen with well-formed batches, but return 0 gracefully.
        if torch.isnan(loss):
            loss = torch.zeros(1, device=logits.device, requires_grad=True)[0]
        return loss, logits

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_breakdown(self) -> dict:
        """Returns a breakdown of parameters by component."""
        breakdown = {}
        for name, module in self.named_children():
            count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            breakdown[name] = count
        breakdown["total"] = self.count_parameters()
        return breakdown

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            input_ids:      (B, T) prompt token ids
            max_new_tokens: maximum tokens to generate
            temperature:    softmax temperature (1.0 = no change, <1 = sharper)
            top_k:          if > 0, only sample from top-k tokens
            top_p:          nucleus sampling threshold (1.0 = disabled)
            eos_token_id:   stop generation when this token is produced

        Returns:
            (B, T + generated) token ids
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            ctx = input_ids[:, -self.config.max_seq_len:]
            logits = self(ctx)                    # (B, T', vocab)
            logits = logits[:, -1, :]             # last position only

            # temperature=0 → greedy argmax (avoids division by zero)
            if temperature <= 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)   # (B, 1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
                continue

            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                threshold = logits.topk(top_k_val, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                cum_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens where cumulative prob exceeds top_p
                sorted_logits[cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p] = float("-inf")
                logits = torch.scatter(logits, -1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (B, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids
