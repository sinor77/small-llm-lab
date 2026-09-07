"""Calculate exact parameter counts for candidate architectures."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import SmallTransformer

candidates = [
    # (label, d_model, n_layers, n_heads, d_ff, vocab, max_seq)
    ("current 8m",        256, 6, 8, 1024, 4096, 256),
    ("d256 L8 ff1024",    256, 8, 8, 1024, 4096, 256),
    ("d256 L8 ff1280",    256, 8, 8, 1280, 4096, 256),
    ("d256 L8 ff1152",    256, 8, 8, 1152, 4096, 256),
    ("d256 L9 ff1024",    256, 9, 8, 1024, 4096, 256),
    ("d288 L7 ff1024",    288, 7, 8, 1024, 4096, 256),
    ("d320 L6 ff1024",    320, 6, 8, 1024, 4096, 256),
    ("d256 L8 ff1200",    256, 8, 8, 1200, 4096, 256),
    ("d256 L7 ff1280",    256, 7, 8, 1280, 4096, 256),
    ("d256 L10 ff1024",   256,10, 8, 1024, 4096, 256),
]

print(f"{'Label':<25} {'d_model':>7} {'layers':>6} {'heads':>5} {'d_ff':>6} "
      f"{'trainable':>12} {'total':>12}")
print("-" * 80)
for label, d, L, H, ff, V, S in candidates:
    cfg = ModelConfig(vocab_size=V, d_model=d, n_layers=L, n_heads=H,
                      d_ff=ff, max_seq_len=S, dropout=0.0)
    m = SmallTransformer(cfg)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m.parameters())
    tied      = m.lm_head.weight is m.token_emb.weight
    print(f"{label:<25} {d:>7} {L:>6} {H:>5} {ff:>6} "
          f"{trainable:>12,} {total:>12,}  tied={tied}")
