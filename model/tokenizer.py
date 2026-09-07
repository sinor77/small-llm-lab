"""
BPETokenizer — wraps HuggingFace `tokenizers` library to provide a
Byte-Pair Encoding tokenizer that:
  - trains on our synthetic dataset text
  - saves / loads deterministically
  - handles special tokens (PAD, BOS, EOS, UNK, SEP)
  - exposes a clean encode / decode interface

Why BPE over character-level?
  Arithmetic problems contain multi-digit numbers, keywords like
  "Problem:", "Reasoning:", "Answer:", and operators. A small BPE
  vocabulary (4096 tokens) tokenises these efficiently and reduces
  sequence length compared to character-level, which is important
  when context is limited to 256–512 tokens.
"""

import os
import json
from typing import List, Optional

# HuggingFace `tokenizers` — fast Rust-backed BPE
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


# ── Special token constants ───────────────────────────────────────────────────

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SEP_TOKEN = "[SEP]"   # separates problem / reasoning / answer sections

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SEP_ID = 4


# ── Tokenizer wrapper ─────────────────────────────────────────────────────────

class BPETokenizer:
    """
    Byte-level BPE tokenizer with fixed special tokens.

    Typical usage:
        tok = BPETokenizer()
        tok.train(texts, vocab_size=4096)
        tok.save("data/tokenizer/")

        tok2 = BPETokenizer.load("data/tokenizer/")
        ids  = tok2.encode("Problem: 3 + 4")
        text = tok2.decode(ids)
    """

    def __init__(self):
        self._tokenizer: Optional[Tokenizer] = None

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        texts: List[str],
        vocab_size: int = 4096,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> None:
        """
        Train BPE from a list of text strings.

        Args:
            texts:         list of training strings
            vocab_size:    target vocabulary size (including special tokens)
            min_frequency: minimum token pair frequency to merge
            show_progress: show training progress bar
        """
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=show_progress,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Post-processor: automatically prepend BOS and append EOS
        tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[
                (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
                (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
            ],
        )

        self._tokenizer = tokenizer
        self._validate_special_token_ids()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save tokenizer files to directory."""
        os.makedirs(directory, exist_ok=True)
        self._tokenizer.save(os.path.join(directory, "tokenizer.json"))
        # Save metadata for easy inspection
        meta = {
            "vocab_size":  self._tokenizer.get_vocab_size(),
            "special_tokens": SPECIAL_TOKENS,
            "special_ids": {
                PAD_TOKEN: self.pad_id,
                BOS_TOKEN: self.bos_id,
                EOS_TOKEN: self.eos_id,
                UNK_TOKEN: self.unk_id,
                SEP_TOKEN: self.sep_id,
            },
        }
        with open(os.path.join(directory, "tokenizer_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "BPETokenizer":
        """Load a previously saved tokenizer."""
        path = os.path.join(directory, "tokenizer.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No tokenizer.json found in '{directory}'")
        tok = cls()
        tok._tokenizer = Tokenizer.from_file(path)
        tok._validate_special_token_ids()
        return tok

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode text to a list of integer token ids.

        BOS and EOS are added automatically by the post-processor
        when add_special_tokens=True (the default).
        """
        enc = self._tokenizer.encode(text)
        if not add_special_tokens:
            # Strip BOS/EOS that the post-processor added
            ids = enc.ids
            if ids and ids[0] == self.bos_id:
                ids = ids[1:]
            if ids and ids[-1] == self.eos_id:
                ids = ids[:-1]
            return ids
        return enc.ids

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        encs = self._tokenizer.encode_batch(texts)
        if not add_special_tokens:
            result = []
            for enc in encs:
                ids = enc.ids
                if ids and ids[0] == self.bos_id:
                    ids = ids[1:]
                if ids and ids[-1] == self.eos_id:
                    ids = ids[:-1]
                result.append(ids)
            return result
        return [e.ids for e in encs]

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tokenizer.token_to_id(PAD_TOKEN)

    @property
    def bos_id(self) -> int:
        return self._tokenizer.token_to_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self._tokenizer.token_to_id(EOS_TOKEN)

    @property
    def unk_id(self) -> int:
        return self._tokenizer.token_to_id(UNK_TOKEN)

    @property
    def sep_id(self) -> int:
        return self._tokenizer.token_to_id(SEP_TOKEN)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_special_token_ids(self) -> None:
        """Confirm all special tokens are in the vocabulary."""
        for tok in SPECIAL_TOKENS:
            tid = self._tokenizer.token_to_id(tok)
            if tid is None:
                raise RuntimeError(
                    f"Special token '{tok}' not found in tokenizer vocabulary. "
                    "Retrain or verify the tokenizer save/load."
                )

    def __repr__(self) -> str:
        if self._tokenizer is None:
            return "BPETokenizer(untrained)"
        return f"BPETokenizer(vocab_size={self.vocab_size})"


# ── Convenience: train from a corpus file ────────────────────────────────────

def train_tokenizer_from_file(
    corpus_path: str,
    save_dir: str,
    vocab_size: int = 4096,
) -> BPETokenizer:
    """
    Train a BPETokenizer from a text file (one document per line).
    Saves the tokenizer to save_dir and returns it.
    """
    with open(corpus_path) as f:
        texts = [line.strip() for line in f if line.strip()]
    tok = BPETokenizer()
    tok.train(texts, vocab_size=vocab_size)
    tok.save(save_dir)
    return tok
