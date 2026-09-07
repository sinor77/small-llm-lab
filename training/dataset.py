"""
ReasoningDataset — PyTorch Dataset for language-model training on
structured reasoning examples.

Each example is formatted as:
    Problem: ...
    Reasoning:      ← (optional, controlled by use_reasoning flag)
    Step 1: ...
    Answer: ...

Training target:
    The input to the model is the tokenized full example.
    The target (labels) is the same sequence shifted by one position to the
    right — standard causal language model training.

    Padding tokens in the labels are set to pad_id (ignored by the loss).

Sequence packing (optional):
    For very short examples, packing multiple examples into one sequence
    improves GPU utilization. Not used in the baseline to keep things simple.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional

from data.generators.arithmetic import Example, build_corpus


class ReasoningDataset(Dataset):
    """
    Language-model dataset from a list of Example objects.

    Args:
        examples:        list of Example objects
        tokenizer:       BPETokenizer instance (must be trained)
        max_seq_len:     maximum sequence length; examples longer than this
                         are truncated (a warning is raised for heavy truncation)
        use_reasoning:   if True, use full reasoning format; else direct format
        pad_to_max_len:  if True, pad all sequences to max_seq_len (for
                         fixed-size batches without DataLoader padding)
    """

    def __init__(
        self,
        examples: List[Example],
        tokenizer,
        max_seq_len: int = 256,
        use_reasoning: bool = True,
        pad_to_max_len: bool = False,
    ):
        self.tokenizer      = tokenizer
        self.max_seq_len    = max_seq_len
        self.use_reasoning  = use_reasoning
        self.pad_to_max_len = pad_to_max_len

        # Tokenise all examples at construction time
        self.input_ids_list: List[torch.Tensor] = []
        self._tokenize(examples)

    def _tokenize(self, examples: List[Example]) -> None:
        truncated = 0
        for ex in examples:
            text = ex.to_reasoning_text() if self.use_reasoning else ex.to_direct_text()
            ids  = self.tokenizer.encode(text, add_special_tokens=True)

            if len(ids) > self.max_seq_len:
                ids       = ids[:self.max_seq_len]
                truncated += 1

            self.input_ids_list.append(torch.tensor(ids, dtype=torch.long))

        if truncated:
            pct = 100 * truncated / len(examples)
            print(
                f"[Dataset] {truncated}/{len(examples)} ({pct:.1f}%) examples were "
                f"truncated to max_seq_len={self.max_seq_len}. "
                "Consider increasing max_seq_len or reducing problem complexity."
            )

    def __len__(self) -> int:
        return len(self.input_ids_list)

    def __getitem__(self, idx: int):
        ids = self.input_ids_list[idx]
        T   = ids.shape[0]

        # Input: all tokens except the last
        # Target: all tokens except the first (shifted by 1)
        if T < 2:
            # Edge case: sequence too short — pad with a dummy
            pad = torch.full((2,), self.tokenizer.pad_id, dtype=torch.long)
            ids = torch.cat([ids, pad])[:2]

        x = ids[:-1]
        y = ids[1:]

        if self.pad_to_max_len:
            pad_len = self.max_seq_len - 1 - x.shape[0]
            if pad_len > 0:
                pad_t = torch.full((pad_len,), self.tokenizer.pad_id, dtype=torch.long)
                x = torch.cat([x, pad_t])
                y = torch.cat([y, pad_t])

        return x, y

    def token_count(self) -> int:
        """Total number of tokens across all examples."""
        return sum(ids.shape[0] for ids in self.input_ids_list)

    def stats(self) -> dict:
        """Basic dataset statistics."""
        lengths = [ids.shape[0] for ids in self.input_ids_list]
        return {
            "n_examples":  len(self),
            "total_tokens": self.token_count(),
            "min_len":     min(lengths),
            "max_len":     max(lengths),
            "avg_len":     sum(lengths) / len(lengths),
        }


def collate_fn(batch, pad_id: int):
    """
    Collate variable-length sequences by right-padding to the longest in the batch.
    Padding tokens in targets are set to pad_id so the loss ignores them.
    """
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)

    padded_x = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), pad_id, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        L = x.shape[0]
        padded_x[i, :L] = x
        padded_y[i, :L] = y

    return padded_x, padded_y


def make_dataloader(
    examples: List[Example],
    tokenizer,
    max_seq_len: int,
    batch_size: int,
    use_reasoning: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience function to build a DataLoader from a list of examples.
    """
    dataset = ReasoningDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        use_reasoning=use_reasoning,
    )

    pad_id = tokenizer.pad_id

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader
