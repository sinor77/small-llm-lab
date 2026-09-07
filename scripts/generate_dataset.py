"""
Standalone script to generate and save a dataset to disk.

Uses generate_all_splits() to guarantee zero cross-split problem overlap.

Usage:
    python scripts/generate_dataset.py \
        --n_train 20000 --n_val 2000 --n_test 1000 \
        --difficulty medium \
        --output data/datasets/medium_20k \
        --seed 42
"""

import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.generators.arithmetic import ArithmeticGenerator, build_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train",       type=int, default=20000)
    parser.add_argument("--n_val",         type=int, default=2000)
    parser.add_argument("--n_test",        type=int, default=1000)
    parser.add_argument("--difficulty",    type=str, default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--output",        type=str, default="data/datasets/default")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--use_reasoning", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    gen = ArithmeticGenerator(seed=args.seed, difficulty=args.difficulty)

    # generate_all_splits guarantees zero cross-split exact overlap
    print(f"Generating splits (deduplicated, seed={args.seed}, "
          f"difficulty={args.difficulty}) ...")
    train_ex, val_ex, test_ex = gen.generate_all_splits(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
    )

    for split_name, examples in [("train", train_ex), ("val", val_ex), ("test", test_ex)]:
        out_path = os.path.join(args.output, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict()) + "\n")
        corpus_path = os.path.join(args.output, f"{split_name}_corpus.txt")
        with open(corpus_path, "w") as f:
            for ex in examples:
                line = ex.to_reasoning_text() if args.use_reasoning else ex.to_direct_text()
                f.write(line + "\n")
        print(f"  {split_name}: {len(examples)} examples -> {out_path}")

    # Verify zero overlap and write to metadata
    train_p = set(e.problem for e in train_ex)
    val_p   = set(e.problem for e in val_ex)
    test_p  = set(e.problem for e in test_ex)
    tv = len(train_p & val_p)
    tt = len(train_p & test_p)
    vt = len(val_p   & test_p)
    print(f"  Overlap check: train/val={tv}, train/test={tt}, val/test={vt}")
    assert tv == 0 and tt == 0 and vt == 0, "Split overlap detected — this is a bug"

    meta = {
        "seed":          args.seed,
        "difficulty":    args.difficulty,
        "n_train":       len(train_ex),
        "n_val":         len(val_ex),
        "n_test":        len(test_ex),
        "use_reasoning": args.use_reasoning,
        "train_val_overlap":  tv,
        "train_test_overlap": tt,
        "val_test_overlap":   vt,
        "deduplication":      "generate_all_splits — zero cross-split overlap guaranteed",
    }
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
