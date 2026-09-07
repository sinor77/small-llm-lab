"""
evaluate.py — top-level evaluation entry point.

Loads a trained checkpoint, generates unseen arithmetic problems,
runs the programmatic verifier, and reports accuracy.

Usage:
    python evaluate.py --checkpoint experiments/results/exp/checkpoints/best.pt
    python evaluate.py --checkpoint ... --n_problems 1000 --difficulty hard
    python evaluate.py --checkpoint ... --generalization   # run 5-level test
    python evaluate.py --checkpoint ... --split test --output my_eval.json
"""

import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from model.config import ModelConfig
from model.transformer import SmallTransformer
from model.tokenizer import BPETokenizer
from training.checkpoint import load_checkpoint
from data.generators.arithmetic import ArithmeticGenerator
from evaluation.benchmark import run_benchmark, run_generalization_benchmark


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Small Reasoning LLM checkpoint")
    parser.add_argument("--checkpoint",     required=True, type=str)
    parser.add_argument("--n_problems",     type=int,   default=500)
    parser.add_argument("--split",          type=str,   default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--difficulty",     type=str,   default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--device",         type=str,   default="auto")
    parser.add_argument("--max_new_tokens", type=int,   default=128)
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--no_reasoning",   action="store_true")
    parser.add_argument("--generalization", action="store_true",
                        help="Run the 5-level generalization benchmark")
    parser.add_argument("--n_per_level",    type=int,   default=200,
                        help="Problems per generalization level")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--output",         type=str,   default=None,
                        help="Save results to this JSON file")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Device] {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ck = torch.load(args.checkpoint, map_location=device)
    model_config = ModelConfig.from_dict(ck["model_config"])

    ck_dir         = os.path.dirname(args.checkpoint)
    experiment_dir = os.path.dirname(ck_dir)
    tok_dir        = os.path.join(experiment_dir, "tokenizer")
    if not os.path.exists(tok_dir):
        tok_dir = model_config.tokenizer_path or ""

    if not tok_dir or not os.path.exists(tok_dir):
        print(f"ERROR: Cannot find tokenizer at {tok_dir}")
        sys.exit(1)

    tokenizer = BPETokenizer.load(tok_dir)
    model     = SmallTransformer(model_config)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    model.eval()

    n_params = model.count_parameters()
    print(f"[Model] {n_params:,} parameters | step={ck.get('global_step', '?')}")
    print(f"[Tokenizer] vocab_size={tokenizer.vocab_size}")

    # ── Generator ─────────────────────────────────────────────────────────────
    generator = ArithmeticGenerator(seed=args.seed, difficulty=args.difficulty)
    use_reasoning = not args.no_reasoning

    # ── Standard benchmark ────────────────────────────────────────────────────
    print(f"\n[Eval] {args.n_problems} problems, split={args.split}, "
          f"difficulty={args.difficulty}")

    bench = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        generator=generator,
        n_problems=args.n_problems,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_reasoning=use_reasoning,
        max_samples_to_save=20,
        device=device,
    )

    print("\n" + bench.summary_str())

    all_results = {"standard": bench.to_dict()}

    # ── Generalization benchmark ──────────────────────────────────────────────
    if args.generalization:
        print(f"\n[Eval] Generalization benchmark ({args.n_per_level} per level)...")
        gen_results = run_generalization_benchmark(
            model=model,
            tokenizer=tokenizer,
            generator=generator,
            n_per_level=args.n_per_level,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            use_reasoning=use_reasoning,
        )
        print("\nGeneralization results:")
        for level, r in gen_results.items():
            print(f"  Level {level}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
        all_results["generalization"] = gen_results

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = args.output
    if output_path is None:
        exp_dir     = os.path.dirname(os.path.dirname(args.checkpoint))
        output_path = os.path.join(exp_dir, f"evaluation_{args.split}.json")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Output] Results saved to {output_path}")


if __name__ == "__main__":
    main()
