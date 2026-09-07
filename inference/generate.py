"""
Generate — interactive inference from a saved checkpoint.

Usage:
    # Interactive mode (reads problems from stdin):
    python inference/generate.py --checkpoint experiments/results/exp/checkpoints/best.pt

    # Single-shot:
    python inference/generate.py \
        --checkpoint experiments/results/exp/checkpoints/best.pt \
        --problem "What is 37 + 58?"

    # From a problems file (one problem per line):
    python inference/generate.py \
        --checkpoint experiments/results/exp/checkpoints/best.pt \
        --problems_file my_problems.txt \
        --output results.json
"""

import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from model.config import ModelConfig
from model.transformer import SmallTransformer
from model.tokenizer import BPETokenizer
from training.checkpoint import load_checkpoint
from evaluation.arithmetic import verify


# ── Loader ────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(checkpoint_path: str, device: str = "cpu"):
    """
    Load model and tokenizer from a checkpoint.

    The checkpoint stores the model_config dict; the tokenizer is stored
    alongside the checkpoint in an adjacent 'tokenizer/' directory
    (standard experiment layout) or can be passed explicitly.

    Returns:
        (model, tokenizer, checkpoint_dict)
    """
    ck = torch.load(checkpoint_path, map_location=device)

    # Load model config from checkpoint
    model_config = ModelConfig.from_dict(ck["model_config"])

    # Locate tokenizer directory (relative to checkpoint)
    ck_dir         = os.path.dirname(checkpoint_path)
    experiment_dir = os.path.dirname(ck_dir)   # checkpoints/ → experiment root
    tok_dir        = os.path.join(experiment_dir, "tokenizer")

    if not os.path.exists(tok_dir):
        # Fallback: look for tokenizer path stored in model_config
        tok_dir = model_config.tokenizer_path or ""

    if not tok_dir or not os.path.exists(tok_dir):
        raise FileNotFoundError(
            f"Cannot locate tokenizer directory. Expected at: {tok_dir}\n"
            "Make sure the experiment directory structure is intact."
        )

    tokenizer = BPETokenizer.load(tok_dir)

    model = SmallTransformer(model_config)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    model.eval()

    return model, tokenizer, ck


# ── Single inference ──────────────────────────────────────────────────────────

def generate_answer(
    problem: str,
    model: SmallTransformer,
    tokenizer: BPETokenizer,
    device: str = "cpu",
    use_reasoning: bool = True,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_k: int = 1,
) -> dict:
    """
    Generate a model answer for a single problem string.

    Returns a dict with:
        prompt, generated_text, full_text, extracted_answer
    """
    if use_reasoning:
        prompt = f"Problem: {problem}\nReasoning:\n"
    else:
        prompt = f"Problem: {problem}\nAnswer: "

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        if temperature <= 0.0:
            # Greedy
            for _ in range(max_new_tokens):
                ctx    = input_tensor[:, -model.config.max_seq_len:]
                logits = model(ctx)[:, -1, :]
                nxt    = logits.argmax(dim=-1, keepdim=True)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)
                if nxt.item() == tokenizer.eos_id:
                    break
        else:
            input_tensor = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=tokenizer.eos_id,
            )

    generated_ids  = input_tensor[0, len(input_ids):].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text      = prompt + generated_text

    # Extract answer
    from evaluation.arithmetic import _extract_answer
    extracted = _extract_answer(full_text)

    return {
        "prompt":            prompt,
        "generated_text":    generated_text,
        "full_text":         full_text,
        "extracted_answer":  extracted,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate answers from a trained checkpoint")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to checkpoint file (.pt)")
    parser.add_argument("--problem", type=str, default=None,
                        help="A single problem string to solve")
    parser.add_argument("--problems_file", type=str, default=None,
                        help="Path to file with one problem per line")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON (for batch mode)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto | cpu | cuda | mps")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Top-k sampling (only used if temperature > 0)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max tokens to generate per answer")
    parser.add_argument("--no_reasoning", action="store_true",
                        help="Use direct problem→answer format (no chain-of-thought)")
    parser.add_argument("--expected_answer", type=str, default=None,
                        help="If provided, run the verifier on the result")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, ck = load_model_and_tokenizer(args.checkpoint, device=device)
    n_params = model.count_parameters()
    print(f"Model loaded: {n_params:,} parameters | step={ck.get('global_step', '?')}")

    use_reasoning = not args.no_reasoning

    # ── Single problem ────────────────────────────────────────────────────────
    if args.problem:
        result = generate_answer(
            problem=args.problem,
            model=model,
            tokenizer=tokenizer,
            device=device,
            use_reasoning=use_reasoning,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print("\n" + "=" * 60)
        print(result["full_text"])
        print("=" * 60)
        print(f"Extracted answer: {result['extracted_answer']}")

        if args.expected_answer is not None:
            try:
                exp_num = float(args.expected_answer)
            except ValueError:
                exp_num = 0.0
            vr = verify(result["full_text"], args.expected_answer, exp_num)
            status = "✓ CORRECT" if vr["correct"] else "✗ INCORRECT"
            print(f"Verification: {status} (expected: {args.expected_answer})")
        return

    # ── Batch from file ───────────────────────────────────────────────────────
    if args.problems_file:
        with open(args.problems_file) as f:
            problems = [l.strip() for l in f if l.strip()]

        results = []
        for i, prob in enumerate(problems):
            res = generate_answer(
                problem=prob,
                model=model,
                tokenizer=tokenizer,
                device=device,
                use_reasoning=use_reasoning,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            res["problem"] = prob
            results.append(res)
            print(f"[{i+1}/{len(problems)}] {prob[:60]} → {res['extracted_answer']}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        return

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("\nInteractive mode. Enter a math problem (or 'quit' to exit).")
    while True:
        try:
            problem = input("\nProblem> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not problem or problem.lower() in ("quit", "exit"):
            break

        result = generate_answer(
            problem=problem,
            model=model,
            tokenizer=tokenizer,
            device=device,
            use_reasoning=use_reasoning,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print("\n" + result["full_text"])
        print(f"\nExtracted answer: {result['extracted_answer']}")


if __name__ == "__main__":
    main()
