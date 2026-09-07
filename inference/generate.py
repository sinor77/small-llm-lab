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

Public API (importable from notebooks/scripts):
    load_model_and_tokenizer(checkpoint_path, device)
    generate_answer(problem, model, tokenizer, ...)
    solve_problem(problem, checkpoint_path, ...)
    interactive_inference(checkpoint_path, ...)
    compare_checkpoints(problem, experiment_dir, ...)
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

    This is GENERATION ONLY. The model output is not verified by another
    LLM. Use the returned 'extracted_answer' with the programmatic verifier
    (evaluation.arithmetic.verify) if you need correctness checking.

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

    from evaluation.arithmetic import _extract_answer
    extracted = _extract_answer(full_text)

    return {
        "prompt":            prompt,
        "generated_text":    generated_text,
        "full_text":         full_text,
        "extracted_answer":  extracted,
    }


# ── One-shot solve ────────────────────────────────────────────────────────────

def solve_problem(
    problem: str,
    checkpoint_path: str,
    device: str = "auto",
    use_reasoning: bool = True,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    expected_answer: str = None,
    verbose: bool = True,
) -> dict:
    """
    Load a checkpoint and solve a single problem in one call.

    Args:
        problem:           the problem text (no "Problem:" prefix needed)
        checkpoint_path:   path to a .pt checkpoint file, OR one of the
                           shorthand labels: "init", "early", "mid",
                           "final", "best" — which resolve relative to the
                           checkpoint's own directory when used via
                           compare_checkpoints().
        device:            "auto" | "cpu" | "cuda" | "mps"
        use_reasoning:     True = chain-of-thought format (default)
        max_new_tokens:    max tokens the model generates
        temperature:       0.0 = greedy (default, deterministic)
        expected_answer:   if provided, runs the programmatic verifier and
                           includes correctness in the return dict.
                           NOTE: verification is purely numerical/exact-match.
                           No LLM is used as a judge.
        verbose:           if True, prints the full model output

    Returns:
        dict with keys: full_text, extracted_answer, correct (if expected given),
                        checkpoint_step, n_params
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, ck = load_model_and_tokenizer(checkpoint_path, device=device)

    result = generate_answer(
        problem=problem,
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_reasoning=use_reasoning,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    if verbose:
        print(result["full_text"])
        print(f"\nExtracted answer: {result['extracted_answer']}")

    output = {
        "full_text":       result["full_text"],
        "extracted_answer": result["extracted_answer"],
        "checkpoint_step": ck.get("global_step", "?"),
        "n_params":        model.count_parameters(),
    }

    if expected_answer is not None:
        try:
            exp_num = float(expected_answer)
        except ValueError:
            exp_num = 0.0
        vr = verify(result["full_text"], expected_answer, exp_num)
        output["correct"]  = vr["correct"]
        output["expected"] = expected_answer
        if verbose:
            status = "CORRECT" if vr["correct"] else "INCORRECT"
            print(f"Verification: {status} (expected: {expected_answer})")

    return output


# ── Interactive inference (Colab-friendly) ────────────────────────────────────

def interactive_inference(
    checkpoint_path: str,
    device: str = "auto",
    use_reasoning: bool = True,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    show_extracted_answer: bool = True,
    verify_answers: bool = False,
) -> None:
    """
    Load a checkpoint and enter an interactive problem-solving loop.

    Designed to be called from a Colab cell or terminal. Accepts problems
    until the user types 'exit' or 'quit' (or presses Ctrl+C).

    This is an INSPECTION TOOL. The model is experimental and currently
    achieves ~17.8% accuracy on unseen medium-difficulty arithmetic.
    Outputs should not be trusted as reliable arithmetic.

    Args:
        checkpoint_path:       path to .pt checkpoint file
        device:                "auto" | "cpu" | "cuda" | "mps"
        use_reasoning:         True = chain-of-thought, False = direct answer
        max_new_tokens:        max tokens generated per response
        temperature:           0.0 = greedy/deterministic (recommended)
        show_extracted_answer: print the parsed answer below each response
        verify_answers:        if True, prompt for expected answer and run
                               the programmatic verifier after each response.
                               NOTE: verification is exact-match only, no LLM.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {checkpoint_path}")
    model, tokenizer, ck = load_model_and_tokenizer(checkpoint_path, device=device)
    n_params = model.count_parameters()
    step     = ck.get("global_step", "?")

    print()
    print("=" * 55)
    print("  Small Reasoning LLM — Interactive Inference")
    print("=" * 55)
    print(f"  Checkpoint:  {os.path.basename(checkpoint_path)}")
    print(f"  Parameters:  {n_params:,}")
    print(f"  Step:        {step}")
    print(f"  Device:      {device}")
    print(f"  Format:      {'reasoning (chain-of-thought)' if use_reasoning else 'direct'}")
    print(f"  Temperature: {temperature} ({'greedy' if temperature == 0.0 else 'sampled'})")
    print()
    print("  NOTE: This model is experimental (~17.8% test accuracy).")
    print("  Type a math problem, or 'exit' to quit.")
    print("=" * 55)

    while True:
        print()
        try:
            problem = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Exiting]")
            break

        if not problem:
            continue

        if problem.lower() in ("exit", "quit", "q"):
            print("[Exiting]")
            break

        # Generate
        result = generate_answer(
            problem=problem,
            model=model,
            tokenizer=tokenizer,
            device=device,
            use_reasoning=use_reasoning,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        print()
        print("Model:")
        print(result["full_text"])

        if show_extracted_answer:
            print(f"\n  [Extracted answer: {result['extracted_answer']}]")

        # Optional programmatic verification (no LLM judge)
        if verify_answers:
            try:
                expected = input("  Expected answer (or Enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if expected:
                try:
                    exp_num = float(expected)
                except ValueError:
                    exp_num = 0.0
                vr = verify(result["full_text"], expected, exp_num)
                status = "CORRECT" if vr["correct"] else "INCORRECT"
                print(f"  Verification: {status} "
                      f"(model={vr['predicted']}, expected={vr['expected']})")


# ── Checkpoint comparison ─────────────────────────────────────────────────────

# Standard label → filename mapping
_CHECKPOINT_LABELS = {
    "init":  "init.pt",
    "early": "early.pt",
    "mid":   "mid.pt",
    "final": "final.pt",
    "best":  "best.pt",
}


def compare_checkpoints(
    problem: str,
    experiment_dir: str,
    labels: list = None,
    device: str = "auto",
    use_reasoning: bool = True,
    max_new_tokens: int = 128,
) -> dict:
    """
    Run the same problem through multiple progression checkpoints and
    compare outputs side-by-side.

    Useful for observing whether the model's reasoning behavior develops
    over training, rather than only examining final accuracy.

    Args:
        problem:        the problem text
        experiment_dir: path to the experiment root directory, e.g.
                        "experiments/results/colab_small_baseline"
        labels:         list of checkpoint labels to compare.
                        Defaults to ["init", "early", "mid", "final", "best"].
                        Each label must correspond to a .pt file in
                        experiment_dir/checkpoints/.
        device:         "auto" | "cpu" | "cuda" | "mps"
        use_reasoning:  True = chain-of-thought format
        max_new_tokens: max tokens generated per checkpoint

    Returns:
        dict mapping label -> {full_text, extracted_answer, step, exists}

    Example:
        results = compare_checkpoints(
            "Calculate: 48 + 18 + 16",
            "experiments/results/colab_small_baseline",
        )
        for label, r in results.items():
            print(f"--- {label} (step {r['step']}) ---")
            print(r['full_text'])
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if labels is None:
        labels = ["init", "early", "mid", "final", "best"]

    ck_dir  = os.path.join(experiment_dir, "checkpoints")
    results = {}

    print(f"Problem: {problem}")
    print("=" * 55)

    for label in labels:
        filename = _CHECKPOINT_LABELS.get(label, label if label.endswith(".pt") else label + ".pt")
        ck_path  = os.path.join(ck_dir, filename)

        if not os.path.exists(ck_path):
            print(f"\n[{label}] checkpoint not found: {ck_path}")
            results[label] = {"exists": False, "full_text": None,
                              "extracted_answer": None, "step": None}
            continue

        model, tokenizer, ck = load_model_and_tokenizer(ck_path, device=device)
        step = ck.get("global_step", "?")

        result = generate_answer(
            problem=problem,
            model=model,
            tokenizer=tokenizer,
            device=device,
            use_reasoning=use_reasoning,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )

        print(f"\n--- {label} (step {step}) ---")
        print(result["full_text"])
        print(f"[Extracted: {result['extracted_answer']}]")

        results[label] = {
            "exists":           True,
            "full_text":        result["full_text"],
            "extracted_answer": result["extracted_answer"],
            "step":             step,
        }

        # Release GPU memory between checkpoints
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


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

    use_reasoning = not args.no_reasoning

    # ── Single problem ────────────────────────────────────────────────────────
    if args.problem:
        solve_problem(
            problem=args.problem,
            checkpoint_path=args.checkpoint,
            device=device,
            use_reasoning=use_reasoning,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            expected_answer=args.expected_answer,
            verbose=True,
        )
        return

    # ── Batch from file ───────────────────────────────────────────────────────
    if args.problems_file:
        model, tokenizer, ck = load_model_and_tokenizer(args.checkpoint, device=device)
        n_params = model.count_parameters()
        print(f"Model loaded: {n_params:,} parameters | step={ck.get('global_step', '?')}")

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
            print(f"[{i+1}/{len(problems)}] {prob[:60]} -> {res['extracted_answer']}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        return

    # ── Interactive mode ──────────────────────────────────────────────────────
    interactive_inference(
        checkpoint_path=args.checkpoint,
        device=device,
        use_reasoning=use_reasoning,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
