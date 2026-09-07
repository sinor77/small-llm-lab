"""
Benchmark runner — evaluates a trained model against unseen generated problems.

Usage:
    python evaluate.py --checkpoint experiments/exp_001/checkpoints/best.pt \
                       --config experiments/exp_001/model_config.json \
                       --tokenizer experiments/exp_001/tokenizer/ \
                       --n_problems 500 \
                       --split test \
                       --output experiments/exp_001/evaluation.json

The runner:
1. Loads model + tokenizer from checkpoint.
2. Generates arithmetic problems for the requested split.
3. Prompts the model with the problem (up to "Answer:").
4. Lets the model generate the answer.
5. Runs the programmatic verifier on each generated answer.
6. Reports accuracy per problem family and per generalization level.
7. Saves detailed results to evaluation.json.
"""

import json
import time
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    total:               int
    correct:             int
    accuracy:            float
    family_accuracy:     dict         # {"single_op": 0.82, ...}
    difficulty_accuracy: dict         # {"easy": 0.9, "medium": 0.7, ...}
    gen_level_accuracy:  dict         # {1: 0.8, 2: 0.6, ...}
    avg_gen_tokens:      float
    eval_time_s:         float
    samples:             list         # list of sample result dicts (first N)

    def summary_str(self) -> str:
        lines = [
            f"Accuracy: {self.correct}/{self.total} = {self.accuracy:.1%}",
            "By family:",
        ]
        for fam, acc in sorted(self.family_accuracy.items()):
            lines.append(f"  {fam:<25} {acc:.1%}")
        lines.append("By difficulty:")
        for diff, acc in sorted(self.difficulty_accuracy.items()):
            lines.append(f"  {diff:<25} {acc:.1%}")
        if self.gen_level_accuracy:
            lines.append("By generalization level:")
            for lev, acc in sorted(self.gen_level_accuracy.items()):
                lines.append(f"  level {lev}            {acc:.1%}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def run_benchmark(
    model,
    tokenizer,
    generator,
    n_problems: int = 500,
    split: str = "test",
    max_new_tokens: int = 64,
    temperature: float = 0.0,   # 0.0 = greedy
    top_k: int = 1,
    use_reasoning: bool = True,
    max_samples_to_save: int = 50,
    device: str = "cpu",
) -> BenchmarkResult:
    """
    Run a complete benchmark evaluation.

    Args:
        model:              SmallTransformer (already on device, in eval mode)
        tokenizer:          BPETokenizer
        generator:          ArithmeticGenerator
        n_problems:         number of evaluation problems
        split:              "train" | "val" | "test"
        max_new_tokens:     max tokens to generate per answer
        temperature:        sampling temperature (0.0 = greedy argmax)
        top_k:              top-k for sampling
        use_reasoning:      if True, prompt ends at "Reasoning:\n"; if False at "Answer:"
        max_samples_to_save: number of full sample dicts to include in results
        device:             torch device string

    Returns:
        BenchmarkResult
    """
    import torch
    from evaluation.arithmetic import verify

    examples = generator.generate(n=n_problems, split=split)

    correct_total  = 0
    family_correct = {}
    family_total   = {}
    diff_correct   = {}
    diff_total     = {}
    total_tokens   = 0
    samples        = []
    t0             = time.time()

    model.eval()

    for i, ex in enumerate(examples):
        # Build prompt: everything up to where the model should generate
        if use_reasoning:
            prompt = f"Problem: {ex.problem}\nReasoning:\n"
        else:
            prompt = f"Problem: {ex.problem}\nAnswer: "

        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Generate
        with torch.no_grad():
            if temperature <= 0.0:
                # Greedy decoding
                output = _greedy_generate(
                    model, input_tensor, max_new_tokens,
                    eos_id=tokenizer.eos_id,
                )
            else:
                output = model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    eos_token_id=tokenizer.eos_id,
                )

        generated_ids  = output[0, len(input_ids):].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text      = prompt + generated_text
        total_tokens  += len(generated_ids)

        # Verify
        vresult = verify(full_text, ex.answer, ex.answer_num)
        if vresult["correct"]:
            correct_total += 1

        # Track by family
        fam = ex.family
        family_total[fam]   = family_total.get(fam, 0) + 1
        family_correct[fam] = family_correct.get(fam, 0) + (1 if vresult["correct"] else 0)

        # Track by difficulty
        diff = ex.difficulty
        diff_total[diff]   = diff_total.get(diff, 0) + 1
        diff_correct[diff] = diff_correct.get(diff, 0) + (1 if vresult["correct"] else 0)

        # Save sample details
        if i < max_samples_to_save:
            samples.append({
                "index":          i,
                "problem":        ex.problem,
                "expected":       ex.answer,
                "generated_text": generated_text,
                "correct":        vresult["correct"],
                "predicted":      vresult["predicted"],
                "reason":         vresult["reason"],
                "family":         fam,
                "difficulty":     diff,
            })

    elapsed = time.time() - t0
    n = len(examples)

    family_acc = {
        fam: family_correct[fam] / family_total[fam]
        for fam in family_total
    }
    diff_acc = {
        diff: diff_correct[diff] / diff_total[diff]
        for diff in diff_total
    }

    return BenchmarkResult(
        total=n,
        correct=correct_total,
        accuracy=correct_total / n if n > 0 else 0.0,
        family_accuracy=family_acc,
        difficulty_accuracy=diff_acc,
        gen_level_accuracy={},
        avg_gen_tokens=total_tokens / n if n > 0 else 0.0,
        eval_time_s=elapsed,
        samples=samples,
    )


def run_generalization_benchmark(
    model,
    tokenizer,
    generator,
    n_per_level: int = 200,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    device: str = "cpu",
    use_reasoning: bool = True,
    exclude_problems: set = None,
) -> dict:
    """
    Run the 5-level generalization benchmark.

    Args:
        exclude_problems: set of problem strings from train/val/test to exclude
                          from the generalization set. Pass this to guarantee
                          zero overlap with training data.

    Returns a dict: level -> result dict
    """
    import torch
    from evaluation.arithmetic import verify

    gen_set = generator.generate_generalization_set(
        n_per_level=n_per_level,
        exclude=exclude_problems,
    )
    results = {}

    model.eval()

    for level, examples in gen_set.items():
        correct = 0
        samples = []
        for i, ex in enumerate(examples):
            if use_reasoning:
                prompt = f"Problem: {ex.problem}\nReasoning:\n"
            else:
                prompt = f"Problem: {ex.problem}\nAnswer: "

            input_ids    = tokenizer.encode(prompt, add_special_tokens=False)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                output = _greedy_generate(
                    model, input_tensor, max_new_tokens,
                    eos_id=tokenizer.eos_id,
                )

            gen_ids  = output[0, len(input_ids):].tolist()
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            full_text = prompt + gen_text

            vr = verify(full_text, ex.answer, ex.answer_num)
            if vr["correct"]:
                correct += 1

            if i < 20:
                samples.append({
                    "problem":   ex.problem,
                    "expected":  ex.answer,
                    "generated": gen_text,
                    "correct":   vr["correct"],
                })

        n = len(examples)
        acc = correct / n if n > 0 else 0.0
        results[level] = {
            "accuracy": acc,
            "correct":  correct,
            "total":    n,
            "samples":  samples,
        }

    return results


def _greedy_generate(model, input_ids, max_new_tokens: int, eos_id: int):
    """Pure greedy decoding (argmax at each step)."""
    import torch
    for _ in range(max_new_tokens):
        ctx    = input_ids[:, -model.config.max_seq_len:]
        logits = model(ctx)[:, -1, :]
        next_t = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_t], dim=1)
        if (next_t == eos_id).all():
            break
    return input_ids
