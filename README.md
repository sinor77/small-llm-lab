# Small Reasoning LLM Lab

A research and engineering project investigating whether a small Transformer language model, trained **completely from random initialization**, can learn to solve procedurally generated reasoning problems and generalize to problems it has never seen.

This is not a chatbot wrapper, RAG application, or fine-tuning project. The core language model is trained from scratch on synthetic data.

---

## What This Project Is

The core question:

> Can a ~8M parameter Transformer, trained from random initialization on synthetic arithmetic problems, learn to solve unseen reasoning problems using chain-of-thought reasoning?

We measure this with an automated verifier. No LLM is used as a judge. Every problem has a programmatically calculated exact answer.

---

## Why It Exists

Most published "small LLM" projects either:

1. Fine-tune an existing pretrained model (not from scratch)
2. Train on internet text (no control over what the model learned)
3. Measure loss, not task accuracy (loss does not equal reasoning ability)
4. Lack programmatic evaluation (human or LLM judges introduce noise)

This project builds a complete, reproducible pipeline where:
- The model starts from random weights
- The data is generated procedurally with exact solutions
- Evaluation is exact-match, no LLM judge
- Generalization is tested explicitly, not assumed

---

## Architecture

A decoder-only Transformer with pre-norm, learned absolute positional embeddings, and GELU activation.

```
Token Embedding (vocab_size × d_model)
    +
Positional Embedding (max_seq_len × d_model)
    ↓
[Decoder Block × n_layers]
    LayerNorm
    → CausalSelfAttention (Q/K/V/O projections, causal mask)
    + residual
    LayerNorm
    → FeedForward (Linear → GELU → Linear)
    + residual
    ↓
LayerNorm
    ↓
LM Head (d_model → vocab_size, weight-tied to token embedding)
```

### Size presets

| Preset | d_model | n_layers | n_heads | d_ff | Exact trainable params (weight-tied, vocab=4096) |
|--------|---------|----------|---------|------|---------|
| debug  | 64      | 2        | 4       | 128  | 97,856  |
| 1m     | 128     | 4        | 4       | 512  | ~0.6M   |
| 8m     | 256     | 8        | 8       | 1152 | **7,949,824** |
| 15m    | 384     | 8        | 8       | 1536 | ~12M    |
| 30m    | 512     | 10       | 8       | 2048 | ~24M    |

All parameter counts are **exact measured trainable counts** with weight tying enabled.  
Weight tying: `lm_head.weight` is the same tensor as `token_emb.weight`. This halves the cost of the output projection and improves gradient flow through the embedding layer.

The baseline experiment uses the **8m** preset.

---

## Tokenizer

Byte-level BPE tokenizer trained on the synthetic dataset using HuggingFace `tokenizers`. Vocabulary size 4096 tokens. Special tokens: `[PAD]`, `[BOS]`, `[EOS]`, `[UNK]`, `[SEP]`.

The tokenizer is trained fresh for each experiment and saved deterministically.

---

## Synthetic Data Generation

Problems are generated procedurally using `data/generators/arithmetic.py`. Every problem has a programmatically calculated exact answer.

### Problem families

| Family | Example |
|--------|---------|
| `single_op` | What is 37 + 58? → 95 |
| `multi_step` | Calculate: 4 + 3 * 2 - 1 → 9 |
| `comparison` | Which is greater: 5 + 3 or 12? → 12 |
| `percentage` | What is 25% of 80? → 20 |
| `ratio` | Ratio 3:2, total 50, first quantity? → 30 |
| `algebra_linear` | Solve for x: 2x + 3 = 11 → 4 |
| `word_problem` | Alice has 12 apples, buys 7 more. How many? → 19 |
| `number_sequence` | Next: 2, 4, 8, 16, ...? → 32 |

### Difficulty levels

| Level | Number range | Chain length | Notes |
|-------|-------------|--------------|-------|
| easy | 1–20 | 1–2 ops | single-digit arithmetic |
| medium | 1–100 | 2–4 ops | two-digit arithmetic |
| hard | 1–999 | 4–6 ops | three-digit, longer chains |

### Reasoning format

```
Problem: What is 37 + 58?
Reasoning:
Step 1: Add 37 and 58.
Step 2: 37 + 58 = 95.
Answer: 95
```

The reasoning format is defined once in `data/generators/arithmetic.py`. Experiments can compare the reasoning format vs. direct format (problem → answer only).

### Split separation

Train, validation, and test data are generated with non-overlapping random seeds:
- train: `seed + 0`
- val: `seed + 1_000_000`
- test: `seed + 2_000_000`

---

## Training Process

```bash
# Debug run (CPU, ~30 seconds)
python training/train.py --preset debug

# Baseline experiment (requires GPU)
python training/train.py --preset colab_small

# Custom config
python training/train.py --config experiments/configs/exp_01_baseline.json

# Resume
python training/train.py --preset colab_small \
    --resume experiments/results/colab_small_baseline/checkpoints/step_05000.pt
```

### Training features

- AdamW optimizer with cosine LR schedule and warmup
- Gradient accumulation (for effective larger batch sizes)
- Gradient clipping (norm 1.0)
- Mixed precision (FP16 on CUDA)
- Automatic checkpoint save on new validation best
- Periodic checkpoint save (every N steps)
- Mid-training reasoning accuracy evaluation

---

## Google Colab Training

Open `colab/train_small_reasoning_llm.ipynb`.

The notebook:
1. Installs dependencies
2. Clones the repository
3. Detects the GPU
4. Selects and runs a training preset
5. Evaluates the trained model
6. Exports the checkpoint for download

**Important:** Set `Runtime → Change runtime type → T4 GPU` before running.

Training presets designed for Colab:

| Preset | Model | Dataset | Approx. time |
|--------|-------|---------|-------------|
| `colab_small` | ~8M | 20k examples | ~1 hour (T4) |
| `colab_medium` | ~8M | 50k examples | ~2–3 hours (T4) |

Checkpoints are saved frequently. If a session disconnects, training can be resumed.

---

## Evaluation

### Programmatic verifier

The verifier extracts the model's answer and compares it to the exact expected answer numerically. No LLM is involved.

```python
verify(generated_text, expected_answer, expected_num)
# Returns:
# { "correct": True, "predicted": "95", "expected": "95", "pred_num": 95.0, ... }
```

### Standard benchmark

```bash
python evaluate.py --checkpoint experiments/results/exp/checkpoints/best.pt
```

Reports:
- Overall accuracy
- Per-family accuracy
- Per-difficulty accuracy

### Generalization benchmark

```bash
python evaluate.py --checkpoint ... --generalization
```

Tests 5 levels of generalization:

| Level | Description |
|-------|-------------|
| 1 | Same problem family, new values (same difficulty) |
| 2 | Same family, harder values |
| 3 | Mixed families |
| 4 | Longer reasoning chains |
| 5 | Compositional problems |

---

## Inference

```bash
# Interactive
python inference/generate.py --checkpoint experiments/.../best.pt

# Single problem
python inference/generate.py --checkpoint ... --problem "What is 37 + 58?"

# Batch from file
python inference/generate.py --checkpoint ... --problems_file problems.txt --output results.json
```

---

## Experiment Methodology

Each experiment saves:

```
experiments/results/<name>/
    train_config.json     # full training configuration
    model_config.json     # model architecture
    tokenizer/            # trained BPE tokenizer
    checkpoints/          # model checkpoints
    metrics.jsonl         # step-by-step training/validation metrics
    evaluation.json       # final benchmark results
    generalization.json   # 5-level generalization results
    samples.json          # sample model outputs for inspection
    summary.json          # condensed results
```

### Planned experiment roadmap

| Exp | Description | Status |
|-----|-------------|--------|
| 01 | Baseline: 8M + 20k medium + reasoning format | Config ready |
| 02 | Ablation: no chain-of-thought | Config ready |
| 03 | Curriculum: easy → medium → hard | Config ready |
| 04 | Larger dataset (50k) | Planned |
| 05 | Harder difficulty | Planned |
| 06 | Different model sizes (1M, 15M, 30M) | Planned |
| 07 | Verifier-based feedback | Planned |

---

## Current Results

*No training runs completed yet. This section will be updated after the first experiments.*

The baseline experiment (8M model, 20k training examples) is expected to establish the control result. All future experiments will be compared against this baseline.

---

## Known Limitations

- The model has a fixed context window (256–512 tokens) which limits multi-step reasoning chains.
- Small vocabulary (4096 tokens) may not efficiently represent all number formats.
- No curriculum learning in the baseline — easy and hard problems are mixed during training.
- The tokenizer is trained on the training set only — test distributions may use tokens with lower frequency.
- Arithmetic reasoning does not require world knowledge, so results should not be extrapolated to other domains.

---

## Running Tests

```bash
cd small-llm-lab
pytest tests/ -v
```

The most important test:

```bash
pytest tests/test_training.py::test_tiny_model_overfits_tiny_dataset -v
```

If this test fails, do not proceed to large training.

---

## Project Structure

```
small-llm-lab/
    model/
        config.py          # ModelConfig + size presets
        attention.py       # CausalSelfAttention
        transformer.py     # SmallTransformer
        tokenizer.py       # BPETokenizer

    data/
        generators/
            arithmetic.py  # procedural problem generator + corpus builder

    training/
        config.py          # TrainConfig + Colab presets
        dataset.py         # ReasoningDataset + DataLoader
        checkpoint.py      # save / load / find latest
        train.py           # full training pipeline + CLI

    evaluation/
        arithmetic.py      # programmatic verifier
        benchmark.py       # benchmark runner + generalization benchmark

    inference/
        generate.py        # interactive inference + CLI

    experiments/
        configs/           # experiment config JSONs
        results/           # training outputs (gitignored)

    tests/
        test_model.py
        test_tokenizer.py
        test_data_generator.py
        test_verifier.py
        test_training.py   # includes mandatory overfit test

    colab/
        train_small_reasoning_llm.ipynb

    evaluate.py            # top-level evaluation CLI
    requirements.txt
    README.md
```

---

## Requirements

```
torch>=2.1.0
numpy>=1.24.0
tokenizers>=0.15.0
tqdm>=4.66.0
pyyaml>=6.0
pytest>=7.4.0
rich>=13.0.0
```

Python 3.10+.

---

## Engineering Principles

1. No pretrained weights — the model starts from random initialization.
2. No LLM as evaluator — only exact programmatic verification.
3. Loss is not the primary metric — reasoning accuracy on unseen problems is.
4. Every experiment is reproducible via a config file.
5. Test before scaling — the overfit test must pass before full training.
6. Failures are recorded, not hidden.
