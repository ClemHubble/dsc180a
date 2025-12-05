# Bayesian LORA through Backpropagation(BLoB) PEFT: Uncertainty Quantification for Large Language Models

This repository contains implementations of Bayesian uncertainty quantification methods for Parameter-Efficient Fine-Tuning (PEFT) of Large Language Models, supporting both **multiple-choice QA** and **open-ended generative QA** tasks.

## 📖 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Running Experiments](#running-experiments)
   - [Multiple-Choice QA (MCQ)](#multiple-choice-qa-mcq)
   - [Open-Ended Generative QA](#open-ended-generative-qa)
6. [Evaluation Methods](#evaluation-methods)
7. [Arguments Reference](#arguments-reference)
8. [Key Considerations](#key-considerations)
9. [References](#references)

---

## Overview

This codebase implements several uncertainty quantification methods for LLMs:

| Method | Description | Wrapper |
|--------|-------------|---------|
| **BLoB (Gaussian)** | Bayesian Low-Rank Adaptation with Gaussian noise on LoRA weights | `blob` |
| **BLoB (Laplace)** | Bayesian Low-Rank Adaptation with Laplace noise on LoRA weights | `blob_laplace` |
| **MC Dropout** | Monte Carlo Dropout for uncertainty estimation | `mcdropout` |
| **Deep Ensemble** | Ensemble of independently trained models | `deepensemble` |
| **MLE** | Maximum Likelihood Estimation baseline (no uncertainty) | `mle` |
| **MAP** | Maximum A Posteriori baseline | `map` |

### Supported Datasets

**Multiple-Choice QA:**
- ARC-Challenge, ARC-Easy
- OBQA (OpenBookQA)
- PIQA, WinoGrande, HellaSwag

**Open-Ended Generative QA:**
- TriviaQA
- TruthfulQA

---

## Installation

```bash
# Create conda environment
conda create --name bayesian-peft python=3.10
conda activate bayesian-peft

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install transformers datasets evaluate accelerate bitsandbytes
pip install jaxtyping torchmetrics setproctitle peft wandb nltk scikit-learn

# For NLI evaluation (recommended for generative QA)
pip install scipy
```

### HuggingFace Authentication (for gated models like Llama)

```bash
# Set your HuggingFace token
export HF_TOKEN="your_huggingface_token"

# Or pass via command line
--hf-token YOUR_TOKEN
```

---

## Project Structure

```
bayesian-peft/
├── models/                     # Model implementations
│   ├── causallm.py            # Causal LM (Llama, Qwen, etc.)
│   └── seqcls.py              # Sequence classification (RoBERTa)
├── modelwrappers/             # Uncertainty quantification methods
│   ├── blob.py                # BLoB with Gaussian noise
│   ├── blob_laplace.py        # BLoB with Laplace noise
│   ├── mcdropout.py           # MC Dropout
│   ├── deepensemble.py        # Deep Ensemble
│   ├── mle.py                 # MLE baseline
│   └── wrapperbase.py         # Base wrapper class
├── dataset/                   # Dataset loaders
│   └── utils/dsets.py         # Dataset definitions
├── run/                       # Main execution
│   ├── main.py                # Entry point
│   ├── evaluation.py          # MCQ evaluation
│   └── evaluation_generative.py # Generative QA evaluation
├── utils/                     # Utilities
│   ├── args.py                # Argument definitions
│   ├── nli_evaluator.py       # NLI-based evaluation
│   └── llm_judge.py           # LLM-as-judge evaluation
├── scripts/dsmlp/             # Experiment runner scripts
│   ├── run_generative_qa.py   # Generative QA experiments
│   ├── run_bayesian_zeroshot_qa.py # Zero-shot Bayesian evaluation
│   └── run_experiment.py      # General experiment runner
└── requirements.txt           # Dependencies
```

---

## Quick Start

### Run a Simple MCQ Experiment (ARC-Challenge with Gaussian BLoB)

```bash
python run/main.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --model-type causallm \
    --dataset ARC-Challenge \
    --dataset-type mcdataset \
    --modelwrapper blob \
    --max-train-steps 2000 \
    --batch-size 4 \
    --seed 1 \
    --nowand
```

### Run a Simple Generative QA Experiment (TriviaQA with NLI evaluation)

```bash
python run/main.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --model-type causallm \
    --dataset TriviaQA \
    --dataset-type oedataset \
    --modelwrapper blob \
    --eval-mode generative \
    --generative-eval-method nli \
    --max-new-tokens 50 \
    --max-train-steps 2000 \
    --batch-size 2 \
    --seed 1 \
    --nowand
```

---

## Running Experiments

### Multiple-Choice QA (MCQ)

For MCQ datasets, the model predicts probabilities over fixed answer choices (A, B, C, D).

```bash
python run/main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model-type causallm \
    --dataset ARC-Challenge \           # ARC-Challenge, ARC-Easy, obqa, PIQA, etc.
    --dataset-type mcdataset \          # Multi-choice dataset
    --modelwrapper blob \               # blob, blob_laplace, mcdropout, mle
    --max-train-steps 5000 \
    --eval-per-steps 1000 \
    --batch-size 4 \
    --lr 1e-4 \
    --lora-r 8 \
    --lora-alpha 16 \
    --bayes-beta 0.2 \                  # KL divergence weight
    --bayes-eval-n-samples 5 \          # Bayesian samples for evaluation
    --seed 1 \
    --log-path my-experiment \
    --nowand
```

**Key MCQ Arguments:**
- `--dataset-type mcdataset`: Use for multiple-choice datasets
- `--eval-mode classification` (default): Evaluates based on predicted class probabilities
- Uncertainty is derived directly from output probabilities

### Open-Ended Generative QA

For open-ended QA, the model generates free-form text answers that are evaluated semantically.

```bash
python run/main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model-type causallm \
    --dataset TriviaQA \                # TriviaQA, TruthfulQA
    --dataset-type oedataset \          # Open-ended dataset
    --modelwrapper blob \
    --eval-mode generative \            # REQUIRED for generative QA
    --generative-eval-method nli \      # nli, llm_judge, or string_match
    --max-new-tokens 50 \               # Max tokens to generate
    --generation-temperature 0.7 \
    --max-seq-len 2048 \                # Important for long contexts 
    --max-train-steps 2000 \
    --batch-size 2 \
    --bayes-eval-n-samples 5 \
    --nli-model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli \
    --nli-threshold 0.5 \
    --seed 1 \
    --nowand
```

**Key Generative QA Arguments:**
- `--dataset-type oedataset`: Use for open-ended datasets
- `--eval-mode generative`: REQUIRED - enables text generation evaluation
- `--generative-eval-method`: Choose evaluation method (see below)
- `--max-new-tokens`: Limit generation length (50 recommended)
- `--max-seq-len`: Important for datasets with long contexts 
### Using the Convenience Script

For easier experiment management, use the provided Python wrapper:

```bash
# Single experiment
python scripts/dsmlp/run_generative_qa.py \
    --model llama-1b \
    --dataset TriviaQA \
    --method gaussian \
    --eval-method nli \
    --seed 1

# Compare Gaussian vs Laplace
python scripts/dsmlp/run_generative_qa.py \
    --model llama-8b \
    --dataset TriviaQA \
    --compare-methods \
    --seeds 1 2 3

# Run all QA datasets
python scripts/dsmlp/run_generative_qa.py \
    --model llama-1b \
    --all-qa \
    --method gaussian \
    --eval-method nli
```

### Zero-Shot Bayesian Evaluation (No Fine-Tuning)

Evaluate uncertainty with Bayesian perturbations but WITHOUT fine-tuning:

```bash
python scripts/dsmlp/run_bayesian_zeroshot_qa.py \
    --model llama-1b \
    --dataset TriviaQA \
    --method gaussian \           # gaussian or laplace
    --n-samples 5 \               # Bayesian samples per question
    --sigma 0.05 \                # Perturbation strength
    --eval-method nli \
    --subset-size 500             # For quick testing
```

---

## Evaluation Methods

### For Multiple-Choice QA

Evaluation is straightforward: compare predicted label to ground truth. Uncertainty metrics (ECE, NLL, Brier) are computed directly from output probabilities.

### For Open-Ended Generative QA

Three evaluation methods are available:

| Method | Flag | Description | Speed | Accuracy |
|--------|------|-------------|-------|----------|
| **NLI (Recommended)** | `--generative-eval-method nli` | Uses NLI model to check mutual entailment between generated and reference answers | Fast | High |
| **LLM Judge** | `--generative-eval-method llm_judge` | Uses another LLM to judge semantic equivalence | Slow | Highest |
| **String Match** | `--generative-eval-method string_match` | Simple substring matching | Fastest | Low |

#### NLI Evaluation (Recommended)

Uses a Natural Language Inference model to check if the generated answer and reference answer mutually entail each other.

```bash
--generative-eval-method nli \
--nli-model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli \
--nli-threshold 0.5
```

**How it works:**
1. Check if generated answer → reference answer (entailment)
2. Check if reference answer → generated answer (entailment)
3. Answer is correct if BOTH directions show entailment

**Pros:** Fast, consistent, works well for factual QA
**Cons:** May miss paraphrases with very different wording

#### LLM Judge Evaluation

Uses a separate LLM to judge if two answers are semantically equivalent.

```bash
--generative-eval-method llm_judge \
--llm-judge-model meta-llama/Llama-3.1-8B-Instruct \
--judge-gpu-id 1  # Use a separate GPU for the judge
```

**Pros:** Most accurate, handles paraphrases well
**Cons:** Slow (requires LLM inference per evaluation), needs extra GPU memory

#### Multi-GPU Setup for LLM Judge

When using LLM judge with limited memory, put the judge on a separate GPU:

```bash
--judge-gpu-id 1  # Judge on GPU 1, main model on GPU 0
```

---

## Arguments Reference

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | required | HuggingFace model ID |
| `--model-type` | str | required | `causallm` or `seqcls` |
| `--dataset` | str | required | Dataset name |
| `--dataset-type` | str | required | `mcdataset` or `oedataset` |
| `--modelwrapper` | str | required | `blob`, `blob_laplace`, `mcdropout`, `mle` |

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-train-steps` | int | 0 | Total training steps |
| `--batch-size` | int | 4 | Batch size |
| `--lr` | float | 1e-4 | Learning rate |
| `--seed` | int | 1 | Random seed |
| `--eval-per-steps` | int | 500 | Evaluation frequency |

### LoRA Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lora-r` | int | 8 | LoRA rank |
| `--lora-alpha` | int | 16 | LoRA alpha scaling |
| `--lora-dropout` | float | 0 | LoRA dropout |

### Bayesian Arguments (BLoB)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bayes-beta` | float | 0.2 | KL divergence weight |
| `--bayes-eval-n-samples` | int | 1 | Samples during training eval |
| `--bayes-eval-n-samples-final` | int | 10 | Samples for final eval |
| `--bayes-eps` | float | 0.05 | Epsilon for variance |
| `--bayes-inference-notsample` | flag | - | Use mean only (no sampling) |

### Generative QA Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--eval-mode` | str | classification | `classification` or `generative` |
| `--generative-eval-method` | str | string_match | `nli`, `llm_judge`, or `string_match` |
| `--max-new-tokens` | int | 50 | Max tokens to generate |
| `--generation-temperature` | float | 0.7 | Sampling temperature |
| `--max-seq-len` | int | 300 | Max input sequence length |

### NLI Evaluation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--nli-model` | str | DeBERTa-v3-large | NLI model for evaluation |
| `--nli-threshold` | float | 0.5 | Entailment probability threshold |
| `--judge-gpu-id` | int | None | GPU for NLI/judge model |

---

## Key Considerations

### MCQ vs Generative QA: Key Differences

| Aspect | Multiple-Choice QA | Open-Ended Generative QA |
|--------|-------------------|-------------------------|
| **Output** | Probability over choices (A, B, C, D) | Free-form text |
| **Correctness** | Argmax matches ground truth | Semantic equivalence check |
| **Confidence** | Softmax probability of prediction | NLI entailment score or sample consistency |
| **Uncertainty** | Directly from logits | Aggregated across Bayesian samples |
| **Speed** | Fast (single forward pass) | Slow (autoregressive generation + NLI) |
| **dataset-type** | `mcdataset` | `oedataset` |
| **eval-mode** | `classification` | `generative` |

### Uncertainty Metrics

All methods output these calibration metrics:

- **Accuracy**: Percentage of correct predictions
- **ECE (Expected Calibration Error)**: Measures calibration quality (lower is better)
- **NLL (Negative Log-Likelihood)**: Log probability of correct answer
- **Brier Score**: Mean squared error of probabilities

For Bayesian methods:
- **Predictive Entropy**: Disagreement across Bayesian samples (higher = more uncertain)

### Memory Considerations

| Model | MCQ Batch Size | Generative Batch Size | GPU Memory |
|-------|---------------|----------------------|------------|
| Llama-3.2-1B | 8 | 4 | ~16GB |
| Llama-3.1-8B | 2-4 | 1-2 | ~40GB |
| Qwen2.5-7B | 2-4 | 1-2 | ~40GB |

### Dataset-Specific Notes

**TriviaQA:**
- Answers can have multiple valid forms
- NLI evaluation handles this well

**TruthfulQA:**
- Tests for truthful vs. common misconceptions
- Lower baseline accuracy expected

---

## References

**BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models**
Yibin Wang*, Haizhou Shi*, Ligong Han, Dimitris Metaxas, Hao Wang
*NeurIPS 2024*
[[Paper](https://arxiv.org/abs/2406.11675)] [[OpenReview](https://openreview.net/forum?id=MaDykgj4Ru)]

**Training-Free Bayesianization for Low-Rank Adapters of Large Language Models**
Haizhou Shi*, Yibin Wang*, Ligong Han, Huan Zhang, Hao Wang
*ICLR Workshop 2025*
[[Paper](https://arxiv.org/abs/2412.05723)]

```bibtex
@article{wang2024blob,
  title={BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models},
  author={Wang, Yibin and Shi, Haizhou and Han, Ligong and Metaxas, Dimitris and Wang, Hao},
  journal={arXiv preprint arXiv:2406.11675},
  year={2024}
}
```

---

## License

See [LICENSE](LICENSE) for details.
