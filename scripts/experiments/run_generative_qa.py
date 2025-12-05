#!/usr/bin/env python3
"""
Generative QA Experiment Runner

A wrapper script to run generative QA experiments with BLoB (Gaussian/Laplace) methods.
Supports TriviaQA, TruthfulQA, and other open-ended QA datasets.

Usage:
    python scripts/experiments/run_generative_qa.py \\
        --model llama-1b \\
        --dataset TruthfulQA \\
        --method gaussian \\
        --eval-method nli \\
        --seed 1

    # Compare Gaussian vs Laplace
    python scripts/experiments/run_generative_qa.py \\
        --model llama-8b \\
        --dataset TriviaQA \\
        --compare-methods \\
        --seeds 1 2 3
"""
import subprocess
import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Get absolute path to project root dynamically
# This script is in: bayesian-peft/scripts/dsmlp/run_generative_qa.py
# So project root is: ../../ from script location
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent  # scripts/dsmlp/
BASE_DIR = SCRIPT_DIR.parent.parent  # bayesian-peft/

# Verify we're in the right place
if not (BASE_DIR / "run" / "main.py").exists():
    # Try to find it relative to current working directory
    cwd = Path.cwd()
    if (cwd / "run" / "main.py").exists():
        BASE_DIR = cwd
    elif (cwd.parent / "run" / "main.py").exists():
        BASE_DIR = cwd.parent
    elif (cwd.parent.parent / "run" / "main.py").exists():
        BASE_DIR = cwd.parent.parent
    else:
        print(f"ERROR: Cannot find project root. run/main.py not found.", file=sys.stderr)
        print(f"Script location: {SCRIPT_PATH}", file=sys.stderr)
        print(f"Tried BASE_DIR: {BASE_DIR}", file=sys.stderr)
        print(f"Current directory: {Path.cwd()}", file=sys.stderr)
        sys.exit(1)

print(f"INFO: Project root: {BASE_DIR}", file=sys.stderr)

# Model mappings
MODELS = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
}

# Dataset mappings for open-ended QA
DATASETS = {
    "TruthfulQA": "TruthfulQA",
    "SQuADV2": "SQuADV2", 
    "TriviaQA": "TriviaQA",
    "qa_all": ["TruthfulQA", "SQuADV2", "TriviaQA"],
}

# Method mappings
METHODS = {
    "gaussian": "blob",
    "laplace": "blob_laplace",
    "mle": "mle",
}


def setup_logging(log_dir="logs_generative_qa"):
    """Setup logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging to: {log_file}")
    return log_file


def run_experiment(
    model_id: str,
    dataset: str,
    method: str,
    steps: int = 2000,
    batch_size: int = None,
    seed: int = 1,
    bayes_beta: float = 0.01,
    bayes_eval_n_samples: int = 5,
    max_new_tokens: int = 50,
    max_seq_len: int = 2048,
    generation_temperature: float = 0.7,
    eval_method: str = "string_match",
    use_llm_judge: bool = False,
    nli_model: str = "microsoft/deberta-v3-large-mnli",
    nli_threshold: float = 0.5,
    judge_gpu_id: int = None,
    subset_size: int = 0,
    hf_token: str = None,
    wandb_key: str = None,
    no_wandb: bool = True,
):
    """Run a single generative QA experiment."""
    
    # Auto-set batch size based on model if not specified
    if batch_size is None:
        if "8B" in model_id or "7B" in model_id:
            batch_size = 2
        else:
            batch_size = 4
    
    # Build experiment name
    model_short = [k for k, v in MODELS.items() if v == model_id][0]
    method_name = [k for k, v in METHODS.items() if v == method][0]
    experiment_name = f"{method_name}-{model_short}-{dataset}-steps{steps}-seed{seed}"
    
    logging.info("="*60)
    logging.info(f"Starting Generative QA Experiment")
    logging.info("="*60)
    logging.info(f"Model: {model_id}")
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Method: {method} ({method_name} noise)")
    logging.info(f"Steps: {steps}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Seed: {seed}")
    logging.info(f"Bayes Beta: {bayes_beta}")
    logging.info(f"Eval Samples: {bayes_eval_n_samples}")
    logging.info(f"Max New Tokens: {max_new_tokens}")
    logging.info(f"Max Sequence Length: {max_seq_len}")
    logging.info(f"Evaluation Method: {eval_method}")
    if eval_method == "nli":
        logging.info(f"NLI Model: {nli_model}")
        logging.info(f"NLI Threshold: {nli_threshold}")
    if judge_gpu_id is not None:
        logging.info(f"Judge/NLI GPU ID: {judge_gpu_id}")
    if subset_size > 0:
        logging.info(f"Using subset size: {subset_size}")
    logging.info(f"Experiment Name: {experiment_name}")
    logging.info("="*60)
    
    # Build command with all arguments
    cmd = [
        "python",
        str(BASE_DIR / "run" / "main.py"),
        "--model", model_id,
        "--model-type", "causallm",  # Required: model type
        "--dataset", dataset,
        "--dataset-type", "oedataset",  # Open-ended dataset for generation
        "--modelwrapper", method,
        "--bayes-beta", str(bayes_beta),
        "--bayes-eval-n-samples", str(bayes_eval_n_samples),
        "--max-train-steps", str(steps),
        "--eval-per-steps", str(steps + 1000),  # Only evaluate at the end
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--lr", "5e-5",  # Lower LR for stability with generative QA
        "--lora-r", "8",
        "--lora-alpha", "16",
        "--lora-dropout", "0.1",
        "--testing-set", "val",
        "--log-path", experiment_name,
        "--max-seq-len", str(max_seq_len),
    ]
    
    # Add subset size if specified
    if subset_size > 0:
        cmd.extend(["--subset-size", str(subset_size)])
    
    # Add generative evaluation flags
    cmd.extend([
        "--eval-mode", "generative",
        "--max-new-tokens", str(max_new_tokens),
        "--generation-temperature", str(generation_temperature),
        "--generative-eval-method", eval_method,
    ])
    
    # NLI-specific parameters
    if eval_method == "nli":
        cmd.extend([
            "--nli-model", nli_model,
            "--nli-threshold", str(nli_threshold),
        ])
    
    # Backward compatibility with --use-llm-judge flag
    if use_llm_judge:
        cmd.append("--use-llm-judge")
    
    if judge_gpu_id is not None:
        cmd.extend(["--judge-gpu-id", str(judge_gpu_id)])
    
    # HuggingFace authentication via environment variable
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    if no_wandb:
        cmd.append("--nowand")
    elif wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
    
    logging.info(f"Command: {' '.join(cmd)}")
    
    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            check=True,
            capture_output=False,
            text=True
        )
        logging.info(f"✓ Experiment completed successfully: {experiment_name}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ Experiment failed: {experiment_name}")
        logging.error(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run generative QA experiments on DSMLP cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and dataset
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Model to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset name (TruthfulQA, SQuADV2, TriviaQA)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Multiple datasets to run (e.g., --datasets TruthfulQA SQuADV2)"
    )
    parser.add_argument(
        "--all-qa",
        action="store_true",
        help="Run all QA datasets (TruthfulQA, SQuADV2, TriviaQA)"
    )
    
    # Method(s)
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=list(METHODS.keys()),
        help="Single noise distribution method"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=list(METHODS.keys()),
        default=None,
        help="Multiple methods to run (e.g., --methods gaussian laplace)"
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Run both Gaussian and Laplace for comparison"
    )
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if not specified)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Multiple seeds (e.g., --seeds 1 2 3)")
    parser.add_argument("--bayes-beta", type=float, default=0.01, help="Bayesian beta parameter")
    parser.add_argument("--bayes-eval-n-samples", type=int, default=5, help="Number of evaluation samples")
    
    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length for tokenization (default: 2048)")
    parser.add_argument("--generation-temperature", type=float, default=0.7, help="Generation temperature")
    
    # Evaluation method
    parser.add_argument(
        "--eval-method",
        type=str,
        default="string_match",
        choices=["llm_judge", "nli", "string_match"],
        help="Evaluation method: llm_judge (LLM-as-judge), nli (mutual entailment), or string_match (default)"
    )
    parser.add_argument("--use-llm-judge", action="store_true", default=False, help="[Deprecated] Use --eval-method llm_judge instead")
    parser.add_argument("--no-llm-judge", action="store_false", dest="use_llm_judge", help="[Deprecated] Use --eval-method string_match instead")
    
    # NLI parameters
    parser.add_argument(
        "--nli-model",
        type=str,
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        help="NLI model for mutual entailment evaluation (default: DeBERTa-v3-large fine-tuned on NLI)"
    )
    parser.add_argument(
        "--nli-threshold",
        type=float,
        default=0.5,
        help="Entailment probability threshold for NLI (default: 0.5)"
    )
    
    # Multi-GPU support
    parser.add_argument("--judge-gpu-id", type=int, default=None, help="GPU ID for judge/NLI model (e.g., 1 for cuda:1). Useful for multi-GPU setups.")
    
    # Dataset sampling
    parser.add_argument("--subset-size", type=int, default=0, help="Use a subset of the dataset (0 = use full dataset). Useful for quick testing.")
    
    # Authentication
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--wandb-key", type=str, default=None, help="Weights & Biases API key")
    parser.add_argument("--nowand", action="store_true", default=True, help="Disable W&B logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Log start
    logging.info("="*60)
    logging.info("DSMLP Generative QA Experiment Runner")
    logging.info("="*60)
    logging.info(f"Arguments: {vars(args)}")
    logging.info("="*60)
    
    # Get model ID
    model_id = MODELS[args.model]
    
    # Determine datasets to run
    datasets = []
    if args.all_qa:
        datasets = DATASETS["qa_all"]
        logging.info(f"Running ALL QA datasets: {datasets}")
    elif args.datasets:
        datasets = args.datasets
        logging.info(f"Running multiple datasets: {datasets}")
    elif args.dataset:
        datasets = [args.dataset]
        logging.info(f"Running single dataset: {args.dataset}")
    else:
        logging.error("Must specify --dataset, --datasets, or --all-qa")
        return 1
    
    # Determine methods to run
    methods = []
    if args.compare_methods:
        methods = ["gaussian", "laplace"]
        logging.info("Running comparison: Gaussian vs Laplace")
    elif args.methods:
        methods = args.methods
        logging.info(f"Running multiple methods: {methods}")
    elif args.method:
        methods = [args.method]
        logging.info(f"Running single method: {args.method}")
    else:
        # Default to Gaussian
        methods = ["gaussian"]
        logging.info("Using default method: Gaussian")
    
    # Determine seeds to run
    seeds = args.seeds if args.seeds else [args.seed]
    logging.info(f"Running with seeds: {seeds}")
    
    # Calculate total experiments
    total_experiments = len(datasets) * len(methods) * len(seeds)
    logging.info(f"Total experiments to run: {total_experiments}")
    logging.info("="*60)
    
    # Run all experiments
    success_count = 0
    experiment_num = 0
    
    for dataset in datasets:
        for method_name in methods:
            method = METHODS[method_name]
            for seed in seeds:
                experiment_num += 1
                logging.info(f"\n{'='*60}")
                logging.info(f"Experiment {experiment_num}/{total_experiments}")
                logging.info(f"Dataset: {dataset}, Method: {method_name}, Seed: {seed}")
                logging.info(f"{'='*60}")
                
                success = run_experiment(
                    model_id=model_id,
                    dataset=dataset,
                    method=method,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    seed=seed,
                    bayes_beta=args.bayes_beta,
                    bayes_eval_n_samples=args.bayes_eval_n_samples,
                    max_new_tokens=args.max_new_tokens,
                    max_seq_len=args.max_seq_len,
                    generation_temperature=args.generation_temperature,
                    eval_method=args.eval_method,
                    use_llm_judge=args.use_llm_judge,
                    nli_model=args.nli_model,
                    nli_threshold=args.nli_threshold,
                    judge_gpu_id=args.judge_gpu_id,
                    subset_size=args.subset_size,
                    hf_token=args.hf_token,
                    wandb_key=args.wandb_key,
                    no_wandb=args.nowand,
                )
                
                if success:
                    success_count += 1
                
                logging.info(f"Progress: {success_count}/{experiment_num} successful")
    
    # Final summary
    logging.info("\n" + "="*60)
    logging.info("FINAL EXPERIMENT SUMMARY")
    logging.info("="*60)
    logging.info(f"Model: {args.model}")
    logging.info(f"Datasets: {datasets}")
    logging.info(f"Methods: {methods}")
    logging.info(f"Seeds: {seeds}")
    logging.info(f"Completed: {success_count}/{total_experiments} experiments")
    logging.info("="*60)
    
    if success_count == total_experiments:
        logging.info("✓ All experiments completed successfully!")
        return 0
    else:
        logging.warning(f"✗ {total_experiments - success_count} experiment(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())