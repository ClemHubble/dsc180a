#!/usr/bin/env python3
"""
BLoB Experiment Runner

A wrapper script to run BLoB (Bayesian Low-Rank Adaptation) experiments
for uncertainty quantification on multiple-choice QA datasets.

Usage:
    # Quick test
    python scripts/experiments/run_experiment.py --quick-test

    # Single experiment
    python scripts/experiments/run_experiment.py \\
        --model llama-1b \\
        --method gaussian \\
        --dataset ARC-Challenge \\
        --seed 1

    # Full experiment suite
    python scripts/experiments/run_experiment.py \\
        --full \\
        --model llama-1b \\
        --method laplace
"""

import argparse
import subprocess
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Get the base directory (bayesian-peft root)
SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/experiments/
BASE_DIR = SCRIPT_DIR.parent.parent  # bayesian-peft/

# Model configurations
MODELS = {
    "llama-1b": "meta-llama/Llama-3.2-1B",
    "llama-8b": "meta-llama/Meta-Llama-3.1-8B",
    "qwen2-7b": "Qwen/Qwen2-7B",  # Qwen2 base (7B, not 8B)
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",  # Qwen2.5 instruct (7B)
}

# Dataset configurations
DATASETS = {
    "quick": ["winogrande_s"],  # 640 samples, ~20-30 min
    "validation": ["winogrande_s", "ARC-Challenge"],  # ~1-2 hours
    "science": ["ARC-Easy", "ARC-Challenge", "obqa"],  # Science Q&A only, ~3-4 hours
    "full": ["winogrande_s", "ARC-Challenge", "ARC-Easy", "winogrande_m", "obqa", "boolq"],  # Full paper
}

# Method configurations
METHODS = {
    "gaussian": "blob",
    "laplace": "blob_laplace",
    "mle": "mle",
}


def setup_logging(log_dir="logs_wrapper"):
    """Setup logging to both file and console"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("DSMLPExperiment")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file


def run_training(
    model_name,
    model_path,
    dataset,
    method,
    seed,
    batch_size,
    max_steps,
    wandb_project,
    logger,
    args,
    output_dir=None,
    gpu_id=0,
):
    """Run a single training experiment"""
    
    # Construct experiment name
    method_short = "gaussian" if method == "blob" else "laplace" if method == "blob_laplace" else method
    exp_name = f"{method_short}-{model_name}-{dataset}-steps{max_steps}-seed{seed}"
    
    logger.info("="*60)
    logger.info(f"Running: {exp_name}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Method: {method}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max steps: {max_steps}")
    logger.info("="*60)
    
    # Build command with absolute path to main.py
    main_script = os.path.join(BASE_DIR, "run", "main.py")
    if not os.path.exists(main_script):
        logger.error(f"ERROR: Cannot find main.py at {main_script}")
        logger.error(f"BASE_DIR is: {BASE_DIR}")
        logger.error("Make sure you're in the bayesian-peft directory")
        return False
    
    cmd = [
        "python", main_script,
        "--dataset-type", "mcdataset",
        "--dataset", dataset,
        "--model-type", "causallm",
        "--model", model_path,
        "--modelwrapper", method,
        "--lr", "1e-4",
        "--batch-size", str(batch_size),
        "--opt", "adamw",
        "--warmup-ratio", "0.06",
        "--max-seq-len", "300",
        "--seed", str(seed),
        "--wandb-name", exp_name,
        "--wandb-project", wandb_project,
        "--apply-classhead-lora",
        "--lora-r", "8",
        "--lora-alpha", "16",
        "--lora-dropout", "0",
        "--log-path", exp_name,
        "--max-train-steps", str(max_steps),
        "--eval-per-steps", str(max_steps + 1000),  # Eval at end
    ]
    
    # Add nowand flag if specified
    if hasattr(args, 'nowand') and args.nowand:
        cmd.append("--nowand")
        logger.info("WandB logging disabled (--nowand flag set)")
    
    # Add BLoB-specific arguments
    if method in ["blob", "blob_laplace"]:
        cmd.extend([
            "--bayes-klreweighting",
            "--bayes-eps", "0.05",
            "--bayes-beta", "0.2",
            "--bayes-gamma", "8",
            "--bayes-kllr", "0.01",
            "--bayes-datasetrescaling",
            "--bayes-train-n-samples", "1",
            "--bayes-eval-n-samples", "10",
            "--bayes-eval-n-samples-final", "10",
        ])
        
        # Add Laplace-specific argument
        if method == "blob_laplace":
            cmd.extend(["--laplace-scale-factor", "1.0"])
    
    # Set CUDA device, authentication tokens, and working directory
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Pass authentication tokens if available
    if "HF_TOKEN" in env:
        logger.info("Using HuggingFace token from environment")
    if "WANDB_API_KEY" in env:
        logger.info("Using WandB API key from environment")
    
    # Run command
    logger.info("Starting training...")
    logger.info(f"Working directory: {BASE_DIR}")
    logger.info(f"Command: {' '.join(cmd[:5])}...")
    
    try:
        # Run with real-time output, using cwd to set working directory
        process = subprocess.Popen(
            cmd, 
            env=env,
            cwd=BASE_DIR,  # Set working directory for subprocess
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            print(line, end='')  # Print to console
            sys.stdout.flush()
        
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"✅ Completed: {exp_name}")
            return True
        else:
            logger.error(f"❌ Failed: {exp_name} (exit code: {process.returncode})")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed: {exp_name}")
        logger.error(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning(f"⚠️  Interrupted: {exp_name}")
        process.terminate()
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run BLoB experiments on DSMLP cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (single run, small dataset)
  python scripts/experiments/run_experiment.py --quick-test
  
  # Specific configuration
  python scripts/experiments/run_experiment.py --model llama-1b --method laplace --dataset winogrande_s --seed 1
  
  # Validation (2 datasets, 3 seeds)
  python scripts/experiments/run_experiment.py --validation --model llama-1b --method laplace
  
  # Full experiment (all datasets, 3 seeds)
  python scripts/experiments/run_experiment.py --full --model llama-1b --method laplace
  
  # Compare Gaussian vs Laplace
  python scripts/experiments/run_experiment.py --compare --model llama-1b --dataset winogrande_s
        """
    )
    
    # Experiment presets
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument("--quick-test", action="store_true",
                             help="Quick test: 1 dataset, 1 seed, 1000 steps (~20-30 min)")
    preset_group.add_argument("--validation", action="store_true",
                             help="Validation: 2 datasets, 3 seeds, 3000 steps (~2-3 hours)")
    preset_group.add_argument("--science", action="store_true",
                             help="Science Q&A: 3 datasets (ARC-Easy, ARC-Challenge, obqa), 3 seeds, 5000 steps (~3-5 hours)")
    preset_group.add_argument("--full", action="store_true",
                             help="Full: 6 datasets, 3 seeds, 5000 steps (~8-12 hours)")
    preset_group.add_argument("--compare", action="store_true",
                             help="Compare Gaussian vs Laplace on specified dataset")
    
    # Model selection
    parser.add_argument("--model", type=str, default="llama-1b",
                       choices=list(MODELS.keys()),
                       help="Model to use (default: llama-1b)")
    
    # Method selection
    parser.add_argument("--method", type=str, default="gaussian",
                       choices=list(METHODS.keys()),
                       help="Training method (default: gaussian)")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str,
                       help="Specific dataset (overrides preset)")
    
    # Training parameters
    parser.add_argument("--seed", type=int, nargs="+",
                       help="Random seed(s) (default: [1, 2, 3])")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size (default: 4, reduce to 2 for 8B models)")
    parser.add_argument("--max-steps", type=int,
                       help="Max training steps (overrides preset)")
    
    # Output
    parser.add_argument("--wandb-project", type=str,
                       help="WandB project name (auto-generated if not specified)")
    parser.add_argument("--gpu-id", type=int, default=0,
                       help="GPU ID (default: 0, should be 0 on DSMLP single-GPU containers)")
    
    # Authentication (for batch jobs)
    parser.add_argument("--hf-token", type=str,
                       help="HuggingFace token (for Llama models). Get from https://huggingface.co/settings/tokens")
    parser.add_argument("--wandb-key", type=str,
                       help="WandB API key (for logging). Get from https://wandb.ai/authorize")
    parser.add_argument("--nowand", action="store_true",
                       help="Disable WandB logging (useful if wandb not installed)")
    
    args = parser.parse_args()
    
    # Setup logging in the base directory
    log_dir = os.path.join(BASE_DIR, "logs_wrapper")
    logger, log_file = setup_logging(log_dir)
    logger.info("="*60)
    logger.info("DSMLP Experiment Runner Started")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*60)
    
    # Setup authentication tokens if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
        logger.info("✅ HuggingFace token set from argument")
    elif os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        logger.info("✅ HuggingFace token found in environment")
    elif args.model in ["llama-1b", "llama-8b"]:
        logger.warning("⚠️  No HuggingFace token provided!")
        logger.warning("   Llama models require authentication.")
        logger.warning("   Use --hf-token argument or run 'huggingface-cli login' first.")
        logger.warning("   Get token from: https://huggingface.co/settings/tokens")
    
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
        logger.info("✅ WandB API key set from argument")
    elif os.environ.get("WANDB_API_KEY"):
        logger.info("✅ WandB API key found in environment")
    else:
        logger.warning("⚠️  No WandB API key provided!")
        logger.warning("   Results won't be logged unless you've run 'wandb login'.")
        logger.warning("   Use --wandb-key argument or run 'wandb login' first.")
        logger.warning("   Get key from: https://wandb.ai/authorize")
    
    # Determine configuration based on presets or explicit args
    if args.quick_test:
        datasets = ["winogrande_s"]
        seeds = [1]
        max_steps = 1000
        methods = [METHODS[args.method]]
    elif args.validation:
        datasets = DATASETS["validation"]
        seeds = args.seed or [1, 2, 3]
        max_steps = 3000
        methods = [METHODS[args.method]]
    elif args.science:
        datasets = DATASETS["science"]
        seeds = args.seed or [1, 2, 3]
        max_steps = 5000
        methods = [METHODS[args.method]]
    elif args.full:
        datasets = DATASETS["full"]
        seeds = args.seed or [1, 2, 3]
        max_steps = 5000
        methods = [METHODS[args.method]]
    elif args.compare:
        datasets = [args.dataset] if args.dataset else ["winogrande_s"]
        seeds = args.seed or [1, 2, 3]
        max_steps = args.max_steps or 5000
        methods = [METHODS["gaussian"], METHODS["laplace"]]  # Both methods
    else:
        # Custom configuration
        datasets = [args.dataset] if args.dataset else ["winogrande_s"]
        seeds = args.seed or [1]
        max_steps = args.max_steps or 5000
        methods = [METHODS[args.method]]
    
    # Get model info
    model_name = args.model
    model_path = MODELS[model_name]
    
    # Auto-generate WandB project name
    if args.wandb_project:
        wandb_project = args.wandb_project
    else:
        method_names = "-".join([m.replace("blob_laplace", "Laplace").replace("blob", "Gaussian") 
                                for m in set(methods)])
        wandb_project = f"BLoB-{method_names}-{model_name}-DSMLP"
    
    # Summary
    logger.info("="*60)
    logger.info("DSMLP Experiment Configuration")
    logger.info("="*60)
    logger.info(f"Model: {model_name} ({model_path})")
    logger.info(f"Methods: {', '.join(methods)}")
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"WandB project: {wandb_project}")
    logger.info(f"Total runs: {len(methods) * len(datasets) * len(seeds)}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*60)
    
    # Confirm if interactive
    if sys.stdin.isatty():
        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Aborted by user.")
            sys.exit(0)
    
    logger.info("\nStarting experiments...")
    
    # Run experiments
    total_runs = len(methods) * len(datasets) * len(seeds)
    completed = 0
    failed = 0
    run_number = 0
    
    for method in methods:
        for dataset in datasets:
            for seed in seeds:
                run_number += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"Run {run_number}/{total_runs}")
                logger.info(f"{'='*60}")
                
                success = run_training(
                    model_name=model_name,
                    model_path=model_path,
                    dataset=dataset,
                    method=method,
                    seed=seed,
                    batch_size=args.batch_size,
                    max_steps=max_steps,
                    wandb_project=wandb_project,
                    logger=logger,
                    args=args,
                    gpu_id=args.gpu_id,
                )
                
                if success:
                    completed += 1
                else:
                    failed += 1
                
                logger.info(f"Progress: {completed} completed, {failed} failed, {total_runs - run_number} remaining")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Experiment Summary")
    logger.info("="*60)
    logger.info(f"Total runs: {total_runs}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {completed/total_runs*100:.1f}%")
    logger.info(f"Log file: {log_file}")
    logger.info(f"WandB project: https://wandb.ai → {wandb_project}")
    logger.info("="*60)
    
    if failed > 0:
        logger.error("Some experiments failed. Check logs above.")
        sys.exit(1)
    else:
        logger.info("✅ All experiments completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

