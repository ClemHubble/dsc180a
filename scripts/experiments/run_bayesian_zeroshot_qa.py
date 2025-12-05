#!/usr/bin/env python3
"""
Bayesian zero-shot evaluation script for generative QA.
Applies Bayesian perturbations to base model WITHOUT fine-tuning.

This evaluates: base_model + random_perturbations (no training)
Useful for understanding uncertainty quantification independent of fine-tuning.
"""

import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project to path
project_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_path))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

from utils.nli_evaluator import NLIEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian zero-shot generative QA evaluation")
    
    # Model args
    parser.add_argument("--model", type=str, required=True, 
                       choices=["llama-8b", "llama-1b", "llama-3b", "qwen2-1.5b", "qwen2-7b"],
                       help="Model to evaluate")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Override model path (optional)")
    
    # Dataset args
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["TriviaQA", "SQuAD", "ARC-Challenge", "ARC-Easy", "obqa", "TruthfulQA"],
                       help="Dataset to evaluate on")
    parser.add_argument("--subset-size", type=int, default=None,
                       help="Number of samples to evaluate (None = full dataset)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for evaluation")
    
    # Bayesian sampling args
    parser.add_argument("--method", type=str, default="gaussian",
                       choices=["gaussian", "laplace"],
                       help="Perturbation distribution")
    parser.add_argument("--n-samples", type=int, default=5,
                       help="Number of Bayesian samples per question")
    parser.add_argument("--sigma", type=float, default=0.05,
                       help="Standard deviation for perturbations")
    
    # LoRA args (for adding perturbable parameters)
    parser.add_argument("--lora-r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                       help="LoRA dropout")
    
    # Generation args
    parser.add_argument("--max-new-tokens", type=int, default=32,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0.0 = greedy)")
    
    # Evaluation args
    parser.add_argument("--eval-method", type=str, default="nli",
                       choices=["nli", "string_match"],
                       help="Evaluation method")
    parser.add_argument("--nli-model", type=str, 
                       default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                       help="NLI model for evaluation")
    parser.add_argument("--nli-threshold", type=float, default=0.5,
                       help="NLI entailment threshold")
    parser.add_argument("--judge-gpu-id", type=int, default=None,
                       help="GPU ID for NLI judge")
    
    # Output args
    parser.add_argument("--output-dir", type=str, default="results_bayesian_zeroshot",
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # HuggingFace token
    parser.add_argument("--hf-token", type=str, default=None,
                       help="HuggingFace token for gated models")
    
    return parser.parse_args()


def get_model_path(model_name):
    """Map model name to HuggingFace path."""
    model_paths = {
        "llama-8b": "meta-llama/Llama-3.2-3B-Instruct",
        "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
        "qwen2-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2-7b": "Qwen/Qwen2.5-7B-Instruct",
    }
    return model_paths.get(model_name, model_name)


def perturb_lora_weights(model, sigma, method="gaussian"):
    """
    Apply random perturbations to LoRA weights.
    
    Args:
        model: PEFT model with LoRA adapters
        sigma: Standard deviation of perturbations
        method: 'gaussian' or 'laplace'
    """
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            if method == "gaussian":
                noise = torch.randn_like(param) * sigma
            else:  # laplace
                # Laplace: sample from exponential and randomly flip sign
                noise = torch.empty_like(param).exponential_(1.0 / sigma)
                noise = noise * torch.sign(torch.randn_like(param))
            
            param.data.add_(noise)


def reset_lora_weights(model):
    """
    Reset LoRA weights to their initial (random) state.
    We'll store initial weights and restore them.
    """
    pass  # Will be handled by re-initializing before each sample


def main():
    args = parse_args()
    
    # Setup accelerator
    accelerator = Accelerator()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup logging
    output_dir = Path(args.output_dir) / args.dataset / f"bayesian-zeroshot-{args.method}-{args.model}-seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "log.txt"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"🚀 Starting Bayesian zero-shot evaluation (NO FINE-TUNING)")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Method: {args.method}")
    logging.info(f"N Samples: {args.n_samples}")
    logging.info(f"Sigma: {args.sigma}")
    logging.info(f"Eval method: {args.eval_method}")
    
    # Load base model
    model_path = args.model_path or get_model_path(args.model)
    logging.info(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=args.hf_token,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=args.hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Add LoRA adapters (NOT trained, just for perturbation)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(base_model, lora_config)
    
    logging.info(f"✅ Model loaded with LoRA adapters (NOT trained, for perturbation only)")
    logging.info(f"Trainable parameters: {model.print_trainable_parameters()}")
    
    # Store initial LoRA weights (random initialization)
    initial_lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            initial_lora_state[name] = param.data.clone()
    
    # Load dataset
    from datasets import load_dataset
    
    dataset_map = {
        "TriviaQA": ("trivia_qa", "unfiltered.nocontext"),
        "SQuAD": ("squad",),
        "ARC-Challenge": ("allenai/ai2_arc", "ARC-Challenge"),
        "ARC-Easy": ("allenai/ai2_arc", "ARC-Easy"),
        "obqa": ("allenai/openbookqa", "main"),
        "TruthfulQA": ("truthful_qa", "generation"),
    }
    
    if args.dataset in dataset_map:
        dataset_args = dataset_map[args.dataset]
        dataset = load_dataset(*dataset_args)
        
        # Get test split
        if "test" in dataset:
            test_data = dataset["test"]
        elif "validation" in dataset:
            test_data = dataset["validation"]
        else:
            test_data = dataset["train"]
        
        if args.subset_size:
            test_data = test_data.select(range(min(args.subset_size, len(test_data))))
        
        logging.info(f"Test samples: {len(test_data)}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Initialize NLI evaluator
    if args.eval_method == "nli":
        nli_evaluator = NLIEvaluator(
            model_name=args.nli_model,
            device="cuda",
            gpu_id=args.judge_gpu_id,
            load_in_8bit=True
        )
        logging.info(f"✅ NLI evaluator loaded")
    else:
        nli_evaluator = None
    
    # Evaluate with Bayesian sampling
    correct = 0
    total = 0
    all_results = []
    all_entropies = []
    all_nli_probs = []
    
    model.eval()
    
    print(f"\n🔥 Starting evaluation with {args.n_samples} Bayesian samples per question...")
    print(f"This will be SLOW because we generate {args.n_samples}x predictions and evaluate each with NLI\n")
    
    for idx, example in enumerate(tqdm(test_data, desc="Evaluating")):
        # Extract question and answer based on dataset format
        if args.dataset == "TriviaQA":
            question = example["question"]
            reference_answer = example["answer"]["value"]
        elif args.dataset in ["ARC-Challenge", "ARC-Easy"]:
            question = example["question"]
            choices = example["choices"]["text"]
            answer_key = example["answerKey"]
            choice_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
            answer_idx = choice_map.get(answer_key, int(answer_key) - 1 if answer_key.isdigit() else 0)
            reference_answer = choices[answer_idx]
        elif args.dataset == "obqa":
            question = example["question_stem"]
            choices = example["choices"]["text"]
            answer_key = example["answerKey"]
            choice_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            answer_idx = choice_map.get(answer_key, 0)
            reference_answer = choices[answer_idx]
        else:
            continue
        
        # Format prompt
        if "llama" in args.model.lower():
            prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        # Generate multiple samples with different perturbations
        sample_predictions = []
        sample_nli_details = []
        
        with torch.no_grad():
            for sample_idx in range(args.n_samples):
                # Reset LoRA to initial state
                for name, param in model.named_parameters():
                    if 'lora_' in name and param.requires_grad:
                        param.data.copy_(initial_lora_state[name])
                
                # Apply perturbation
                perturb_lora_weights(model, args.sigma, args.method)
                
                # Generate
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature if args.temperature > 0 else None,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )
                
                generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                generated_text = generated_text.strip()
                sample_predictions.append(generated_text)
        
        # Batch evaluate all samples with NLI (much faster!)
        if args.eval_method == "nli":
            # Prepare all pairs for batch evaluation
            batch_texts1 = sample_predictions
            batch_texts2 = [reference_answer] * len(sample_predictions)
            
            # Batch NLI evaluation
            batch_results = nli_evaluator.evaluate_batch(
                batch_texts1, 
                batch_texts2,
                threshold=args.nli_threshold,
                batch_size=min(32, args.n_samples)
            )
            
            sample_correct = [result[0] for result in batch_results]
            sample_nli_details = [result[1] for result in batch_results]
        else:
            # Simple string matching
            sample_correct = [reference_answer.lower() in pred.lower() for pred in sample_predictions]
            sample_nli_details = [{}] * len(sample_predictions)
        
        # Majority vote for final prediction
        is_correct = sum(sample_correct) > (args.n_samples / 2)
        correct += int(is_correct)
        total += 1
        
        # Calculate predictive entropy (uncertainty)
        p_correct = sum(sample_correct) / args.n_samples
        p_incorrect = 1 - p_correct
        if p_correct > 0 and p_incorrect > 0:
            entropy = -p_correct * np.log(p_correct) - p_incorrect * np.log(p_incorrect)
        else:
            entropy = 0.0
        all_entropies.append(entropy)
        
        # Store NLI probabilities
        if sample_nli_details[0]:
            avg_nli_prob = np.mean([d.get('avg_prob', 0.5) for d in sample_nli_details])
            all_nli_probs.append(avg_nli_prob)
        
        all_results.append({
            "question": question,
            "reference": reference_answer,
            "samples": sample_predictions,
            "sample_correct": sample_correct,
            "majority_correct": is_correct,
            "entropy": entropy
        })
        
        if (idx + 1) % 10 == 0:
            logging.info(f"Progress: {idx+1}/{len(test_data)}, Accuracy: {correct/total:.4f}, Avg Entropy: {np.mean(all_entropies):.4f}")
            print(f"✓ Processed {idx+1}/{len(test_data)} questions", flush=True)
    
    # Final results
    accuracy = correct / total if total > 0 else 0.0
    avg_entropy = np.mean(all_entropies) if all_entropies else 0.0
    
    # Calculate ECE (Expected Calibration Error)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = [0] * n_bins
    bin_correct = [0] * n_bins
    bin_confidence = [0] * n_bins
    
    for idx, result in enumerate(all_results):
        confidence = sum(result['sample_correct']) / args.n_samples
        is_correct = result['majority_correct']
        
        bin_idx = min(int(confidence * n_bins), n_bins - 1)
        bin_counts[bin_idx] += 1
        bin_correct[bin_idx] += is_correct
        bin_confidence[bin_idx] += confidence
    
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_acc = bin_correct[i] / bin_counts[i]
            bin_conf = bin_confidence[i] / bin_counts[i]
            ece += (bin_counts[i] / total) * abs(bin_acc - bin_conf)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"FINAL RESULTS (Bayesian Zero-Shot, NO FINE-TUNING)")
    logging.info(f"{'='*60}")
    logging.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    logging.info(f"ECE: {ece:.4f}")
    logging.info(f"Avg Predictive Entropy: {avg_entropy:.4f}")
    logging.info(f"Method: {args.method}")
    logging.info(f"Sigma: {args.sigma}")
    logging.info(f"N Samples: {args.n_samples}")
    logging.info(f"Results saved to: {output_dir}")
    
    # Save detailed results
    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "ece": ece,
            "avg_entropy": avg_entropy,
            "correct": correct,
            "total": total,
            "args": vars(args),
            "samples": all_results
        }, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"Accuracy: {accuracy:.2%} / ECE: {ece:.4f}")
    print(f"This used NO FINE-TUNING, only Bayesian perturbations of random LoRA weights")


if __name__ == "__main__":
    main()


