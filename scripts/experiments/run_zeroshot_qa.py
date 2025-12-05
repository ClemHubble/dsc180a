#!/usr/bin/env python3
"""
Zero-shot evaluation script for generative QA.
Evaluates the base model WITHOUT any LoRA adapters.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project to path
project_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_path))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm

from dataset.utils.dsets import get_dataset
from utils.nli_evaluator import NLIEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot generative QA evaluation")
    
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
    parser.add_argument("--output-dir", type=str, default="results_zeroshot",
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


def main():
    args = parse_args()
    
    # Setup accelerator
    accelerator = Accelerator()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Setup logging
    output_dir = Path(args.output_dir) / args.dataset / f"zeroshot-{args.model}-seed{args.seed}"
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
    
    logging.info(f"🚀 Starting zero-shot evaluation")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Eval method: {args.eval_method}")
    
    # Load model WITHOUT LoRA/PEFT
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
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=args.hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    logging.info(f"✅ Model loaded (no LoRA adapters)")
    
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
    
    # Evaluate
    correct = 0
    total = 0
    all_results = []
    
    model.eval()
    with torch.no_grad():
        for idx, example in enumerate(tqdm(test_data, desc="Evaluating")):
            # Extract question and answer based on dataset format
            if args.dataset == "TriviaQA":
                question = example["question"]
                reference_answer = example["answer"]["value"]
            elif args.dataset in ["ARC-Challenge", "ARC-Easy"]:
                question = example["question"]
                choices = example["choices"]["text"]
                answer_key = example["answerKey"]
                # Map answer key to choice
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
            
            # Evaluate
            if args.eval_method == "nli":
                is_correct, details = nli_evaluator.check_mutual_entailment(
                    generated_text, 
                    reference_answer,
                    threshold=args.nli_threshold
                )
            else:
                # Simple string matching
                is_correct = reference_answer.lower() in generated_text.lower()
            
            correct += int(is_correct)
            total += 1
            
            all_results.append({
                "question": question,
                "reference": reference_answer,
                "generated": generated_text,
                "correct": is_correct
            })
            
            if (idx + 1) % 100 == 0:
                logging.info(f"Progress: {idx+1}/{len(test_data)}, Accuracy: {correct/total:.4f}")
    
    # Final results
    accuracy = correct / total if total > 0 else 0.0
    
    logging.info(f"\n{'='*50}")
    logging.info(f"FINAL RESULTS (Zero-Shot)")
    logging.info(f"{'='*50}")
    logging.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    logging.info(f"Results saved to: {output_dir}")
    
    # Save detailed results
    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "args": vars(args),
            "samples": all_results
        }, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()

