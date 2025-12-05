#!/usr/bin/env python3
"""
Evaluation utilities for generative (open-ended) QA tasks.
Uses LLM-as-judge for semantic equivalence evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.llm_judge import LLMJudge, simple_string_match
from utils.nli_evaluator import NLIEvaluator
import logging


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GenerativeEvaluator:
    """
    Evaluator for open-ended generative QA tasks.
    Supports Bayesian uncertainty quantification via multiple samples.
    
    Evaluation methods:
    - LLM Judge: Uses LLM to check semantic equivalence
    - NLI (Mutual Entailment): Uses NLI model to check bidirectional entailment
    - String Matching: Simple exact/fuzzy string matching
    """
    
    def __init__(
        self,
        eval_method: str = "llm_judge",
        judge_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        nli_model: str = "microsoft/deberta-v3-large-mnli",
        nli_threshold: float = 0.5,
        device: str = "cuda",
        judge_gpu_id: int = None,
    ):
        """
        Initialize generative evaluator.
        
        Args:
            eval_method: Evaluation method - "llm_judge", "nli", or "string_match"
            judge_model: Model to use as LLM judge
            nli_model: Model to use for NLI-based evaluation
            nli_threshold: Probability threshold for NLI entailment (default: 0.5)
            device: Device for judge model
            judge_gpu_id: Specific GPU ID for judge model (e.g., 1 for cuda:1). 
                         If None, uses device_map="auto"
        """
        self.eval_method = eval_method.lower()
        self.nli_threshold = nli_threshold
        
        if self.eval_method == "llm_judge":
            self.use_llm_judge = True
            self.judge = LLMJudge(
                model_name=judge_model, 
                device=device,
                gpu_id=judge_gpu_id
            )
            self.nli_evaluator = None
            if judge_gpu_id is not None:
                logging.info(f"Initialized LLM judge for evaluation on GPU {judge_gpu_id}")
            else:
                logging.info("Initialized LLM judge for evaluation")
        
        elif self.eval_method == "nli":
            self.use_llm_judge = False
            self.judge = None
            self.nli_evaluator = NLIEvaluator(
                model_name=nli_model,
                device=device,
                gpu_id=judge_gpu_id,
                load_in_8bit=True  # Use 8-bit for efficiency
            )
            logging.info(f"Initialized NLI evaluator (mutual entailment) with threshold={nli_threshold}")
        
        else:  # string_match
            self.use_llm_judge = False
            self.judge = None
            self.nli_evaluator = None
            logging.info("Using simple string matching for evaluation")
    
    def evaluate_single(
        self,
        generated_answer: str,
        reference_answer: str,
        question: str = "",
        context: Optional[str] = None,
    ) -> int:
        """
        Evaluate a single generated answer.
        
        Args:
            generated_answer: Model's generated answer
            reference_answer: Ground truth answer
            question: The question (used by LLM judge)
            context: Optional context (for SQUAD-style questions)
        
        Returns:
            1 if correct, 0 if incorrect
        """
        if self.eval_method == "llm_judge":
            return self.judge.judge_answer(
                question=question,
                reference_answer=reference_answer,
                candidate_answer=generated_answer,
                context=context
            )
        elif self.eval_method == "nli":
            correctness, _ = self.nli_evaluator.evaluate_single(
                generated_answer=generated_answer,
                reference_answer=reference_answer,
                threshold=self.nli_threshold
            )
            return correctness
        else:  # string_match
            return simple_string_match(reference_answer, generated_answer)
    
    def evaluate_batch(
        self,
        generated_answers: List[str],
        reference_answers: List[str],
        questions: List[str],
        contexts: Optional[List[str]] = None,
    ) -> List[int]:
        """
        Evaluate a batch of generated answers.
        
        Args:
            generated_answers: List of model's generated answers
            reference_answers: List of ground truth answers
            questions: List of questions
            contexts: Optional list of contexts
        
        Returns:
            List of judgments (1 = correct, 0 = incorrect)
        """
        if self.eval_method == "llm_judge":
            return self.judge.judge_batch(
                questions=questions,
                reference_answers=reference_answers,
                candidate_answers=generated_answers,
                contexts=contexts
            )
        elif self.eval_method == "nli":
            correctness_list, _ = self.nli_evaluator.evaluate_batch(
                generated_answers=generated_answers,
                reference_answers=reference_answers,
                threshold=self.nli_threshold
            )
            return correctness_list
        else:  # string_match
            return [
                simple_string_match(ref, gen)
                for ref, gen in zip(reference_answers, generated_answers)
            ]
    
    def evaluate_with_uncertainty(
        self,
        generated_samples: List[List[str]],
        reference_answers: List[str],
        questions: List[str],
        contexts: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate with Bayesian uncertainty quantification.
        For each question, we have N sampled answers from the posterior.
        
        Args:
            generated_samples: List of [N samples per question]
            reference_answers: Ground truth answers
            questions: Questions asked
            contexts: Optional contexts
        
        Returns:
            Dictionary with metrics:
                - accuracy: Overall accuracy (majority vote)
                - confidence: Average posterior probability of being correct
                - consistency: Average agreement among samples
        """
        n_questions = len(generated_samples)
        n_samples = len(generated_samples[0]) if n_questions > 0 else 0
        
        if contexts is None:
            contexts = [None] * n_questions
        
        correct_votes = []
        confidences = []
        
        for i in range(n_questions):
            samples = generated_samples[i]
            reference = reference_answers[i]
            question = questions[i]
            context = contexts[i]
            
            # Judge each sample
            judgments = []
            for sample in samples:
                judgment = self.evaluate_single(
                    generated_answer=sample,
                    reference_answer=reference,
                    question=question,
                    context=context
                )
                judgments.append(judgment)
            
            # Confidence = fraction of samples judged correct
            confidence = np.mean(judgments)
            confidences.append(confidence)
            
            # Majority vote: correct if >50% of samples are correct
            is_correct = 1 if confidence > 0.5 else 0
            correct_votes.append(is_correct)
        
        # Compute metrics
        accuracy = np.mean(correct_votes)
        avg_confidence = np.mean(confidences)
        
        # Consistency: How much do samples agree? (1 = all same, 0 = 50/50 split)
        consistency = np.mean([
            1 - 2 * abs(0.5 - conf) for conf in confidences
        ])
        
        return {
            "accuracy": accuracy,
            "confidence": avg_confidence,
            "consistency": consistency,
            "n_samples": n_samples,
            "n_questions": n_questions
        }


def compute_generation_metrics(
    generated_texts: List[str],
    reference_texts: List[str]
) -> Dict[str, float]:
    """
    Compute simple statistics about generated text.
    
    Args:
        generated_texts: Generated answers
        reference_texts: Reference answers
    
    Returns:
        Dictionary with statistics
    """
    gen_lengths = [len(text.split()) for text in generated_texts]
    ref_lengths = [len(text.split()) for text in reference_texts]
    
    return {
        "avg_gen_length": np.mean(gen_lengths),
        "avg_ref_length": np.mean(ref_lengths),
        "length_ratio": np.mean(gen_lengths) / (np.mean(ref_lengths) + 1e-6),
    }


def extract_questions_and_contexts(batch, dataset_name: str) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """
    Extract questions, answers, and contexts from a batch based on dataset type.
    
    Args:
        batch: Batch from dataloader
        dataset_name: Name of the dataset
    
    Returns:
        Tuple of (questions, answers, contexts)
    """
    # This is dataset-specific and would need to be customized
    # based on the actual batch structure from your datasets
    
    questions = []
    answers = []
    contexts = []
    
    # Placeholder - you'll need to implement based on your actual batch structure
    # For now, returning empty lists
    logging.warning("extract_questions_and_contexts needs dataset-specific implementation")
    
    return questions, answers, None if not contexts else contexts
