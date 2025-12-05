#!/usr/bin/env python3
"""
Natural Language Inference (NLI) based evaluator for semantic equivalence.
Uses mutual entailment to check if two answers mean the same thing.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
import logging

log = logging.getLogger(__name__)


class NLIEvaluator:
    """
    Evaluates semantic equivalence using Natural Language Inference (NLI).
    Checks for mutual entailment between generated and reference answers.
    """
    
    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device: str = "cuda",
        gpu_id: int = None,
        load_in_8bit: bool = False,
    ):
        """
        Initialize NLI evaluator.
        
        Args:
            model_name: HuggingFace model for NLI (default: DeBERTa-v3-large-mnli)
                       Alternatives: "roberta-large-mnli", "facebook/bart-large-mnli"
            device: Device to load model on
            gpu_id: Specific GPU ID (for multi-GPU setups)
            load_in_8bit: Whether to load model in 8-bit precision
        """
        self.device = device
        if gpu_id is not None:
            self.device = f"cuda:{gpu_id}"
        
        log.info(f"Loading NLI model: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if load_in_8bit:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map={"": self.device} if gpu_id is not None else "auto"
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to(self.device)
        
        self.model.eval()
        
        # Label mapping for MNLI models
        # 0: entailment, 1: neutral, 2: contradiction
        self.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
        
        log.info(f"NLI model loaded successfully on {self.device}")
    
    def check_entailment(
        self,
        premise: str,
        hypothesis: str,
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Check if premise entails hypothesis.
        
        Args:
            premise: The premise text
            hypothesis: The hypothesis text
            threshold: Probability threshold for entailment (default: 0.5)
        
        Returns:
            (is_entailment, entailment_probability)
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entailment_prob = probs[0, self.label2id["entailment"]].item()
        
        return entailment_prob >= threshold, entailment_prob
    
    def check_mutual_entailment(
        self,
        text1: str,
        text2: str,
        threshold: float = 0.5
    ) -> Tuple[bool, dict]:
        """
        Check if two texts mutually entail each other.
        
        Args:
            text1: First text (e.g., model answer)
            text2: Second text (e.g., reference answer)
            threshold: Probability threshold for entailment
        
        Returns:
            (mutual_entailment, details_dict)
            - mutual_entailment: True if both directions entail
            - details_dict: Contains individual entailment results and probabilities
        """
        # Check text1 → text2 (does model answer entail reference?)
        inputs_forward = self.tokenizer(
            text1,
            text2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs_forward = self.model(**inputs_forward)
            probs_forward = torch.softmax(outputs_forward.logits, dim=-1)
            forward_entailment_prob = probs_forward[0, self.label2id["entailment"]].item()
        
        # Check text2 → text1 (does reference entail model answer?)
        inputs_backward = self.tokenizer(
            text2,
            text1,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs_backward = self.model(**inputs_backward)
            probs_backward = torch.softmax(outputs_backward.logits, dim=-1)
            backward_entailment_prob = probs_backward[0, self.label2id["entailment"]].item()
        
        forward_entails = forward_entailment_prob >= threshold
        backward_entails = backward_entailment_prob >= threshold
        mutual_entailment = forward_entails and backward_entails
        
        details = {
            "forward_entailment": forward_entails,
            "backward_entailment": backward_entails,
            "forward_prob": forward_entailment_prob,
            "backward_prob": backward_entailment_prob,
            "mutual_entailment": mutual_entailment,
            "min_prob": min(forward_entailment_prob, backward_entailment_prob),
            "avg_prob": (forward_entailment_prob + backward_entailment_prob) / 2
        }
        
        return mutual_entailment, details
    
    def evaluate_single(
        self,
        generated_answer: str,
        reference_answer: str,
        threshold: float = 0.5
    ) -> Tuple[int, dict]:
        """
        Evaluate a single generated answer against reference.
        
        Args:
            generated_answer: Model-generated answer
            reference_answer: Ground truth answer
            threshold: Entailment probability threshold
        
        Returns:
            (correctness, details)
            - correctness: 1 if correct (mutual entailment), 0 otherwise
            - details: Dictionary with entailment probabilities
        """
        mutual, details = self.check_mutual_entailment(
            generated_answer,
            reference_answer,
            threshold
        )
        return 1 if mutual else 0, details
    
    def evaluate_batch(
        self,
        generated_answers: List[str],
        reference_answers: List[str],
        threshold: float = 0.5,
        verbose: bool = False,
        batch_size: int = 32
    ) -> Tuple[List[int], List[dict]]:
        """
        Evaluate a batch of generated answers against references with batched inference.
        
        Args:
            generated_answers: List of model-generated answers
            reference_answers: List of reference answers
            threshold: Entailment probability threshold
            verbose: Whether to log detailed results
            batch_size: Batch size for NLI model inference (default: 32)
        
        Returns:
            (correctness_list, details_list)
            - correctness_list: List of 1 (correct) or 0 (incorrect)
            - details_list: List of detail dicts for each pair
        """
        n_pairs = len(generated_answers)
        
        # Prepare all forward and backward pairs
        forward_premises = []
        forward_hypotheses = []
        backward_premises = []
        backward_hypotheses = []
        
        for gen, ref in zip(generated_answers, reference_answers):
            # Forward: does generated answer entail reference?
            forward_premises.append(gen)
            forward_hypotheses.append(ref)
            # Backward: does reference entail generated answer?
            backward_premises.append(ref)
            backward_hypotheses.append(gen)
        
        # Batch process all forward entailments
        forward_probs = []
        for i in range(0, n_pairs, batch_size):
            batch_premises = forward_premises[i:i+batch_size]
            batch_hypotheses = forward_hypotheses[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                entailment_probs = probs[:, self.label2id["entailment"]].cpu().tolist()
                forward_probs.extend(entailment_probs)
        
        # Batch process all backward entailments
        backward_probs = []
        for i in range(0, n_pairs, batch_size):
            batch_premises = backward_premises[i:i+batch_size]
            batch_hypotheses = backward_hypotheses[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                entailment_probs = probs[:, self.label2id["entailment"]].cpu().tolist()
                backward_probs.extend(entailment_probs)
        
        # Compute mutual entailment for all pairs
        correctness = []
        details = []
        
        for i, (fwd_prob, bwd_prob) in enumerate(zip(forward_probs, backward_probs)):
            forward_entails = fwd_prob >= threshold
            backward_entails = bwd_prob >= threshold
            mutual = forward_entails and backward_entails
            
            detail = {
                "forward_entailment": forward_entails,
                "backward_entailment": backward_entails,
                "forward_prob": fwd_prob,
                "backward_prob": bwd_prob,
                "mutual_entailment": mutual,
                "min_prob": min(fwd_prob, bwd_prob),
                "avg_prob": (fwd_prob + bwd_prob) / 2
            }
            
            correctness.append(1 if mutual else 0)
            details.append(detail)
            
            if verbose:
                log.info(f"Example {i+1}:")
                log.info(f"  Generated: {generated_answers[i][:100]}...")
                log.info(f"  Reference: {reference_answers[i][:100]}...")
                log.info(f"  Forward: {fwd_prob:.3f}, Backward: {bwd_prob:.3f}")
                log.info(f"  Mutual entailment: {mutual}")
        
        return correctness, details


