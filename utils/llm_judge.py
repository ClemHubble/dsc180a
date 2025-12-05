#!/usr/bin/env python3
"""
LLM-as-Judge evaluation for open-ended QA tasks.
Uses Llama-3.1-8B-Instruct to judge semantic equivalence between generated and reference answers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import List, Tuple, Optional
import logging


class LLMJudge:
    """
    LLM-based judge for evaluating semantic equivalence of QA answers.
    Returns binary correctness: 1 (correct) or 0 (incorrect).
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        load_in_8bit: bool = True,
        gpu_id: int = None,
    ):
        """
        Initialize LLM judge.
        
        Args:
            model_name: HuggingFace model ID for the judge model
            device: Device to load model on (e.g., "cuda", "cuda:0", "cuda:1")
            load_in_8bit: Whether to use 8-bit quantization
            gpu_id: Specific GPU ID to use (e.g., 0, 1). If None, uses device_map="auto"
        """
        self.device = device
        self.model_name = model_name
        
        # Determine device mapping
        if gpu_id is not None:
            # Force model to specific GPU
            device_map = {"": f"cuda:{gpu_id}"}
            self.device = f"cuda:{gpu_id}"
            logging.info(f"Loading LLM judge: {model_name} on GPU {gpu_id}")
        else:
            # Let HuggingFace decide (default behavior)
            device_map = "auto"
            logging.info(f"Loading LLM judge: {model_name} with device_map='auto'")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optional 8-bit quantization
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                trust_remote_code=True,
            )
        
        self.model.eval()
        logging.info(f"LLM judge loaded successfully on {self.device}")
    
    def _create_judge_prompt(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        context: Optional[str] = None
    ) -> str:
        """
        Create a prompt for the LLM judge to evaluate semantic equivalence.
        
        Args:
            question: The question that was asked
            reference_answer: The ground truth answer
            candidate_answer: The model's generated answer
            context: Optional context (for SQUAD-style questions)
        
        Returns:
            Formatted prompt for the judge
        """
        if context:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert answer evaluator. Judge if two answers mean the same thing.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}
Question: {question}

Ground truth answer: {reference_answer}
Student answer: {candidate_answer}

Does the student answer match the ground truth? Reply with exactly one word: CORRECT or INCORRECT<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert answer evaluator. Judge if two answers mean the same thing.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Ground truth answer: {reference_answer}
Student answer: {candidate_answer}

Does the student answer match the ground truth? Reply with exactly one word: CORRECT or INCORRECT<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    def _parse_judgment(self, response: str) -> int:
        """
        Parse the LLM's response to extract binary judgment.
        
        Args:
            response: Raw text response from the LLM
        
        Returns:
            1 if correct, 0 if incorrect
        """
        response = response.strip().upper()
        
        # Remove common prefixes that models add
        response = response.replace("ANSWER:", "").replace("RESPONSE:", "").strip()
        
        # Check first word/token for quick decision
        first_word = response.split()[0] if response.split() else ""
        
        if first_word == "CORRECT":
            return 1
        elif first_word == "INCORRECT":
            return 0
        
        # Look for explicit correct/incorrect markers anywhere in response
        if "INCORRECT" in response:
            return 0
        elif "CORRECT" in response:
            return 1
        
        # Fallback: look for affirmative patterns
        affirmative_patterns = [
            r'\bYES\b', r'\bTRUE\b', r'\bRIGHT\b',
            r'\bEQUIVALENT\b', r'\bMATCH(ES)?\b', r'\bAGREE'
        ]
        
        negative_patterns = [
            r'\bNO\b', r'\bFALSE\b', r'\bWRONG\b',
            r'\bDIFFER', r'\bNOT\s+MATCH'
        ]
        
        # Check negative patterns first
        for pattern in negative_patterns:
            if re.search(pattern, response):
                return 0
        
        # Then check affirmative
        for pattern in affirmative_patterns:
            if re.search(pattern, response):
                return 1
        
        # Default to incorrect if uncertain
        logging.warning(f"Ambiguous judge response: {response[:100]}")
        return 0
    
    @torch.no_grad()
    def judge_answer(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        context: Optional[str] = None,
        max_new_tokens: int = 20
    ) -> int:
        """
        Judge whether a candidate answer is correct.
        
        Args:
            question: The question
            reference_answer: Ground truth answer
            candidate_answer: Model's generated answer
            context: Optional context
            max_new_tokens: Max tokens for judge response
        
        Returns:
            1 if correct, 0 if incorrect
        """
        # Handle empty or very short candidate answers
        if not candidate_answer or len(candidate_answer.strip()) < 2:
            return 0
        
        prompt = self._create_judge_prompt(
            question, reference_answer, candidate_answer, context
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for consistency
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        judgment = self._parse_judgment(response)
        
        return judgment
    
    @torch.no_grad()
    def judge_batch(
        self,
        questions: List[str],
        reference_answers: List[str],
        candidate_answers: List[str],
        contexts: Optional[List[str]] = None,
        max_new_tokens: int = 20
    ) -> List[int]:
        """
        Judge a batch of answers (batched for efficiency).
        
        Args:
            questions: List of questions
            reference_answers: List of ground truth answers
            candidate_answers: List of model's generated answers
            contexts: Optional list of contexts
            max_new_tokens: Max tokens for judge responses
        
        Returns:
            List of judgments (1 = correct, 0 = incorrect)
        """
        if contexts is None:
            contexts = [None] * len(questions)
        
        # Create prompts for all examples
        prompts = [
            self._create_judge_prompt(q, r, c, ctx)
            for q, r, c, ctx in zip(questions, reference_answers, candidate_answers, contexts)
        ]
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate responses
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode and parse all responses
        judgments = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(
                output[inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            judgment = self._parse_judgment(response)
            judgments.append(judgment)
        
        return judgments


def simple_string_match(reference: str, candidate: str) -> int:
    """
    Simple baseline: normalized string matching.
    Can be used as a fast alternative to LLM judge for debugging.
    
    Args:
        reference: Ground truth answer
        candidate: Model's generated answer
    
    Returns:
        1 if match, 0 otherwise
    """
    def normalize(s):
        # Basic normalization
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation
        s = re.sub(r'\s+', ' ', s)  # Normalize whitespace
        return s
    
    ref_norm = normalize(reference)
    cand_norm = normalize(candidate)
    
    # Exact match after normalization
    if ref_norm == cand_norm:
        return 1
    
    # Check if reference is contained in candidate
    if ref_norm in cand_norm:
        return 1
    
    # Check if candidate is contained in reference (for shorter answers)
    if cand_norm in ref_norm and len(cand_norm) > 3:
        return 1
    
    return 0
