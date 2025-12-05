#!/usr/bin/env python3
"""
Mixin class to add text generation capabilities to model wrappers.
Supports Bayesian uncertainty quantification via posterior sampling during generation.
"""

import torch
from typing import List, Dict, Optional
from run.evaluation_generative import GenerativeEvaluator, AverageMeter
import logging


class GenerativeMixin:
    """
    Mixin to add generative evaluation capabilities to existing model wrappers.
    
    This allows Bayesian LoRA models to generate text with uncertainty quantification
    by sampling from the posterior distribution during generation.
    """
    
    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text from the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate per input
        
        Returns:
            List of generated text strings
        """
        self.eval()
        
        # Generate text
        outputs = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.config.pad_token_id or self.config.eos_token_id,
            eos_token_id=self.config.eos_token_id,
        )
        
        # Decode generated text (excluding the input prompt)
        input_length = input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
        
        # Get tokenizer from the model
        if hasattr(self, 'tokenizer'):
            tokenizer = self.tokenizer
        elif hasattr(self, 'get_base_model'):
            base_model = self.get_base_model()
            tokenizer = getattr(base_model, 'tokenizer', None)
        else:
            raise AttributeError("Cannot find tokenizer for decoding")
        
        if tokenizer is None:
            raise ValueError("Tokenizer not found in model")
        
        generated_texts = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return generated_texts
    
    @torch.no_grad()
    def generate_with_bayesian_sampling(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_samples: int = 5,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> List[List[str]]:
        """
        Generate text with Bayesian sampling.
        For each input, generate n_samples by sampling from the posterior.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            n_samples: Number of posterior samples to draw
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation arguments
        
        Returns:
            List of [n_samples generated texts] for each input
        """
        self.eval()
        batch_size = input_ids.shape[0]
        all_samples = [[] for _ in range(batch_size)]
        
        for sample_idx in range(n_samples):
            # For Bayesian models, each forward pass samples from the posterior
            # The sampling happens automatically in the forward pass if sample=True
            generated_texts = self.generate_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )
            
            for batch_idx, text in enumerate(generated_texts):
                all_samples[batch_idx].append(text)
        
        return all_samples
    
    def evaluate_generative(
        self,
        eval_loader,
        judge_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        use_llm_judge: bool = True,
        n_samples: int = 1,
        max_new_tokens: int = 50,
        generation_kwargs: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on generative QA task.
        
        Args:
            eval_loader: DataLoader with evaluation data
            judge_model: Model to use as LLM judge
            use_llm_judge: Whether to use LLM judge (vs simple string matching)
            n_samples: Number of posterior samples for Bayesian evaluation
            max_new_tokens: Max tokens to generate
            generation_kwargs: Additional generation parameters
        
        Returns:
            Dictionary with evaluation metrics
        """
        if generation_kwargs is None:
            generation_kwargs = {"temperature": 0.7, "do_sample": True, "top_p": 0.9}
        
        # Initialize evaluator
        evaluator = GenerativeEvaluator(
            judge_model=judge_model,
            use_llm_judge=use_llm_judge,
            device=self.accelerator.device if hasattr(self, 'accelerator') else 'cuda'
        )
        
        self.eval()
        status = self.training
        
        all_generated = []
        all_references = []
        all_questions = []
        all_contexts = []
        
        samples_seen = 0
        
        logging.info(f"Starting generative evaluation with {n_samples} samples per question...")
        
        for step, batch in enumerate(eval_loader):
            # Extract prompts and targets
            prompts, targets, targets_aliases = batch
            
            if isinstance(prompts, dict):
                input_ids = prompts['input_ids']
                attention_mask = prompts['attention_mask']
            else:
                input_ids = prompts
                attention_mask = torch.ones_like(input_ids)
            
            # Generate text (with optional Bayesian sampling)
            if n_samples > 1:
                # Bayesian: multiple samples from posterior
                generated_samples = self.generate_with_bayesian_sampling(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    n_samples=n_samples,
                    max_new_tokens=max_new_tokens,
                    **generation_kwargs
                )
                # For now, take the first sample for simple accuracy
                generated_texts = [samples[0] for samples in generated_samples]
            else:
                # Single deterministic generation
                generated_texts = self.generate_text(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic for single sample
                )
            
            # Extract reference answers
            if isinstance(targets, dict):
                # Tokenized targets, need to decode
                if hasattr(self, 'tokenizer'):
                    tokenizer = self.tokenizer
                else:
                    raise ValueError("Tokenizer not available for decoding targets")
                
                reference_texts = tokenizer.batch_decode(
                    targets['input_ids'],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            elif isinstance(targets, list):
                reference_texts = targets
            else:
                raise ValueError(f"Unsupported targets type: {type(targets)}")
            
            all_generated.extend(generated_texts)
            all_references.extend(reference_texts)
            
            # TODO: Extract questions and contexts from batch
            # This is dataset-specific
            all_questions.extend(["" for _ in generated_texts])
            all_contexts.extend([None for _ in generated_texts])
            
            if (step + 1) % 10 == 0:
                logging.info(f"Processed {step + 1}/{len(eval_loader)} batches")
        
        # Evaluate using LLM judge
        logging.info("Evaluating generated answers with LLM judge...")
        judgments = evaluator.evaluate_batch(
            generated_answers=all_generated,
            reference_answers=all_references,
            questions=all_questions,
            contexts=all_contexts,
        )
        
        # Compute metrics
        accuracy = sum(judgments) / len(judgments) if judgments else 0.0
        
        self.train(status)
        
        metrics = {
            "accuracy": accuracy,
            "n_samples": n_samples,
            "n_evaluated": len(judgments)
        }
        
        if self.accelerator.is_local_main_process if hasattr(self, 'accelerator') else True:
            logging.info(f"Generative Evaluation Results:")
            logging.info(f"  Accuracy: {accuracy:.4f}")
            logging.info(f"  Samples: {n_samples}")
            logging.info(f"  Evaluated: {len(judgments)} examples")
        
        return metrics



