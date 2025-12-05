# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from contextlib import suppress
from tqdm import tqdm
import math
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from torchmetrics import Accuracy, CalibrationError

from transformers import PreTrainedModel
from peft import PeftModel
from peft.config import PeftConfig

from utils import create_if_not_exists

optimizer_dict = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
}


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def accuracy_topk(output, target, k=1):
    """Computes the topk accuracy"""
    batch_size = target.size(0)

    _, pred = torch.topk(output, k=k, dim=1, largest=True, sorted=True)

    res_total = 0
    for curr_k in range(k):
        curr_ind = pred[:, curr_k]
        num_eq = torch.eq(curr_ind, target).sum()
        acc = num_eq / len(output)
        res_total += acc
    return res_total * 100


class AverageMeter(object):
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


class WrapperBase(PeftModel):
    """
    Base ModelWrapper for this project.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        args,
        accelerator,
        adapter_name: str = "default",
    ):
        """Initializes the model wrapper.

        Args:
            model (PreTrainedModel): The pretrained model to wrap.
            peft_config (PeftConfig): The configuration for parameter-efficient fine-tuning (PEFT).
            args (argparse.Namespace): Arguments with configuration for training.
            accelerator (Accelerator): The accelerator to handle multi-GPU or mixed precision.
            adapter_name (str, optional): The name of the adapter. Defaults to "default".
        """
        super().__init__(model, peft_config, adapter_name)

        self.loss = F.nll_loss
        self.args = args
        self.accelerator = accelerator
        self.target_ids = None

        self.batch_size = args.batch_size
        self.num_epochs = args.n_epochs
        self.num_training_steps = args.max_train_steps
        self.step = 0
        self.num_classes = args.outdim
        self.eval_n_samples = 1

        if args.max_train_steps == 0:
            num_training_steps = args.num_samples * args.n_epochs // args.batch_size
        else:
            num_training_steps = args.max_train_steps
        warmup_steps = num_training_steps * args.warmup_ratio
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                # set weight_decay
                "weight_decay": args.opt_wd,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if args.opt == "adamw" or args.opt == "adam":
            self.opt = optimizer_dict[args.opt](
                optimizer_grouped_parameters,
                lr=args.lr,
                eps=args.adam_epsilon,
                weight_decay=args.opt_wd,
            )
        else:
            self.opt = optimizer_dict[args.opt](
                optimizer_grouped_parameters, lr=args.lr, weight_decay=args.opt_wd
            )
        self.scheduler = get_linear_schedule_with_warmup(
            self.opt, warmup_steps, num_training_steps
        )

    def forward_logits(self, *args, **kwargs) -> torch.Tensor:
        """Defines the forward pass for computing logits. This method is not implemented 
        in the base class and needs to be implemented in subclasses.
        
        Returns:
            torch.Tensor: The logits predicted by the model.
        """
        raise NotImplementedError("Forward not implemented.")

    def fit(self, train_loader, eval_loader):
        """Fits the model using the training data and evaluates it periodically.
        
        Args:
            train_loader (DataLoader): The training data loader.
            eval_loader (DataLoader): The evaluation data loader.
        """
        nll_losses = AverageMeter()
        accs = AverageMeter()
        samples_seen = 0
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {self.args.epoch+1}/{self.args.n_epochs}",
            leave=False,
        ) as pbar:
            for i, batch in enumerate(train_loader):
                if self.args.dataset_type == "mcdataset":
                    _, golds, _ = batch
                elif self.args.dataset_type == "bertds":
                    golds = batch["labels"]
                else:
                    raise NotImplementedError(
                        f"Dataset type {self.args.dataset_type} not implemented."
                    )
                logits = self.forward_logits(batch).mean(1)
                output = torch.log_softmax(logits, dim=1)
                nll = self.loss(output, golds, reduction="mean")

                self.accelerator.backward(nll)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                acc = accuracy_topk(output.data, golds)
                acc, nll_loss = acc.item(), nll.detach().cpu().numpy()

                if self.args.dataset_type == "mcdataset":
                    _, classes, _ = batch
                    references = self.accelerator.gather(classes)
                else:
                    references = self.accelerator.gather(batch["labels"])
                if self.accelerator.num_processes > 1:
                    if i == len(train_loader) - 1:
                        references = references[
                            : len(train_loader.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += references.shape[0]
                len_batch = references.shape[0]
                nll_losses.update(nll_loss, len_batch)
                accs.update(acc, len_batch)

                assert not math.isnan(nll_loss)
                if self.accelerator.is_local_main_process:
                    if self.wandb_logger is not None:
                        self.wandb_logger.log(
                            {
                                "train_acc": accs.avg,
                                "train_nll_loss": nll_losses.avg,
                                "lr": self.opt.param_groups[0]["lr"],
                            }
                        )

                self.step += self.accelerator.num_processes
                pbar.update(1)
                if self.step >= self.args.eval_per_steps:
                    self.step -= self.args.eval_per_steps
                    # Use generative evaluation if specified
                    if hasattr(self.args, 'eval_mode') and self.args.eval_mode == "generative":
                        # Skip LLM judge during intermediate evals to save GPU memory
                        self.evaluate_generative(eval_loader, use_llm_judge=False)
                    else:
                        self.evaluate(eval_loader)

    def evaluate(self, eval_loader):
        """Evaluates the model using the evaluation data.
    
        Args:
            eval_loader (DataLoader): The evaluation data loader.
        
        Returns:
            tuple: The evaluation results: accuracy, ECE (Expected Calibration Error), 
                negative log-likelihood (NLL), and Brier score.
        """
        self.eval()
        status = self.training
        nlls = AverageMeter()
        metric_kwargs = {"task": "multiclass", "num_classes": self.num_classes}
        acc_metric = Accuracy(**metric_kwargs).to(self.accelerator.device)
        ece_metric = CalibrationError(**metric_kwargs, n_bins=self.args.num_bins).to(
            self.accelerator.device
        )
        briers = AverageMeter()

        samples_seen = 0
        for step, batch in enumerate(eval_loader):
            with torch.no_grad() and torch.inference_mode():
                logits = self.forward_logits(
                    batch, sample=True, n_samples=self.eval_n_samples
                ).detach()
                if self.args.dataset_type == "mcdataset":
                    _, labels, _ = batch
                else:
                    labels = batch["labels"]
                logits, labels = self.accelerator.gather([logits, labels])
                if self.accelerator.num_processes > 1:
                    if step == len(eval_loader) - 1:
                        labels = labels[: len(eval_loader.dataset) - samples_seen]
                        logits = logits[: len(eval_loader.dataset) - samples_seen]
                    else:
                        samples_seen += labels.shape[0]
                probs = torch.softmax(logits, dim=-1).mean(dim=1)
                if self.eval_n_samples > 1:
                    std = torch.softmax(logits, dim=-1).std(dim=1).mean()
                else:
                    std = 0

                acc_metric(probs, labels)
                ece_metric(probs, labels)
                nll = self.loss(torch.log(probs), labels, reduction="mean")
                if torch.isnan(nll):
                    if self.accelerator.is_local_main_process:
                        print("nll:", nll)
                        print("probs:", probs)
                        print("logits:", logits)
                        exit()
                nlls.update(nll)

                brier = (
                    (probs - F.one_hot(labels, num_classes=logits.size(-1)))
                    .pow(2)
                    .sum(dim=-1)
                    .mean()
                )
                briers.update(brier)

        val_acc = acc_metric.compute().item()
        val_ece = ece_metric.compute().item()
        val_nll = nlls.avg
        val_brier = briers.avg
        self.train(status)

        if self.accelerator.is_local_main_process:
            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    {
                        "val_acc": val_acc,
                        "val_ece": val_ece,
                        "val_nll": val_nll,
                        "std": std,
                        "val_brier": val_brier,
                    }
                )
        return val_acc, val_ece, val_nll, val_brier

    def evaluate_generative(self, eval_loader, use_llm_judge=None):
        """Evaluates the model using text generation for open-ended QA.
        
        Args:
            eval_loader (DataLoader): The evaluation data loader.
            use_llm_judge (bool, optional): Override to force LLM judge on/off. 
                If None, uses args.use_llm_judge. Set to False for intermediate evals to save memory.
        
        Returns:
            float: The accuracy score based on LLM-as-judge or string matching.
        """
        # TEST MARKER - If you see this, the function is being called!
        print("\n" + "="*80, flush=True)
        print("🔥🔥🔥 EVALUATE_GENERATIVE FUNCTION CALLED! 🔥🔥🔥", flush=True)
        print("="*80 + "\n", flush=True)
        
        from run.evaluation_generative import GenerativeEvaluator
        import logging as log
        
        self.eval()
        status = self.training
        
        # Determine evaluation method
        # Priority: explicit use_llm_judge param > generative_eval_method > use_llm_judge flag > default
        if use_llm_judge is not None:
            eval_method = "llm_judge" if use_llm_judge else "string_match"
        elif hasattr(self.args, 'generative_eval_method'):
            eval_method = self.args.generative_eval_method
        elif hasattr(self.args, 'use_llm_judge') and self.args.use_llm_judge:
            eval_method = "llm_judge"  # Backward compatibility
        else:
            eval_method = "string_match"
        
        # Initialize evaluator
        evaluator = GenerativeEvaluator(
            eval_method=eval_method,
            judge_model=self.args.llm_judge_model if hasattr(self.args, 'llm_judge_model') else "meta-llama/Llama-3.1-8B-Instruct",
            nli_model=self.args.nli_model if hasattr(self.args, 'nli_model') else "microsoft/deberta-v3-large-mnli",
            nli_threshold=self.args.nli_threshold if hasattr(self.args, 'nli_threshold') else 0.5,
            device=self.accelerator.device,
            judge_gpu_id=self.args.judge_gpu_id if hasattr(self.args, 'judge_gpu_id') else None
        )
        
        all_generated = []
        all_references = []
        all_correct = []
        all_bayesian_correct = []  # Store all Bayesian sample results for each test sample
        all_nll = []  # Track sequence-level negative log-likelihood
        all_nli_probs = []  # Track NLI confidence scores
        all_bayesian_samples = []  # Store all Bayesian samples per question
        samples_seen = 0
        
        # Calculate total generations needed
        total_samples = len(eval_loader.dataset) if hasattr(eval_loader, 'dataset') else 0
        eval_n_samples = self.eval_n_samples if hasattr(self, 'eval_n_samples') else 1
        total_generations = total_samples * eval_n_samples
        generations_completed = 0
        
        log.info(f"Starting generative evaluation...")
        log.info(f"Evaluation method: {eval_method}")
        log.info(f"Max new tokens: {self.args.max_new_tokens if hasattr(self.args, 'max_new_tokens') else 50}")
        log.info(f"Temperature: {self.args.generation_temperature if hasattr(self.args, 'generation_temperature') else 0.7}")
        if eval_method == "nli":
            log.info(f"NLI threshold: {self.args.nli_threshold if hasattr(self.args, 'nli_threshold') else 0.5}")
        log.info(f"Total test samples: {total_samples}, Bayesian samples: {eval_n_samples}, Total generations: {total_generations}")
        
        # Also print to stdout for kubectl logs visibility
        print(f"\n{'='*60}", flush=True)
        print(f"🚀 Starting generative evaluation...", flush=True)
        print(f"📊 Evaluation method: {eval_method}", flush=True)
        print(f"📝 Total test samples: {total_samples}", flush=True)
        print(f"🎲 Bayesian samples: {eval_n_samples}", flush=True)
        print(f"🔢 Total generations needed: {total_generations}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        for step, batch in enumerate(eval_loader):
            with torch.no_grad() and torch.inference_mode():
                # Extract prompts and targets
                prompts, targets, targets_aliases = batch
                
                # Extract tensors from BatchEncoding or dict
                if hasattr(prompts, 'input_ids'):
                    input_ids = prompts.input_ids
                    attention_mask = prompts.attention_mask if hasattr(prompts, 'attention_mask') else None
                elif isinstance(prompts, dict):
                    input_ids = prompts['input_ids']
                    attention_mask = prompts.get('attention_mask', None)
                else:
                    input_ids = prompts
                    attention_mask = None
                
                # Create attention mask if not present
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                
                # For Bayesian evaluation: generate multiple samples per input
                batch_bayesian_samples = []  # Store all samples for this batch
                batch_nlls = []  # Store NLL for each sample
                
                for sample_idx in range(eval_n_samples):
                    # Generate text with probability tracking
                    outputs = self.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.args.max_new_tokens if hasattr(self.args, 'max_new_tokens') else 50,
                        temperature=self.args.generation_temperature if hasattr(self.args, 'generation_temperature') else 0.7,
                        do_sample=self.args.generation_do_sample if hasattr(self.args, 'generation_do_sample') else True,
                        top_p=self.args.generation_top_p if hasattr(self.args, 'generation_top_p') else 0.9,
                        pad_token_id=self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else self.config.eos_token_id,
                        eos_token_id=self.config.eos_token_id,
                        return_dict_in_generate=True,  # Get dict output with scores
                        output_scores=True,  # Get token-level logits
                    )
                
                    # Calculate NLL from output scores
                    # outputs.scores is a tuple of tensors, one per generation step
                    # Each tensor has shape [batch_size, vocab_size]
                    sequence_nll = []
                    if hasattr(outputs, 'scores') and outputs.scores is not None and len(outputs.scores) > 0:
                        # Get the generated token IDs (excluding prompt)
                        input_length = input_ids.shape[1]
                        generated_ids = outputs.sequences[:, input_length:]
                        
                        # Calculate NLL for each sequence in batch
                        for batch_idx in range(generated_ids.shape[0]):
                            token_nlls = []
                            for step_idx, score in enumerate(outputs.scores):
                                if step_idx < generated_ids.shape[1]:
                                    # Get log probabilities
                                    log_probs = torch.log_softmax(score[batch_idx], dim=-1)
                                    # Get log prob of the actually generated token
                                    token_id = generated_ids[batch_idx, step_idx]
                                    token_nll = -log_probs[token_id].item()
                                    token_nlls.append(token_nll)
                            # Sum token NLLs to get sequence NLL
                            sequence_nll.append(sum(token_nlls) if token_nlls else 0.0)
                    else:
                        # If scores not available, use 0
                        sequence_nll = [0.0] * outputs.sequences.shape[0]
                    
                    # Decode generated text (excluding the input prompt)
                    generated_tokens = outputs.sequences[:, input_length:]
                    
                    # Get tokenizer - try multiple sources
                    tokenizer = None
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                        tokenizer = self.tokenizer
                    elif hasattr(self, 'base_model') and hasattr(self.base_model, 'config'):
                        # Load tokenizer from model's config
                        try:
                            from transformers import AutoTokenizer
                            tokenizer = AutoTokenizer.from_pretrained(
                                self.base_model.config._name_or_path,
                                trust_remote_code=True
                            )
                        except:
                            pass
                    
                    # Fallback: try to get from dataset
                    if tokenizer is None and hasattr(eval_loader, 'dataset'):
                        if hasattr(eval_loader.dataset, 'tokenizer'):
                            tokenizer = eval_loader.dataset.tokenizer
                        elif hasattr(eval_loader.dataset, 'dset') and hasattr(eval_loader.dataset.dset, 'tokenizer'):
                            tokenizer = eval_loader.dataset.dset.tokenizer
                    
                    if tokenizer is None:
                        raise ValueError(
                            "Cannot find tokenizer for decoding. "
                            "Tried: self.tokenizer, base_model.config, eval_loader.dataset.tokenizer"
                        )
                    
                    generated_texts = tokenizer.batch_decode(
                        generated_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Store samples and NLLs
                    batch_bayesian_samples.append(generated_texts)
                    batch_nlls.append(sequence_nll)
                
                # Decode reference answers
                # Extract tensors from BatchEncoding or dict
                if hasattr(targets, 'input_ids'):
                    target_ids = targets.input_ids
                elif isinstance(targets, dict):
                    target_ids = targets['input_ids']
                elif isinstance(targets, list):
                    # Already text
                    reference_texts = targets
                    target_ids = None
                else:
                    target_ids = targets
                
                # Decode if we have token IDs
                if target_ids is not None:
                    reference_texts = tokenizer.batch_decode(
                        target_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                
                # Now we have eval_n_samples generations for each test sample
                # batch_bayesian_samples[sample_idx][batch_idx] = generated text
                # batch_nlls[sample_idx][batch_idx] = NLL
                
                # Transpose to get per-test-sample view
                # We want: for each test sample, get all its Bayesian samples
                batch_size = len(reference_texts)
                for batch_idx in range(batch_size):
                    ref = reference_texts[batch_idx]
                    
                    # Collect all Bayesian samples for this test sample
                    sample_generations = [batch_bayesian_samples[s][batch_idx] for s in range(eval_n_samples)]
                    sample_nlls = [batch_nlls[s][batch_idx] for s in range(eval_n_samples)]
                    
                    # Evaluate each Bayesian sample
                    sample_correct = []
                    sample_nli_probs = []
                    
                    for gen_text in sample_generations:
                        if eval_method == "nli":
                            # Get detailed NLI results
                            correctness, details = evaluator.nli_evaluator.evaluate_single(
                                generated_answer=gen_text,
                                reference_answer=ref,
                                threshold=self.args.nli_threshold if hasattr(self.args, 'nli_threshold') else 0.5
                            )
                            sample_correct.append(correctness)
                            sample_nli_probs.append(details)
                        else:
                            # String matching or LLM judge
                            judgment = evaluator.evaluate_single(
                                generated_answer=gen_text,
                                reference_answer=ref,
                                question="",
                                context=None
                            )
                            sample_correct.append(judgment)
                            sample_nli_probs.append({})
                    
                    # Aggregate across Bayesian samples (majority vote or mean)
                    mean_correctness = sum(sample_correct) / len(sample_correct)
                    final_correct = 1 if mean_correctness > 0.5 else 0
                    
                    # Store results
                    all_correct.append(final_correct)
                    all_bayesian_correct.append(sample_correct)  # Store individual Bayesian results
                    all_generated.append(sample_generations[0])  # Store first sample as representative
                    all_references.append(ref)
                    all_nll.append(sample_nlls)  # Store all NLLs
                    all_nli_probs.append(sample_nli_probs)  # Store all NLI probs
                    all_bayesian_samples.append(sample_generations)  # Store all samples
                
                # Update progress counter (count test samples, not individual generations)
                generations_completed += batch_size * eval_n_samples
                
                # Log progress every 10 batches
                if (step + 1) % 10 == 0:
                    progress_pct = (generations_completed / total_generations * 100) if total_generations > 0 else 0
                    progress_msg = f"🔄 Generation progress: {generations_completed}/{total_generations} ({progress_pct:.1f}%) | Batch {step+1}/{len(eval_loader)}"
                    log.info(progress_msg)
                    print(progress_msg, flush=True)  # Also print to stdout for kubectl logs
        
        # Final progress log
        completion_msg = f"✅ Generation complete: {generations_completed}/{total_generations} (100%)"
        log.info(completion_msg)
        print(f"\n{completion_msg}\n", flush=True)  # Also print to stdout
        
        # Compute accuracy
        accuracy = sum(all_correct) / len(all_correct) if all_correct else 0.0
        
        # Compute uncertainty metrics
        import numpy as np
        
        # 1. Average NLL (Negative Log-Likelihood)
        all_nlls_flat = [nll for sample_nlls in all_nll for nll in sample_nlls]
        avg_nll = np.mean(all_nlls_flat) if all_nlls_flat else 0.0
        
        # 2. Predictive Entropy (uncertainty across Bayesian samples)
        predictive_entropies = []
        for sample_correct_list in all_bayesian_correct:
            # For each test sample, compute entropy over Bayesian predictions
            if len(sample_correct_list) > 1:
                # Count how many samples predicted each class
                p_correct = sum(sample_correct_list) / len(sample_correct_list)
                p_incorrect = 1 - p_correct
                # Binary entropy: H = -Σ p(x) log p(x)
                if p_correct > 0 and p_incorrect > 0:
                    H = -p_correct * np.log(p_correct) - p_incorrect * np.log(p_incorrect)
                    predictive_entropies.append(H)
                elif p_correct == 1.0 or p_incorrect == 1.0:
                    # All predictions agree - zero entropy (certain)
                    predictive_entropies.append(0.0)
        
        avg_predictive_entropy = np.mean(predictive_entropies) if predictive_entropies else 0.0
        
        # 3. Brier Score (mean squared error of probability predictions)
        brier_scores = []
        for idx, correct in enumerate(all_correct):
            # Get the mean probability across Bayesian samples
            if idx < len(all_nli_probs) and all_nli_probs[idx]:
                # For NLI: use avg_prob as confidence
                sample_probs = [details.get('avg_prob', 0.5) for details in all_nli_probs[idx] if details]
                mean_prob = np.mean(sample_probs) if sample_probs else 0.5
            else:
                # For non-NLI: use binary prediction as probability
                mean_prob = float(correct)
            # Brier score: (prediction - truth)^2
            brier_score = (mean_prob - float(correct)) ** 2
            brier_scores.append(brier_score)
        
        avg_brier = np.mean(brier_scores) if brier_scores else 0.0
        
        # 4. ECE (Expected Calibration Error)
        # Group predictions by confidence bins
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_counts = [0] * n_bins
        bin_correct = [0] * n_bins
        bin_confidence = [0] * n_bins
        
        for idx, correct in enumerate(all_correct):
            if idx < len(all_nli_probs) and all_nli_probs[idx]:
                sample_probs = [details.get('avg_prob', 0.5) for details in all_nli_probs[idx] if details]
                confidence = np.mean(sample_probs) if sample_probs else 0.5
            else:
                confidence = float(correct)
            
            # Find which bin this prediction falls into
            bin_idx = min(int(confidence * n_bins), n_bins - 1)
            bin_counts[bin_idx] += 1
            bin_correct[bin_idx] += correct
            bin_confidence[bin_idx] += confidence
        
        # Compute ECE
        ece = 0.0
        total_samples = sum(bin_counts)
        for i in range(n_bins):
            if bin_counts[i] > 0:
                bin_acc = bin_correct[i] / bin_counts[i]
                bin_conf = bin_confidence[i] / bin_counts[i]
                ece += (bin_counts[i] / total_samples) * abs(bin_acc - bin_conf)
        
        # 5. NLI Confidence Statistics
        if all_nli_probs and any(all_nli_probs):
            all_forward_probs = []
            all_backward_probs = []
            all_avg_probs = []
            for sample_details_list in all_nli_probs:
                for details in sample_details_list:
                    if details:
                        all_forward_probs.append(details.get('forward_prob', 0))
                        all_backward_probs.append(details.get('backward_prob', 0))
                        all_avg_probs.append(details.get('avg_prob', 0))
            
            avg_forward_prob = np.mean(all_forward_probs) if all_forward_probs else 0.0
            avg_backward_prob = np.mean(all_backward_probs) if all_backward_probs else 0.0
            avg_nli_conf = np.mean(all_avg_probs) if all_avg_probs else 0.0
        else:
            avg_forward_prob = 0.0
            avg_backward_prob = 0.0
            avg_nli_conf = 0.0
        
        self.train(status)
        
        if self.accelerator.is_local_main_process:
            log.info(f"=" * 80)
            log.info(f"📊 Generative Evaluation Results:")
            log.info(f"=" * 80)
            log.info(f"  Evaluation method: {eval_method}")
            log.info(f"  Test samples: {len(all_correct)}")
            log.info(f"  Bayesian samples per test: {eval_n_samples}")
            log.info(f"  Total generations: {generations_completed}")
            log.info(f"")
            log.info(f"🎯 Accuracy Metrics:")
            log.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            log.info(f"")
            log.info(f"📈 Uncertainty Metrics:")
            log.info(f"  Average NLL: {avg_nll:.4f}")
            log.info(f"  Predictive Entropy: {avg_predictive_entropy:.4f}")
            log.info(f"  Brier Score: {avg_brier:.4f}")
            log.info(f"  ECE (Calibration): {ece:.4f}")
            log.info(f"")
            if eval_method == "nli":
                log.info(f"🔍 NLI Confidence Statistics:")
                log.info(f"  Avg Forward Entailment Prob: {avg_forward_prob:.4f}")
                log.info(f"  Avg Backward Entailment Prob: {avg_backward_prob:.4f}")
                log.info(f"  Avg Overall NLI Confidence: {avg_nli_conf:.4f}")
                log.info(f"")
            log.info(f"=" * 80)
            
            # Also print to stdout for kubectl logs
            print(f"\n{'='*80}", flush=True)
            print(f"📊 Generative Evaluation Results:", flush=True)
            print(f"={'='*80}", flush=True)
            print(f"  Evaluation method: {eval_method}", flush=True)
            print(f"  Test samples: {len(all_correct)}", flush=True)
            print(f"  Bayesian samples per test: {eval_n_samples}", flush=True)
            print(f"  Total generations: {generations_completed}", flush=True)
            print(f"", flush=True)
            print(f"🎯 Accuracy Metrics:", flush=True)
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", flush=True)
            print(f"", flush=True)
            print(f"📈 Uncertainty Metrics:", flush=True)
            print(f"  Average NLL: {avg_nll:.4f}", flush=True)
            print(f"  Predictive Entropy: {avg_predictive_entropy:.4f}", flush=True)
            print(f"  Brier Score: {avg_brier:.4f}", flush=True)
            print(f"  ECE (Calibration): {ece:.4f}", flush=True)
            print(f"", flush=True)
            if eval_method == "nli":
                print(f"🔍 NLI Confidence Statistics:", flush=True)
                print(f"  Avg Forward Entailment Prob: {avg_forward_prob:.4f}", flush=True)
                print(f"  Avg Backward Entailment Prob: {avg_backward_prob:.4f}", flush=True)
                print(f"  Avg Overall NLI Confidence: {avg_nli_conf:.4f}", flush=True)
                print(f"", flush=True)
            print(f"{'='*80}\n", flush=True)
            
            if self.wandb_logger is not None:
                wandb_metrics = {
                    "val_acc": accuracy,
                    "n_evaluated": len(all_correct),
                    "avg_nll": avg_nll,
                    "predictive_entropy": avg_predictive_entropy,
                    "brier_score": avg_brier,
                    "ece": ece,
                }
                if eval_method == "nli":
                    wandb_metrics.update({
                        "nli_forward_prob": avg_forward_prob,
                        "nli_backward_prob": avg_backward_prob,
                        "nli_avg_confidence": avg_nli_conf,
                    })
                self.wandb_logger.log(wandb_metrics)
        
        # Return metrics (accuracy, ece, nll, brier for compatibility)
        return accuracy, ece, avg_nll, avg_brier

    def fit_evaluate(self):
        """Performs the fitting and evaluation process, saving the results to checkpoints 
        and logging them to the WandB logger.
        """
        if self.accelerator.is_local_main_process:
            save_folder = f"checkpoints/{self.args.modelwrapper}/{self.args.model}/{self.args.dataset}/{self.args.log_path}"
            create_if_not_exists(save_folder)
            logging.basicConfig(
                format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                level=logging.INFO,
                filename=save_folder + "/log.txt",
            )
        with tqdm(
            total=self.args.n_epochs, desc=f"Total Training Epochs", leave=True
        ) as pbar:
            for epoch in range(self.args.n_epochs):
                if self.args.early_stop_steps > 0 and epoch >= self.earlystop_n_epochs:
                    break
                self.args.epoch = epoch
                self.fit(self.train_loader, self.test_loader)
                pbar.update(1)

        if hasattr(self.args, "bayes_eval_n_samples_final"):
            self.eval_n_samples = self.args.bayes_eval_n_samples_final

        # Use generative evaluation if specified
        if hasattr(self.args, 'eval_mode') and self.args.eval_mode == "generative":
            val_acc, val_ece, val_nll, val_brier = self.evaluate_generative(self.test_loader)
        else:
            val_acc, val_ece, val_nll, val_brier = self.evaluate(self.test_loader)
        
        logging.info(
            f"val_acc: {val_acc}, val_ece: {val_ece}, val_nll: {val_nll}, val_brier: {val_brier}"
        )
        if self.accelerator.is_local_main_process:
            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    {
                        "final_val_acc": val_acc,
                        "final_val_ece": val_ece,
                        "final_val_nll": val_nll,
                        "final_val_brier": val_brier,
                    }
                )

    def prepare_for_fit_evaluate(self, dataset, wandb_logger=None):
        """Prepares the model and data loaders for training and evaluation.
    
        Args:
            dataset (Dataset): The dataset object containing train and test data loaders.
            wandb_logger (optional): The Weights & Biases logger for tracking experiments.
        """
        self.wandb_logger = wandb_logger
        train_loader, test_loader = dataset.train_dataloader, dataset.test_dataloader
        
        if self.args.testing_set == 'train_train_val' or self.args.testing_set == 'train_val_val':
            anchor_loader = dataset.anchor_dataloader
            anchor_loader = self.accelerator.prepare(anchor_loader)
            self.anchor_loader = anchor_loader

        if self.args.dataset_type == "mcdataset":
            self.tokenizer = dataset.tokenizer
            self.target_ids = dataset.target_ids.squeeze(-1)
        elif self.args.dataset_type == "oedataset":
            # For generative QA tasks, store tokenizer for decoding
            self.tokenizer = dataset.tokenizer if hasattr(dataset, 'tokenizer') else None

        num_update_steps_per_epoch = math.ceil(len(train_loader))
        if self.args.max_train_steps == 0:
            self.args.max_train_steps = self.args.n_epochs * num_update_steps_per_epoch
        self.args.n_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )
        if self.args.early_stop_steps > 0:
            self.earlystop_n_epochs = math.ceil(
                self.args.early_stop_steps / num_update_steps_per_epoch
            )
        else:
            self.earlystop_n_epochs = 0
        if self.accelerator.is_local_main_process:
            print("len(train_loader):", len(train_loader))
            print("num of epochs:", self.args.n_epochs)
        self.step = 0

        self.base_model, self.opt, train_loader, test_loader, self.scheduler = (
            self.accelerator.prepare(
                self.base_model, self.opt, train_loader, test_loader, self.scheduler
            )
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
