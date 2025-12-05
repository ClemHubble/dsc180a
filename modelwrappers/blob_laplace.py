"""
BLoB with Laplace Noise Distribution

This is a variant of BLoB (Bayesian Low-Rank Adaptation by Backpropagation)
that uses Laplace distribution instead of Gaussian distribution for sampling noise.

Key Difference:
- Original BLoB: Uses torch.randn_like() for Gaussian N(0, σ²) noise
- This variant: Uses Laplace distribution with heavier tails

The Laplace distribution has heavier tails than Gaussian, which can better capture
uncertainty in scenarios where occasional large deviations are expected.

Reference: Based on blob.py from the BLoB paper implementation
"""


import torch
import torch.nn as nn
from torch.optim import SGD
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
import math
from tqdm import tqdm

from .wrapperbase import WrapperBase, get_linear_schedule_with_warmup
from utils.args import add_management_args, add_experiment_args, ArgumentParser

from transformers import PreTrainedModel

from peft.config import PeftConfig
from peft.tuners.lora import LoraLayer, Linear
from peft.tuners.lora.bnb import Linear8bitLt

# Import evaluation utilities only when needed
try:
    from run.evaluation import *
except ImportError:
    # For testing without full dependencies
    pass


## Model Specific Argument Parsing
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Bayesian By Backprop with Laplace Noise, BLoB-Laplace.")
    add_management_args(parser)
    add_experiment_args(parser)
    # BLoB-specific arguments.
    parser.add_argument("--bayes-train-n-samples", type=int, default=1)
    parser.add_argument(
        "--bayes-eval-n-samples",
        type=int,
        default=1,
        help="Number of samples to use for evaluation during training.",
    )
    parser.add_argument(
        "--bayes-eval-n-samples-final",
        type=int,
        default=10,
        help="Number of samples to use for evaluation.",
    )

    parser.add_argument("--bayes-eps", type=float, default=0.05)
    parser.add_argument("--bayes-gamma", type=float, default=8)
    parser.add_argument("--bayes-kllr", type=float, default=0.02)
    parser.add_argument("--bayes-beta", type=float, default=0.2)
    parser.add_argument(
        "--bayes-inference-notsample",
        action="store_true",
        help="Whether to sample during inference.",
    )
    parser.add_argument(
        "--bayes-klreweighting", action="store_true", help="Whether to use reweighting."
    )
    parser.add_argument('--bayes-datasetrescaling', action='store_true',
                        help='Whether to use datasetrescaling.')
    # New argument for Laplace scale parameter (optional tuning)
    parser.add_argument("--laplace-scale-factor", type=float, default=1.0,
                        help="Scale factor for Laplace distribution (default 1.0 uses same as Gaussian)")

    return parser


@dataclass
class BLoBLaplaceConfig:
    bayes_eps: float = field(metadata={"help": "Bayes epsilon"})
    bayes_beta: float = field(metadata={"help": "Bayes beta"})
    laplace_scale_factor: float = field(default=1.0, metadata={"help": "Laplace scale factor"})


def blob_laplace_linear_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
    """
    Forward pass with Laplace noise instead of Gaussian noise.
    
    Key Change: Uses Laplace distribution for sampling lora_noise_a
    """
    previous_dtype = x.dtype

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            result = result + lora_B(lora_A(dropout(x))) * scaling  # Changed from += to avoid in-place modification

    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        if self.blobsample:
            if self.bayes_eps < 0:
                A_sigma = torch.log1p(torch.exp(self.lora_A_rho[active_adapter]))
            else:
                A_sigma = self.lora_A_rho[active_adapter] ** 2

            scaling = self.scaling[active_adapter]
            dropout = self.lora_dropout[active_adapter]

            x = x.to(lora_A.weight.dtype)
            if x.dim() == 2:
                r_A = (
                    torch.ones(
                        (x.size(0), self.in_features), device=x.device, dtype=x.dtype
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), self.r[active_adapter]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
            else:
                r_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.in_features),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.r[active_adapter]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )

            x = dropout(x)
            
            # ============ KEY CHANGE: LAPLACE NOISE ============
            # Original: lora_noise_a = A_sigma * torch.randn_like(...)
            # New: Sample from Laplace distribution
            
            # Scale parameter for Laplace distribution
            # For Laplace distribution, scale parameter b relates to variance as: Var = 2b²
            # For Gaussian: Var = σ²
            # To match variance: b = σ/√2, but we allow tuning via laplace_scale_factor
            laplace_scale = A_sigma * self.laplace_scale_factor / math.sqrt(2)
            
            # Create Laplace distribution
            laplace_dist = torch.distributions.Laplace(
                loc=torch.zeros_like(self.lora_A[active_adapter].weight),
                scale=laplace_scale
            )
            lora_noise_a = laplace_dist.sample()
            # ====================================================

            noise = (((x * r_A) @ lora_noise_a.transpose(0, 1)) * s_A) @ self.lora_B[
                active_adapter
            ].weight.transpose(0, 1)

            result = result + noise * scaling  # Changed from += to avoid in-place modification

        result = result.to(previous_dtype)

    return result


def blob_laplace_8bitlinear_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
    """
    8-bit quantized forward pass with Laplace noise.
    """
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = lora_A.weight.dtype
                if x.dtype != compute_dtype:
                    x = x.to(compute_dtype)
            output = lora_B(lora_A(dropout(x)))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * scaling
            result = result + output  # Changed from += to avoid in-place modification
    if self.blobsample:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            if self.bayes_eps < 0:
                A_sigma = torch.log1p(torch.exp(self.lora_A_rho[active_adapter]))
            else:
                A_sigma = self.lora_A_rho[active_adapter] ** 2
            scaling = self.scaling[active_adapter]
            dropout = self.lora_dropout[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = lora_A.weight.dtype
                if x.dtype != compute_dtype:
                    x = x.to(compute_dtype)

            if x.dim() == 2:
                r_A = (
                    torch.ones(
                        (x.size(0), self.in_features), device=x.device, dtype=x.dtype
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), self.r[active_adapter]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
            else:
                r_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.in_features),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.r[active_adapter]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )

            x = dropout(x)
            
            # ============ KEY CHANGE: LAPLACE NOISE ============
            laplace_scale = A_sigma * self.laplace_scale_factor / math.sqrt(2)
            laplace_dist = torch.distributions.Laplace(
                loc=torch.zeros_like(self.lora_A[active_adapter].weight),
                scale=laplace_scale
            )
            lora_noise_a = laplace_dist.sample()
            # ====================================================

            noise = (((x * r_A) @ lora_noise_a.transpose(0, 1)) * s_A) @ self.lora_B[
                active_adapter
            ].weight.transpose(0, 1)

            if requires_conversion:
                noise = noise.to(expected_dtype)

            result = result + noise * scaling  # Changed from += to avoid in-place modification

    return result


def div_posterior_prior(self) -> torch.Tensor:
    """
    KL divergence computation (same as original BLoB).
    Note: The prior is still Gaussian, only the sampling noise is Laplace.
    """
    def kl_div_stable(mu_q, sigma_q, mu_p, sigma_p):
        eps = 1e-6
        kl = (
            math.log(sigma_p + eps)
            - torch.log(sigma_q.to(torch.float64) + eps)
            + (sigma_q.to(torch.float64) ** 2 + (mu_q.to(torch.float64) - mu_p) ** 2)
            / (2 * (sigma_p**2) + eps)
            - 0.5
        )
        return kl.sum()

    kl = 0
    for active_adapter in self.active_adapters:
        if self.bayes_eps < 0:
            sigma_weight = torch.log1p(torch.exp(self.lora_A_rho[active_adapter]))
        else:
            sigma_weight = self.lora_A_rho[active_adapter] ** 2
        kl += kl_div_stable(
            self.lora_A[active_adapter].weight, sigma_weight, 0, self.bayes_beta
        )
    return kl


def sample(self, status=True):
    if self.training is True and status is False:
        raise ValueError("blobsample should be set to True only during training.")
    self.blobsample = status


class BLoBLaplace(WrapperBase):
    """BLoB model with Laplace noise distribution."""

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        args,
        accelerator,
        adapter_name: str = "default",
    ):
        super().__init__(model, peft_config, args, accelerator, adapter_name)

        self.blobconfig = BLoBLaplaceConfig(
            bayes_eps=self.args.bayes_eps,
            bayes_beta=self.args.bayes_beta,
            laplace_scale_factor=getattr(self.args, 'laplace_scale_factor', 1.0),
        )
        self._modify_lora_layers(self.base_model)
        if args.load_lora_path is not None:
            self.load_adapter(args.load_lora_path, adapter_name)

        self.i = 1  # for the KL re-weighting.
        self.ii = 1
        self.M = 0  # for the KL re-weighting.

        self.train_n_samples = self.args.bayes_train_n_samples
        self.eval_n_samples = self.args.bayes_eval_n_samples
        self.klreweighting = self.args.bayes_klreweighting

        if self.args.max_train_steps == 0:
            num_training_steps = (
                self.args.num_samples * self.args.n_epochs // self.args.batch_size
            )
        else:
            num_training_steps = self.args.max_train_steps
        warmup_steps = num_training_steps * self.args.warmup_ratio

        params = [param for name, param in self.named_parameters()]
        self.opt2 = SGD([{"params": params}], lr=args.bayes_kllr)
        self.scheduler2 = get_linear_schedule_with_warmup(
            self.opt2, warmup_steps, num_training_steps
        )

    def _modify_lora_layers(self, module):
        """
        Recursively go through the model and modify LoraLayer instances.
        """
        for name, child in module.named_children():
            if isinstance(child, LoraLayer) and isinstance(child, Linear):
                self._wrap_lora_layer(child)
                # modify existing methods
                setattr(
                    child,
                    "forward",
                    blob_laplace_linear_forward.__get__(child, child.__class__),
                )
                # add new methods
                setattr(
                    child,
                    "div_posterior_prior",
                    div_posterior_prior.__get__(child, child.__class__),
                )
                setattr(child, "sample", sample.__get__(child, child.__class__))
            if isinstance(child, LoraLayer) and isinstance(child, Linear8bitLt):
                self._wrap_lora_layer(child)
                # modify existing methods
                setattr(
                    child,
                    "forward",
                    blob_laplace_8bitlinear_forward.__get__(child, child.__class__),
                )
                # add new methods
                setattr(
                    child,
                    "div_posterior_prior",
                    div_posterior_prior.__get__(child, child.__class__),
                )
                setattr(child, "sample", sample.__get__(child, child.__class__))
            else:
                self._modify_lora_layers(child)

    def _wrap_lora_layer(self, lora_layer):
        lora_layer.lora_A_rho = nn.ParameterDict({})
        lora_layer.bayes_eps = self.blobconfig.bayes_eps
        lora_layer.bayes_beta = self.blobconfig.bayes_beta
        lora_layer.blobsample = True
        lora_layer.laplace_scale_factor = self.blobconfig.laplace_scale_factor

        for adapter_name in lora_layer._active_adapter:
            lora_layer.lora_A_rho[adapter_name] = nn.Parameter(
                lora_layer.lora_A[adapter_name].weight.new_zeros(
                    lora_layer.r[adapter_name], lora_layer.in_features
                )
            )

        if adapter_name in lora_layer.lora_A.keys():
            if lora_layer.bayes_eps < 0:
                nn.init.uniform_(
                    lora_layer.lora_A_rho[adapter_name],
                    lora_layer.bayes_eps - 1,
                    lora_layer.bayes_eps,
                )
            else:
                nn.init.uniform_(
                    lora_layer.lora_A_rho[adapter_name],
                    lora_layer.bayes_eps / math.sqrt(2),
                    lora_layer.bayes_eps,
                )

        return

    def div_posterior_prior(self, module):
        kl = 0
        for name, child in module.named_children():
            if isinstance(child, LoraLayer):
                kl_ = child.div_posterior_prior()
                kl += kl_
            else:
                kl += self.div_posterior_prior(child)
        return kl

    def sample(self, module, status=True):
        """
        Set the sampling status of the model.
        """
        for name, child in module.named_children():
            if isinstance(child, LoraLayer):
                child.sample(status)
            else:
                self.sample(child, status)

    def forward_logits(self, batch, sample=True, n_samples=1, **kwargs) -> torch.Tensor:
        if self.args.dataset_type == "mcdataset":
            inputs, _, _ = batch
            if not sample:
                self.sample(self.base_model, False)
                output = self.base_model(**inputs)
                logits = output.logits[:, -1, self.target_ids]
                self.sample(self.base_model, True)
                return logits.unsqueeze(1)
            else:
                logits_list = []
                for _ in range(n_samples):
                    output = self.base_model(**inputs)
                    logits = output.logits[:, -1, self.target_ids]
                    logits_list.append(logits)
                return torch.stack(logits_list, dim=1)
        else:
            if not sample:
                self.sample(self.base_model, False)
                res = self.base_model(**batch).logits
                self.sample(self.base_model, True)
                return res.unsqueeze(1)
            else:
                res = []
                for _ in range(n_samples):
                    res.append(self.base_model(**batch).logits)
                return torch.stack(res, dim=1)

    def fit(self, train_loader, eval_loader):
        nll_losses = AverageMeter()
        kl_losses = AverageMeter()
        elbo_losses = AverageMeter()
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
                    logits = self.forward_logits(
                        batch, sample=True, n_samples=self.train_n_samples
                    ).mean(1)
                    output = torch.log_softmax(logits, dim=1)
                    nll = self.loss(output, golds, reduction="mean")
                elif self.args.dataset_type == "bertds":
                    golds = batch["labels"]
                    logits = self.forward_logits(
                        batch, sample=True, n_samples=self.train_n_samples
                    ).mean(1)
                    output = torch.log_softmax(logits, dim=1)
                    nll = self.loss(output, golds, reduction="mean")
                elif self.args.dataset_type == "oedataset":
                    # Open-ended dataset: use language modeling loss
                    # Training collate returns 2 items (prompts, targets)
                    # Eval collate returns 3 items (prompts, targets, targets_aliases)
                    if len(batch) == 3:
                        prompts, targets, _ = batch
                    else:
                        prompts, targets = batch
                    
                    # Extract tensors from BatchEncoding or dict
                    # BatchEncoding has attributes, dicts have keys
                    if hasattr(prompts, 'input_ids'):
                        input_ids = prompts.input_ids
                        attention_mask = prompts.attention_mask if hasattr(prompts, 'attention_mask') else None
                    elif isinstance(prompts, dict):
                        input_ids = prompts['input_ids']
                        attention_mask = prompts.get('attention_mask', None)
                    else:
                        input_ids = prompts
                        attention_mask = None
                    
                    if hasattr(targets, 'input_ids'):
                        labels = targets.input_ids
                    elif isinstance(targets, dict):
                        labels = targets['input_ids']
                    else:
                        labels = targets
                    
                    # Use model's built-in LM loss
                    outputs = self.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    nll = outputs.loss
                else:
                    raise NotImplementedError(
                        f"Dataset type {self.args.dataset_type} not implemented."
                    )

                self.accelerator.backward(nll)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                kl_divs = []
                for _ in range(self.train_n_samples):
                    if hasattr(self.base_model, "module"):
                        kl_divs.append(self.div_posterior_prior(self.base_model.module))
                    else:
                        kl_divs.append(self.div_posterior_prior(self.base_model))
                kl = torch.mean(torch.stack(kl_divs), dim=0)

                if self.klreweighting:
                    if self.i % self.M == 0:
                        i = self.M
                    else:
                        i = self.i % self.M
                    self.pi = 2**i / (2 ** (self.M + 1) - 1)
                    self.i += 1
                else:
                    self.pi = 1 / self.M
                kl_div = kl * self.pi
                self.accelerator.backward(kl_div)
                self.opt2.step()
                self.opt2.zero_grad()
                self.scheduler2.step()

                # For generative tasks, we don't compute accuracy during training
                if self.args.dataset_type == "oedataset":
                    acc = 0.0
                    len_batch = input_ids.shape[0]
                else:
                    acc = accuracy_topk(output.data, golds)
                    acc = acc.item()

                loss, acc, nll_loss, kl = (
                    (kl + nll).detach().cpu().numpy(),
                    acc,
                    nll.detach().cpu().numpy(),
                    kl_div.detach().cpu().numpy(),
                )

                if self.args.dataset_type == "oedataset":
                    # For generative tasks, batch size is from input_ids
                    len_batch = input_ids.shape[0]
                elif self.args.dataset_type == "mcdataset":
                    _, classes, _ = batch
                    references = self.accelerator.gather(classes)
                    if self.accelerator.num_processes > 1:
                        if i == len(train_loader) - 1:
                            references = references[
                                : len(train_loader.dataset) - samples_seen
                            ]
                        else:
                            samples_seen += references.shape[0]
                    len_batch = references.shape[0]
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
                kl_losses.update(kl, len_batch)
                nll_losses.update(nll_loss, len_batch)
                elbo_losses.update(loss, len_batch)
                accs.update(acc, len_batch)

                assert not math.isnan(nll_loss)
                assert not math.isnan(kl)
                if self.accelerator.is_local_main_process:
                    if self.wandb_logger is not None:
                        self.wandb_logger.log(
                            {
                                "train_acc": accs.avg,
                                "train_nll_loss": nll_losses.avg,
                                "kl_loss": kl_losses.avg,
                                "elbo_loss": elbo_losses.avg,
                                "lr": self.opt.param_groups[0]["lr"],
                                "pi": self.pi,
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
        print("self.eval_n_samples:", self.eval_n_samples)
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
                    batch,
                    sample=not self.args.bayes_inference_notsample,
                    n_samples=self.eval_n_samples,
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
                std = torch.softmax(logits, dim=-1).std(dim=1).mean()

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

    def prepare_for_fit_evaluate(self, dataset, wandb_logger=None):
        """
        Prepare the model for training and evaluation.
        """
        self.wandb_logger = wandb_logger
        train_loader, test_loader = dataset.train_dataloader, dataset.test_dataloader

        if self.args.dataset_type == "mcdataset":
            self.target_ids = dataset.target_ids.squeeze(-1)

        l_train = len(train_loader)

        num_update_steps_per_epoch = len(train_loader)
        if self.args.max_train_steps == 0:
            self.args.max_train_steps = self.args.n_epochs * num_update_steps_per_epoch
        self.args.n_epochs = (
            math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)
            if self.args.ood_ori_dataset is None
            else 0
        )
        if self.args.early_stop_steps > 0:
            self.earlystop_n_epochs = (
                math.ceil(self.args.early_stop_steps / num_update_steps_per_epoch)
                if self.args.ood_ori_dataset is None
                else 0
            )
        else:
            self.earlystop_n_epochs = 0
        if self.accelerator.is_local_main_process:
            print("len(train_loader):", len(train_loader))
            print("num of epochs:", self.args.n_epochs)
        self.step = 0

        (
            self.base_model,
            self.opt,
            train_loader,
            test_loader,
            self.scheduler,
            self.scheduler2,
            self.opt2,
        ) = self.accelerator.prepare(
            self.base_model,
            self.opt,
            train_loader,
            test_loader,
            self.scheduler,
            self.scheduler2,
            self.opt2,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        if self.args.bayes_datasetrescaling:
            self.M = int(
                100
                * (dataset.num_samples ** (math.pi / self.args.bayes_gamma))
                / (l_train / len(train_loader))
                / self.args.batch_size
            )
        else:
            self.M = len(train_loader)

        print("M:", self.M)
        print(f"Using Laplace noise with scale factor: {self.blobconfig.laplace_scale_factor}")

