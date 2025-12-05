# Llama-3.1-8B-Instruct Analysis

Welcome to our DSC 180A Capstone Repo! This repo contains code for different methods for quantifying LLM uncertainty. We specifically focus on Llama-3.1-8B-Instruct's performance. Each form of analysis is in its own branch.

## QRC
This branch implements a full evaluation pipeline for Quantile Risk Calibration (QRC) on Llama-3.1-8B-Instruct, focusing on how the model’s confidence behaves under different sampling conditions. QRC is a tail-focused calibration metric that examines model performance on the top-k% most confident predictions—in our case, the top 10%. Unlike traditional accuracy or ECE, which average over all predictions, QRC isolates the model’s strongest claims and tests whether high confidence actually corresponds to high correctness.  

Large language models are increasingly deployed in high-risk settings where overconfidence can be dangerous. QRC addresses this by focusing specifically on the region where the model expresses the most confidence. A model with high accuracy but poor QRC is unreliable when it matters most; a model with strong QRC provides meaningful confidence signals that downstream systems (and humans) can trust.

## SPUQ
This branch demonstrates Sampling with Perturbation for Uncertainty Quantification (SPUQ) on Llama-3.1-8B-Instruct. SPUQ is a perturbation-based uncertainty quantification method, where small changes are made to the input prompt, and the impact of those changes is measured on the model's response. Through both prompt perturbations and response aggregation, SPUQ is able to capture both aleatoric and epistemic uncertainty.

By measuring accuracy and ECE, SPUQ is able to help identify topics where a model might lack ground truth knowledge. If small prompt perturbations have a large impact on the output, it suggests that the model lacks confidence about the original prompt. If the model's responses are robust to prompt perturbations, the model likely has high confidence about the original prompt or topic. Understanding what topics models are less confident about can help humans use LLMs more safely.


## BLoB

This branch implements BLoB (Bayesian Low-Rank Adaptation by Backpropagation) for uncertainty quantification on Llama-3.1-8B-Instruct. BLoB is a Bayesian approach that learns probability distributions over LoRA adapter weights rather than fixed point estimates. During inference, multiple weight samples are drawn from these learned distributions, and the variation across model outputs provides a principled measure of epistemic uncertainty. We implement two distribution variants: Gaussian (symmetric, smooth perturbations) and Laplace (heavier tails, capturing occasional large deviations).

Traditional fine-tuning produces models that cannot distinguish between confident correct predictions and confident mistakes. BLoB addresses this by treating the adapter weights as random variables with learned means and variances. When multiple weight samples produce consistent answers, the model has high confidence; when samples disagree, uncertainty is high. This enables reliable confidence estimates that correlate with actual correctness. We evaluate on both multiple-choice QA datasets (ARC-Challenge, OBQA) and open-ended generative QA datasets (TriviaQA, TruthfulQA), using Natural Language Inference for semantic evaluation. Calibration is measured through accuracy, ECE, NLL, Brier Score, and Predictive Entropy, demonstrating how Bayesian weight uncertainty translates to meaningful confidence signals.
