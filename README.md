# SPUQ: Perturbation-Based Uncertainty Quantification for Large Language Models

## Introduction

This repository contains code to run Sampling with Perturbation for Uncertainty Quantification, as outlined in this [paper](https://arxiv.org/abs/2403.02509). There are three methods of perturbations that are tested: paraphrasing, dummy tokens, and system messages. Each of the three methods is tested on 5 datasets: 3 multiple choice question and answer datasets and 2 open ended question and answer datasets.

## Datasets

Multiple Choice Question Answer Datasets:
- [Arc-Challenge](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Challenge)
- [Arc-Easy](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Easy)
- [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa)

Open Ended Question Answer Datasets:
- [TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA)
- [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa)

## To run the code:

- Download dependencies from requirements.txt
- Download datasets
- Run run.py (located in spuq_code folder) to see results

