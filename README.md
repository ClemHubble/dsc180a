# SPUQ: Perturbation-Based Uncertainty Quantification for Large Language Models

## Introduction

This repository contains code to run Sampling with Perturbation for Uncertainty Quantification, as outlined in this [paper](https://arxiv.org/abs/2403.02509). There are three methods of perturbations that are tested: paraphrasing, dummy tokens, and system messages. Each of the three methods is tested on 5 datasets: 3 multiple choice question and answer datasets and 2 open ended question and answer datasets. The model used in this experiment is `Llama-3.1-8B-Instruct`. For each dataset, the accuracy and expected calibration error (ECE) is measured for each perturbation type.

## Datasets

Multiple Choice Question Answer Datasets:
- [Arc-Challenge](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Challenge)
- [Arc-Easy](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Easy)
- [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa)

Open Ended Question Answer Datasets:
- [TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA)
- [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa)

## Repo Structure
```

spuq
├── spuq_code/
│   ├── aggregation.py
│   ├── llms.py
│   ├── perturbation.py
│   ├── run.py
│   ├── spuq.py
│   └── text_sim.py
├── .DS_Store
├── README.md
└── requirements.txt

```


## Configuration

1. Set up environment and download dependencies.

```bash
conda create --name spuq_env python=3.10
conda activate spuq_env

conda install pip
pip install -r requirements.txt
```

2. Obtain Hugging Face token and access to `Llama-3.1-8B-Instruct`

```bash
pip install huggingface-hub
huggingface-cli login
# paste your token when prompted
```
3. Download datasets linked above and place them into spuq_code folder
4. Run run.py (located in spuq_code folder) to see results

```bash
cd spuq_code
python run.py
```

## Modifications

To change perturbation type, modify line 6 of run.py so that the perturbation parameter is one of the following:
- `paraphrasing`
- `system_message`
- `dummy_token`

To change the aggregation method, modify line 6 of run.py so that the aggregation parameter is one of the following:
- Rouge Score:
    - `rouge1`
    - `rouge2`
    - `rougeL`
- Sentence-BERT embedding cosine similarity:
    - `sbert`
- BERT-Score
    - `bertscore`

As a default, run.py will use `paraphrasing` as the perturbation method and `rougeL` as the aggregation method.

```
spuq = SPUQ(llm=llm, perturbation='paraphrasing', aggregation='rougeL', n_perturb=3)
```
