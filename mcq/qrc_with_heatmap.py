# qrc_with_heatmap.py

import os
import json
import time
import random
import re
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def clopper_pearson_upper(k, n, alpha=0.05):
    if n == 0:
        return 1.0
    return beta.ppf(1 - alpha / 2, k + 1, n - k)

def extract_answer(generated_text, prompt):
    """
    Extract A/B/C/D answer from model output.
    Works on most instruction-tuned models.
    """
    response = generated_text[len(prompt):].strip()

    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1)

    first_words = response.split()[:20]
    for w in first_words:
        clean = w.strip('.,!?():"\'-').upper()
        if clean in ['A','B','C','D']:
            return clean

    ans_match = re.search(r'answer\s+is\s+([A-D])', response, re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).upper()

    if response:
        last = response.split()[-1].replace(".","").upper()
        if last in ['A','B','C','D']:
            return last

    return ""

def make_prompt(dataset_name, example):
    """
    Build standard MCQ prompt for ARC and OpenBookQA datasets.
    """
    if dataset_name.startswith("arc"):
        q = example["question"]
        choices = example["choices"]["text"]
        labels = example["choices"]["label"]
        answer = example.get("answerKey","")
        choice_str = "\n".join([f"{l}: {t}" for l,t in zip(labels, choices)])
        return f"Question: {q}\n{choice_str}\nAnswer:", answer

    elif dataset_name == "openbookqa":
        q = example["question_stem"]
        answer = example["answerKey"]
        c = example["choices"]
        labels = c["label"]
        texts = c["text"]
        choice_str = "\n".join([f"{l}: {t}" for l,t in zip(labels,texts)])
        return f"Question: {q}\n{choice_str}\nAnswer:", answer

    else:
        raise ValueError("Unsupported dataset.")

def compute_accuracy(preds, golds):
    return sum([p==g for p,g in zip(preds,golds)]) / len(golds)

def expected_calibration_error(conf, preds, golds, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        idx = [j for j,c in enumerate(conf) if lo <= c < hi]
        if not idx:
            continue
        acc = np.mean([preds[j]==golds[j] for j in idx])
        mean_conf = np.mean([conf[j] for j in idx])
        ece += abs(acc - mean_conf) * len(idx) / len(conf)
    return ece

def compute_qrc_bound(losses, quantile=0.9, alpha=0.05):
    n = len(losses)
    if n == 0:
        return 1.0
    sorted_losses = np.sort(losses)
    k = int(np.ceil(quantile * n))
    successes = np.sum(sorted_losses[:k])
    return clopper_pearson_upper(successes, k, alpha=alpha)

# ------------------------------------------------------------
# Baseline evaluation
# ------------------------------------------------------------

def run_baseline(model, tok, dataset_name, results_dir, device):
    print(f"\n=== Baseline run for {dataset_name} ===")

    if dataset_name == "arc_easy":
        ds = load_dataset("ai2_arc","ARC-Easy")
    elif dataset_name == "arc_challenge":
        ds = load_dataset("ai2_arc","ARC-Challenge")
    elif dataset_name == "openbookqa":
        ds = load_dataset("openbookqa")
    else:
        raise ValueError("Unknown dataset")

    data = list(ds["test"])
    random.shuffle(data)

    half = len(data)//2
    data = data[:half]

    preds, golds, losses, confs = [], [], [], []

    for ex in tqdm(data, desc="baseline"):
        prompt, gold = make_prompt(dataset_name, ex)
        inp = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=32,
                temperature=1.0,
                top_p=0.9,
                do_sample=True
            )

        text = tok.decode(out[0], skip_special_tokens=True)
        pred = extract_answer(text, prompt)
        gold = gold.strip().upper()

        preds.append(pred)
        golds.append(gold)
        loss = 0 if pred==gold else 1
        losses.append(loss)
        confs.append(1.0-loss)

    acc = compute_accuracy(preds,golds)
    ece = expected_calibration_error(confs,preds,golds)
    qrc = compute_qrc_bound(losses)

    print(f"Baseline: Acc={acc:.3f}, ECE={ece:.3f}, QRC={qrc:.3f}")

    return {
        "dataset": dataset_name,
        "accuracy": acc,
        "ece": ece,
        "qrc": qrc,
        "n": len(golds)
    }

# ------------------------------------------------------------
# Grid search + Heatmaps
# ------------------------------------------------------------

def run_grid_search(model, tok, dataset_name, results_dir, device, timestamp):

    print(f"\n=== Grid search for {dataset_name} ===")

    if dataset_name == "arc_easy":
        ds = load_dataset("ai2_arc","ARC-Easy")
    elif dataset_name == "arc_challenge":
        ds = load_dataset("ai2_arc","ARC-Challenge")
    elif dataset_name == "openbookqa":
        ds = load_dataset("openbookqa")
    else:
        raise ValueError("Unknown dataset")

    data = list(ds["test"])
    random.shuffle(data)
    half = len(data)//2
    tune, eval_set = data[:half], data[half:]

    temps = [0.3, 0.5, 0.7, 1.0, 1.2]
    top_ps = [0.7, 0.8, 0.9, 0.95, 1.0]

    records = []
    best_qrc = 1.0
    best_cfg = None

    for T in temps:
        for p in top_ps:
            preds, golds, losses, confs = [], [], [], []
            desc = f"T={T}, p={p}"

            for ex in tqdm(tune, desc=desc, leave=False):
                prompt, gold = make_prompt(dataset_name, ex)
                inp = tok(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    out = model.generate(
                        **inp,
                        max_new_tokens=32,
                        temperature=T,
                        top_p=p,
                        do_sample=True
                    )

                text = tok.decode(out[0], skip_special_tokens=True)
                pred = extract_answer(text, prompt)
                gold = gold.strip().upper()
                loss = 0 if pred==gold else 1

                preds.append(pred)
                golds.append(gold)
                losses.append(loss)
                confs.append(1.0-loss)

            acc = compute_accuracy(preds,golds)
            ece = expected_calibration_error(confs,preds,golds)
            qrc = compute_qrc_bound(losses)

            records.append({
                "T": T,
                "top_p": p,
                "acc": acc,
                "ece": ece,
                "qrc": qrc
            })

            if qrc < best_qrc:
                best_qrc = qrc
                best_cfg = (T, p)

    df = pd.DataFrame(records)
    outdir = os.path.join(results_dir, f"{timestamp}_{dataset_name}")
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, f"{dataset_name}_grid.csv"), index=False)

    # ------------------------------
    # Heatmap Generator
    # ------------------------------
    def heatmap(values, name, cmap):
        table = df.pivot(index="T", columns="top_p", values=values)
        plt.figure(figsize=(8,6))
        sns.heatmap(table, annot=True, fmt=".3f", cmap=cmap)
        plt.title(f"{dataset_name} {name} Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{dataset_name}_{values}_heatmap.png"), dpi=150)
        plt.close()

    # Generate 6 heatmaps
    # Tune splits: acc, ece, qrc
    heatmap("acc", "Accuracy", "YlGnBu")
    heatmap("ece", "ECE", "RdYlGn_r")
    heatmap("qrc", "QRC", "OrRd_r")

    # Also save evaluation using best config
    Tbest, pbest = best_cfg
    preds, golds, losses, confs = [], [], [], []

    for ex in tqdm(eval_set, desc="eval"):
        prompt, gold = make_prompt(dataset_name, ex)
        inp = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=32,
                temperature=Tbest,
                top_p=pbest,
                do_sample=True
            )

        text = tok.decode(out[0], skip_special_tokens=True)
        pred = extract_answer(text, prompt)
        loss = 0 if pred==gold.strip().upper() else 1

        preds.append(pred)
        golds.append(gold.upper())
        losses.append(loss)
        confs.append(1.0-loss)

    acc_eval = compute_accuracy(preds,golds)
    ece_eval = expected_calibration_error(confs,preds,golds)
    qrc_eval = compute_qrc_bound(losses)

    pd.DataFrame([{
        "dataset": dataset_name,
        "accuracy": acc_eval,
        "ece": ece_eval,
        "qrc": qrc_eval,
        "T_best": Tbest,
        "top_p_best": pbest
    }]).to_csv(os.path.join(outdir, f"{dataset_name}_best_eval.csv"), index=False)

    return {
        "dataset": dataset_name,
        "acc": acc_eval,
        "ece": ece_eval,
        "qrc": qrc_eval,
        "best_T": Tbest,
        "best_top_p": pbest
    }

# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

if __name__ == "__main__":
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model (change as needed)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    datasets = ["arc_easy", "arc_challenge", "openbookqa"]

    baseline = []
    optimized = []

    # Baseline
    for d in datasets:
        baseline.append(run_baseline(model, tokenizer, d, results_dir, device))

    pd.DataFrame(baseline).to_csv(
        os.path.join(results_dir, f"{timestamp}_baseline.csv"), index=False
    )

    # Grid + heatmaps
    for d in datasets:
        optimized.append(run_grid_search(model, tokenizer, d, results_dir, device, timestamp))

    pd.DataFrame(optimized).to_csv(
        os.path.join(results_dir, f"{timestamp}_optimized.csv"), index=False
    )
