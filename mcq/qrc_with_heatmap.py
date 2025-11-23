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
    Extract answer letter (A, B, C, or D) from generated text.
    """
    response = generated_text[len(prompt):].strip()
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1)
    first_words = response.split()[:20]
    for word in first_words:
        clean = word.strip('.,!?():"\'-').upper()
        if clean in ['A','B','C','D']:
            return clean
    answer_match = re.search(r'answer\s+is\s+([A-D])', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    last_word = response.split()[-1].replace(".","").upper() if response.split() else ""
    if last_word in ['A','B','C','D']:
        return last_word
    return ""

def make_prompt(dataset_name, example):
    if dataset_name.startswith("arc"):
        question = example["question"]
        choices = example["choices"]["text"]
        labels = example["choices"]["label"]
        answer = example.get("answerKey","")
        choice_str = "\n".join([f"{label}: {text}" for label,text in zip(labels,choices)])
        return f"Question: {question}\n{choice_str}\nAnswer:", answer
    elif dataset_name == "openbookqa":
        if not isinstance(example, dict):
            example = dict(example)
        qstem = example.get("question_stem","")
        answer = example.get("answerKey","")
        choices_field = example.get("choices",{})
        if isinstance(choices_field, dict):
            choices = choices_field.get("text",[])
            labels = choices_field.get("label",[])
        elif isinstance(choices_field, list):
            choices = [c.get("text","") for c in choices_field]
            labels = [c.get("label","") for c in choices_field]
        else:
            choices, labels = [], []
        choice_str = "\n".join([f"{label}: {text}" for label,text in zip(labels,choices)])
        return f"Question: {qstem}\n{choice_str}\nAnswer:", answer
    else:
        raise ValueError(f"Unrecognized example format: {example.keys()}")

def compute_accuracy(preds,golds):
    correct = sum([p==g for p,g in zip(preds,golds)])
    return correct/len(golds) if golds else 0.0

def expected_calibration_error(confidences,preds,golds,n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo,hi = bins[i], bins[i+1]
        idx = [j for j,c in enumerate(confidences) if lo<=c<hi]
        if not idx:
            continue
        acc = np.mean([preds[j]==golds[j] for j in idx])
        conf = np.mean([confidences[j] for j in idx])
        ece += abs(acc-conf)*len(idx)/len(confidences)
    return ece

def compute_qrc_bound(losses, quantile=0.9, alpha=0.05):
    n = len(losses)
    if n==0:
        return 1.0
    sorted_losses = np.sort(losses)
    k = int(np.ceil(quantile*n))
    num_success = np.sum(sorted_losses[:k])
    return clopper_pearson_upper(num_success,k,alpha=alpha)

# ------------------------------------------------------------
# Baseline evaluation
# ------------------------------------------------------------

def run_baseline_eval(model, tokenizer, dataset_name, results_dir, device):
    print(f"\n=== Running BASELINE for {dataset_name} (T=1.0, top_p=0.9) ===")
    try:
        if dataset_name=="arc_easy":
            dataset = load_dataset("ai2_arc","ARC-Easy")
        elif dataset_name=="arc_challenge":
            dataset = load_dataset("ai2_arc","ARC-Challenge")
        elif dataset_name=="openbookqa":
            dataset = load_dataset("openbookqa")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None

    data = list(dataset["test"])
    random.shuffle(data)
    eval_size = len(data)//2
    baseline_data = data[:eval_size]

    preds, golds, losses, confs = [], [], [], []
    failed_count = 0
    debug_count = 0

    # --- Letter to token mapping
    letter_to_token_id = {ltr: tokenizer.encode(ltr, add_special_tokens=False)[0] for ltr in ["A","B","C","D"]}

    for ex in tqdm(baseline_data, desc="baseline"):
        try:
            prompt, gold = make_prompt(dataset_name, ex)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    temperature=1.0,
                    do_sample=True,
                    top_p=0.9,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
            pred = extract_answer(text, prompt)
            gold_upper = gold.strip().upper()

            if debug_count < 3:
                print(f"\n--- Example {debug_count+1} ---")
                print(f"Gold: {gold_upper}")
                print(f"Generated: {text[len(prompt):].strip()[:100]}...")
                print(f"Extracted: {pred}")
                debug_count += 1

            preds.append(pred)
            golds.append(gold_upper)

            # --- Confidence from last token logits
            last_logits = output.scores[-1][0]  # [vocab_size]
            pred_token_id = letter_to_token_id.get(pred,None)
            conf = torch.softmax(last_logits, dim=-1)[pred_token_id].item() if pred_token_id else 0.0
            confs.append(conf)

            # Loss for QRC
            loss_i = 0 if pred==gold_upper else 1
            losses.append(loss_i)

        except Exception as e:
            failed_count += 1
            if failed_count < 5:
                print(f"Error: {e}")
            continue

    if failed_count>0:
        print(f"⚠️ {failed_count} examples failed")

    acc = compute_accuracy(preds, golds)
    ece = expected_calibration_error(confs, preds, golds)
    qrc = compute_qrc_bound(losses)

    print(f"✅ BASELINE {dataset_name}: Acc={acc:.3f}, ECE={ece:.3f}, QRC={qrc:.3f}")

    return {
        "dataset": dataset_name,
        "config": "baseline",
        "temperature": 1.0,
        "top_p": 0.9,
        "accuracy": acc,
        "ece": ece,
        "qrc_bound": qrc,
        "n_samples": len(golds)
    }

# ------------------------------------------------------------
# Grid search evaluation with heatmaps
# ------------------------------------------------------------

def run_qrc_eval_with_heatmap(model, tokenizer, dataset_name, results_dir, device, timestamp):
    print(f"\n=== Running GRID SEARCH for {dataset_name} ===")
    try:
        if dataset_name=="arc_easy":
            dataset = load_dataset("ai2_arc","ARC-Easy")
        elif dataset_name=="arc_challenge":
            dataset = load_dataset("ai2_arc","ARC-Challenge")
        elif dataset_name=="openbookqa":
            dataset = load_dataset("openbookqa")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None

    data = list(dataset["test"])
    random.shuffle(data)
    half = len(data)//2
    tune_split, eval_split = data[:half], data[half:]
    print(f"n_total={len(data)}, tune={len(tune_split)}, eval={len(eval_split)}")

    # --- Letter to token mapping
    letter_to_token_id = {ltr: tokenizer.encode(ltr, add_special_tokens=False)[0] for ltr in ["A","B","C","D"]}

    temps = [0.3,0.5,0.7,1.0,1.2]
    top_ps = [0.7,0.8,0.9,0.95,1.0]
    best_qrc = 1.0
    best_config = None
    tuning_records = []

    print(f"Testing {len(temps)} × {len(top_ps)} configurations...")

    for T in temps:
        for top_p in top_ps:
            preds, golds, losses, confs = [], [], [], []
            desc = f"T{T:.1f}_p{top_p:.2f}"

            for ex in tqdm(tune_split, desc=desc, leave=False):
                try:
                    prompt,gold = make_prompt(dataset_name,ex)
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=32,
                            temperature=T,
                            do_sample=True,
                            top_p=top_p,
                            output_scores=True,
                            return_dict_in_generate=True
                        )
                    text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                    pred = extract_answer(text, prompt)
                    gold_upper = gold.strip().upper()
                    preds.append(pred)
                    golds.append(gold_upper)

                    # Confidence
                    last_logits = output.scores[-1][0]
                    pred_token_id = letter_to_token_id.get(pred,None)
                    conf = torch.softmax(last_logits, dim=-1)[pred_token_id].item() if pred_token_id else 0.0
                    confs.append(conf)

                    loss_i = 0 if pred==gold_upper else 1
                    losses.append(loss_i)
                except:
                    continue

            acc_tune = compute_accuracy(preds,golds)
            ece_tune = expected_calibration_error(confs,preds,golds)
            qrc_bound = compute_qrc_bound(losses)

            tuning_records.append({
                "T": T, "top_p": top_p,
                "acc_tune": acc_tune,
                "ece_tune": ece_tune,
                "qrc_bound": qrc_bound
            })

            if qrc_bound < best_qrc:
                best_qrc = qrc_bound
                best_config = (T, top_p)

    print(f"\n🎯 Best config: T={best_config[0]}, top_p={best_config[1]}, QRC={best_qrc:.3f}")

    # --- Evaluation with best config
    T_best, top_p_best = best_config
    preds_eval, golds_eval, losses_eval, confs_eval = [],[],[],[]
    for ex in tqdm(eval_split, desc="eval"):
        try:
            prompt,gold = make_prompt(dataset_name,ex)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    temperature=T_best,
                    do_sample=True,
                    top_p=top_p_best,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
            pred = extract_answer(text, prompt)
            gold_upper = gold.strip().upper()
            preds_eval.append(pred)
            golds_eval.append(gold_upper)

            last_logits = output.scores[-1][0]
            pred_token_id = letter_to_token_id.get(pred,None)
            conf = torch.softmax(last_logits, dim=-1)[pred_token_id].item() if pred_token_id else 0.0
            confs_eval.append(conf)

            loss_i = 0 if pred==gold_upper else 1
            losses_eval.append(loss_i)
        except:
            continue

    acc_eval = compute_accuracy(preds_eval,golds_eval)
    ece_eval = expected_calibration_error(confs_eval,preds_eval,golds_eval)
    qrc_eval = compute_qrc_bound(losses_eval)

    # --- Save results
    dataset_dir = os.path.join(results_dir,f"{timestamp}_{dataset_name}")
    os.makedirs(dataset_dir,exist_ok=True)
    pd.DataFrame(tuning_records).to_csv(os.path.join(dataset_dir,f"{dataset_name}_tuning.csv"),index=False)
    pd.DataFrame([{
        "dataset": dataset_name,
        "accuracy": acc_eval,
        "ece": ece_eval,
        "qrc_bound": qrc_eval,
        "T_best": T_best,
        "top_p_best": top_p_best,
        "n_eval": len(golds_eval)
    }]).to_csv(os.path.join(dataset_dir,f"{dataset_name}_eval.csv"),index=False)

    # --- Heatmaps
    df_grid = pd.DataFrame(tuning_records)
    for metric, cmap, fname in [("acc_tune","YlGnBu","accuracy_heatmap.png"),
                                ("qrc_bound","OrRd_r","qrc_heatmap.png"),
                                ("ece_tune","RdYlGn_r","ece_heatmap.png")]:
        heatmap_data = df_grid.pivot(index="T", columns="top_p", values=metric)
        plt.figure(figsize=(8,6))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap=cmap, cbar_kws={'label':metric})
        plt.title(f"{dataset_name} {metric} heatmap (T vs top_p)")
        plt.xlabel("top_p")
        plt.ylabel("Temperature")
        plt.tight_layout()
        plt.savefig(os.path.join(dataset_dir,f"{dataset_name}_{fname}"), dpi=150)
        plt.close()

    print(f"✅ {dataset_name}: Acc={acc_eval:.3f}, ECE={ece_eval:.3f}, QRC={qrc_eval:.3f}")
    return {
        "dataset": dataset_name,
        "accuracy": acc_eval,
        "ece": ece_eval,
        "qrc_eval": qrc_eval,
        "T_best": T_best,
        "top_p_best": top_p_best
    }

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__=="__main__":
    start_time = time.time()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results_dir = os.path.join(os.getcwd(),"results")
    os.makedirs(results_dir,exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded!")

    datasets = ["arc_easy","arc_challenge","openbookqa"]
    all_results = []
    baseline_results = []

    # --- Baseline
    print("\n=== PHASE 1: BASELINE EVALUATION ===")
    for ds_name in datasets:
        baseline_res = run_baseline_eval(model, tokenizer, ds_name, results_dir, device)
        if baseline_res:
            baseline_results.append(baseline_res)
    pd.DataFrame(baseline_results).to_csv(os.path.join(results_dir,f"{timestamp}_baseline_results.csv"),index=False)

    # --- Grid search
    print("\n=== PHASE 2: GRID SEARCH OPTIMIZATION ===")
    for ds_name in datasets:
        res = run_qrc_eval_with_heatmap(model, tokenizer, ds_name, results_dir, device, timestamp)
        if res:
            all_results.append(res)
    pd.DataFrame(all_results).to_csv(os.path.join(results_dir,f"{timestamp}_optimized_results.csv"),index=False)

    # --- Comparison table
    comparison = []
    for baseline, optimized in zip(baseline_results, all_results):
        comparison.append({
            "dataset": baseline["dataset"],
            "baseline_acc": baseline["accuracy"],
            "baseline_qrc": baseline["qrc_bound"],
            "optimized_acc": optimized["accuracy"],
            "optimized_qrc": optimized["qrc_eval"],
            "acc_improvement": optimized["accuracy"]-baseline["accuracy"],
            "qrc_improvement": baseline["qrc_bound"]-optimized["qrc_eval"],
            "best_T": optimized["T_best"],
            "best_top_p": optimized["top_p_best"]
        })
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(os.path.join(results_dir,f"{timestamp}_comparison.csv"),index=False)
    print("\n" + comparison_df.to_string(index=False))

    metadata = {
        "timestamp": timestamp,
        "device": device,
        "datasets": datasets,
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "baseline_config": {"temperature":1.0,"top_p":0.9},
        "grid_search": {"temperatures":[0.3,0.5,0.7,1.0,1.2],"top_p_values":[0.7,0.8,0.9,0.95,1.0]},
        "runtime_minutes": (time.time()-start_time)/60
    }
    with open(os.path.join(results_dir,f"{timestamp}_metadata.json"),"w") as f:
        json.dump(metadata,f,indent=2)

    print(f"\n✅ All done! Total runtime: {(time.time()-start_time)/60:.2f} minutes")
    print(f"📁 Results saved to: {results_dir}")
