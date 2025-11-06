import os
import json
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# =============================
# Config
# =============================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RESULTS_DIR = "results/fast_fixed"
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# Model loader
# =============================
def load_llama():
    print(f"[model] loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

# =============================
# Generate helper
# =============================
def generate_answer(prompt, model, tokenizer, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.3, top_p=0.9)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # remove the prompt portion
    return text[len(prompt):].strip()

# =============================
# Semantic equivalence scorer
# =============================
def semantic_equivalence(model, tokenizer, answer, reference):
    """Returns True if model judges semantic equivalence, else False"""
    judge_prompt = f"""You are a strict grader. 
Determine whether the following two answers mean the same thing.

Answer A: {answer}
Answer B: {reference}

Respond only with 'yes' or 'no'."""
    judgment = generate_answer(judge_prompt, model, tokenizer, max_new_tokens=10).lower()
    return "yes" in judgment

# =============================
# Metrics
# =============================
def expected_calibration_error(preds, confs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (confs >= bins[i]) & (confs < bins[i+1])
        if mask.any():
            acc = preds[mask].mean()
            conf = confs[mask].mean()
            ece += np.abs(acc - conf) * mask.mean()
    return ece

# =============================
# Dataset loaders
# =============================
def load_squad_v2(n=1000):
    ds = load_dataset("squad_v2", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    return [{"question": d["question"], "answers": d["answers"]["text"], "context": d["context"]} for d in ds]

def load_truthfulqa(n=500):
    ds = load_dataset("truthful_qa", "generation", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    return [{"question": d["question"], "answers": [d["best_answer"]] + d["correct_answers"]} for d in ds]

def load_triviaqa(n=500):
    ds = load_dataset("trivia_qa", "rc", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    data = []
    for d in ds:
        if "question" in d and "answer" in d:
            answers = d["answer"].get("aliases", [])
            if answers:
                data.append({"question": d["question"], "answers": answers})
    return data[:n]

DATASETS = {
    "squad_v2": load_squad_v2,
    "truthfulqa": load_truthfulqa,
    "triviaqa": load_triviaqa
}

# =============================
# Main evaluation
# =============================
def evaluate_dataset(name, loader, model, tokenizer):
    print(f"\n=== Evaluating {name} ===")
    data = loader()
    correct = []
    confs = []
    results = []

    for ex in tqdm(data, desc=name):
        q = ex["question"]
        context = ex.get("context", "")
        ref_answers = ex["answers"]

        prompt = f"Answer the following question based on the given context (if any).\n\nContext: {context}\n\nQuestion: {q}\nAnswer:"
        pred = generate_answer(prompt, model, tokenizer)

        # Confidence proxy = 1 - normalized perplexity over last sentence
        conf = random.random() * 0.5 + 0.5  # mock confidence (for now)
        is_correct = any(semantic_equivalence(model, tokenizer, pred, ref) for ref in ref_answers)

        correct.append(int(is_correct))
        confs.append(conf)
        results.append({
            "question": q,
            "prediction": pred,
            "answers": ref_answers,
            "is_correct": is_correct,
            "confidence": conf
        })

    acc = np.mean(correct)
    ece = expected_calibration_error(np.array(correct), np.array(confs))
    qrc90 = np.quantile(np.abs(np.array(correct) - np.array(confs)), 0.9)
    cvar90 = np.mean(sorted(np.abs(np.array(correct) - np.array(confs)))[int(0.9 * len(correct)):])

    os.makedirs(f"{RESULTS_DIR}/{name}", exist_ok=True)
    with open(f"{RESULTS_DIR}/{name}/results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    with open(f"{RESULTS_DIR}/{name}/metrics.csv", "w") as f:
        f.write("dataset,accuracy,ece,qrc90,cvar90,n\n")
        f.write(f"{name},{acc:.3f},{ece:.3f},{qrc90:.3f},{cvar90:.3f},{len(data)}\n")

    print(f"[done] {name}: acc={acc:.3f}, ece={ece:.3f}, qrc90={qrc90:.3f}, cvar90={cvar90:.3f}")
    return acc, ece


# =============================
# Entrypoint
# =============================
if __name__ == "__main__":
    model, tokenizer = load_llama()
    for name, loader in DATASETS.items():
        try:
            evaluate_dataset(name, loader, model, tokenizer)
        except Exception as e:
            print(f"[!] Skipped {name}: {e}")
