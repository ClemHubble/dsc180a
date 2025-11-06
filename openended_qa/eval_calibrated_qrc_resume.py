# ===============================================================
# eval_calibrated_qrc_resume.py
# ===============================================================
import os, json, random, numpy as np, torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RESULTS_DIR = "results/calibrated"
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMPERATURES = [0.7, 1.0]  # resume from here
PERTURBATIONS = [0.5, 0.9, 0.0, 0.5, 0.9]  

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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

def generate_answer(prompt, model, tokenizer, temperature, perturbation, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = text[len(prompt):].strip()
    if perturbation > 0:
        words = answer.split()
        n = max(1, int(len(words) * perturbation * 0.3))
        for _ in range(n):
            idx = random.randint(0, len(words) - 1)
            words[idx] = words[idx] + random.choice(["...", "?", " (maybe)", " I think"])
        answer = " ".join(words)
    return answer.strip()

def semantic_equivalence(model, tokenizer, answer, reference):
    judge_prompt = f"""You are a strict grader. 
Determine whether the following two answers mean the same thing.

Answer A: {answer}
Answer B: {reference}

Respond only with 'yes' or 'no'."""
    judgment = generate_answer(judge_prompt, model, tokenizer, temperature=0.3, perturbation=0)
    return "yes" in judgment.lower()

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

def load_squad_v2(n=1000):
    ds = load_dataset("squad_v2", split="validation").shuffle(seed=42).select(range(n))
    return [{"question": d["question"], "answers": d["answers"]["text"], "context": d["context"]} for d in ds]

def load_truthfulqa(n=500):
    ds = load_dataset("truthful_qa", "generation", split="validation").shuffle(seed=42).select(range(n))
    return [{"question": d["question"], "answers": [d["best_answer"]] + d["correct_answers"]} for d in ds]

def load_triviaqa(n=500):
    ds = load_dataset("trivia_qa", "rc", split="validation").shuffle(seed=42).select(range(n))
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

def evaluate_dataset(name, loader, model, tokenizer, temperature, perturbation):
    print(f"\n=== Evaluating {name} | T={temperature} | P={perturbation} ===")
    data = loader()
    correct, confs, results = [], [], []

    for ex in tqdm(data, desc=f"{name} (T={temperature}, P={perturbation})"):
        q = ex["question"]
        context = ex.get("context", "")
        ref_answers = ex["answers"]
        prompt = f"Answer the following question based on the given context (if any).\n\nContext: {context}\n\nQuestion: {q}\nAnswer:"
        pred = generate_answer(prompt, model, tokenizer, temperature, perturbation)
        conf = random.random() * 0.5 + 0.5
        is_correct = any(semantic_equivalence(model, tokenizer, pred, ref) for ref in ref_answers)
        correct.append(int(is_correct))
        confs.append(conf)
        results.append({
            "question": q, "prediction": pred, "answers": ref_answers,
            "is_correct": is_correct, "confidence": conf
        })

    acc = np.mean(correct)
    ece = expected_calibration_error(np.array(correct), np.array(confs))
    qrc90 = np.quantile(np.abs(np.array(correct) - np.array(confs)), 0.9)
    cvar90 = np.mean(sorted(np.abs(np.array(correct) - np.array(confs)))[int(0.9 * len(correct)):])

    run_dir = f"{RESULTS_DIR}/T{temperature}_P{perturbation}/{name}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(f"{run_dir}/metrics.csv", "w") as f:
        f.write("dataset,temperature,perturbation,accuracy,ece,qrc90,cvar90,n\n")
        f.write(f"{name},{temperature},{perturbation},{acc:.3f},{ece:.3f},{qrc90:.3f},{cvar90:.3f},{len(data)}\n")
    print(f"[done] {name} | T={temperature}, P={perturbation}: acc={acc:.3f}, ece={ece:.3f}, qrc90={qrc90:.3f}, cvar90={cvar90:.3f}")

if __name__ == "__main__":
    model, tokenizer = load_llama()
    for T in TEMPERATURES:
        for P in PERTURBATIONS:
            # skip combos that already have output files
            if os.path.exists(f"{RESULTS_DIR}/T{T}_P{P}/triviaqa/metrics.csv"):
                print(f"[skip] Already finished T={T}, P={P}")
                continue
            for name, loader in DATASETS.items():
                try:
                    evaluate_dataset(name, loader, model, tokenizer, T, P)
                except Exception as e:
                    print(f"[!] Skipped {name} (T={T}, P={P}): {e}")
