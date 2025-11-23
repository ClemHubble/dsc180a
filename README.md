# LLama QRC Reliability Project

## 1. Introduction

This repository evaluates reliability, calibration, and robustness of `meta-llama/Llama-3.1-8B-Instruct` across temperature and perturbation settings.  
It contains **two subprojects**:

- **Section 5 — MCQ subproject**: multiple-choice calibration / QRC experiments (ARC, OpenBook). Files: `qrc_experiment.sh`, `qrc_with_heatmap.py`. Results summarized in `qrc_llama_presentation.html`.
- **Section 6 — Open-Ended QA subproject**: open-ended QA calibration & accuracy on SQuADv2, TriviaQA, TruthfulQA. Files: `eval_qrc_reduced_48h.py`, `generate_heatmaps.py`, `run_reduced_48h.sh`, and `heatmaps/` with result images.

The README explains folder structure, environment setup, and how to run each subproject reproducibly.

---

## 2. Folder Structure

```

llama_qrc_project
│
├── mcq/
│   ├── qrc_with_heatmap.py
│   ├── qrc_experiment.sh
│   └── qrc_llama_presentation.html
│
└── openended_qa/
├── heatmaps/
│   ├── CVAR90.png
│   ├── ECE90.png
│   ├── QRC90.png
│   ├── SQuAD_Accuracy.png
│   ├── TriviaQA_Accuracy.png
│   └── TruthfulQA_Accuracy.png
│
├── eval_qrc_reduced_48h.py
├── generate_heatmaps.py
└── run_reduced_48h.sh
│
└── README.md

````

---

## 3. Setup

### Python Environment

Recommended conda setup:

```bash
conda create -n llama_qrc python=3.10 -y
conda activate llama_qrc

pip install numpy pandas matplotlib seaborn tqdm torch transformers datasets scipy
````

### Hugging Face Cache (optional but recommended)

```bash
export HF_HOME="$HOME/.hf_home"
export HF_HUB_CACHE="$HOME/.hf_cache"
export TRANSFORMERS_CACHE="$HOME/.hf_transformers"
export HF_DATASETS_CACHE="$HOME/.hf_datasets"
mkdir -p $HF_HOME $HF_HUB_CACHE $TRANSFORMERS_CACHE $HF_DATASETS_CACHE
```

### Hugging Face Token

```bash
export HF_TOKEN="your_hf_token_here"
# Or:
huggingface-cli login
```

Make sure your token has access to the Llama model family.

---

## 4. Common Notes

* Temperature grid:

  ```
  [0.0, 0.3, 0.7, 1.0]
  ```
* Perturbation grid:

  ```
  [0.0, 0.5, 0.9]
  ```
* Metrics produced:

  * **ECE90**
  * **QRC90**
  * **CVAR90**
  * **Accuracy:** for SQuADv2, TriviaQA, TruthfulQA

---

## 5. MCQ Subproject — Usage

**Location:** `mcq/`

### Files

* `qrc_with_heatmap.py`
  Runs MCQ-based calibration experiments, computes QRC, ECE, and accuracy, and produces heatmaps.

* `qrc_experiment.sh`
  SLURM submission script for running the MCQ experiment.

* `qrc_llama_presentation.html`
  Contains summarized results and visuals.

### Running the MCQ Experiment

Local run:

```bash
cd mcq
python qrc_with_heatmap.py
```

SLURM run:

```bash
cd mcq
sbatch qrc_experiment.sh
```

### Outputs

Inside `mcq/results/<timestamp>_<dataset>/`:

* `<dataset>_tuning.csv`
* `<dataset>_eval.csv`
* `<dataset>_accuracy_heatmap.png`
* `<dataset>_qrc_heatmap.png`
* `<dataset>_ece_heatmap.png`

---

## 6. Open-Ended QA Subproject — Usage

**Location:** `openended_qa/`

### Files

* `eval_qrc_reduced_48h.py`
  Main evaluation script for open-ended QA.

* `generate_heatmaps.py`
  Produces 6 heatmaps:

  * ECE90
  * QRC90
  * CVAR90
  * SQuAD Accuracy
  * TriviaQA Accuracy
  * TruthfulQA Accuracy

* `run_reduced_48h.sh`
  SLURM script for running the full 48h sweep.

### Running the Sweep

Local testing:

```bash
cd openended_qa
python eval_qrc_reduced_48h.py
```

Full cluster run:

```bash
cd openended_qa
sbatch run_reduced_48h.sh
```

### Generating Heatmaps

After results are generated:

```bash
cd openended_qa
python generate_heatmaps.py
```

Outputs appear in:

```
openended_qa/heatmaps/
```

---

## 7. Optional Helpful Commands

### Check HF cache sizes

```bash
du -sh $HF_HOME $HF_HUB_CACHE $TRANSFORMERS_CACHE $HF_DATASETS_CACHE
```

### Clear cache if necessary

```bash
rm -rf ~/.cache/huggingface
```

---

## 8. Troubleshooting

* **Token errors:**
  Ensure `HF_TOKEN` is exported and has model access.

* **Disk quota exceeded:**
  Reassign HF caches to `$HOME` using the setup instructions.

* **Model download failure:**
  Run `huggingface-cli login` interactively.

* **SLURM job finishes instantly:**
  Check `*.err` output for missing env or cache directories.

---


