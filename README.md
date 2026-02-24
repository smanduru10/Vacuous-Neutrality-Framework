# Accepted at EACL 2026
https://arxiv.org/abs/2506.08487

# Bias and Competence Evaluation on the BBQ Dataset

This repository contains code to evaluate instruction-tuned language models (along with quantized variants) on the **BBQ (Bias Benchmark for Question Answering)** dataset. The evaluation covers both **task competence (accuracy, precision, recall, F1)** and **social bias behavior (sDIS, sAmbig, confusion matrices)** across 11 social categories.

---

## 📦 Features

- Supports both full-precision and quantized models via AWQ
- Evaluates across:
  - Social categories (e.g., Race, Gender, SES)
  - Ambiguous vs. disambiguated contexts
  - Positive vs. negative polarity questions
- Computes:
  - Accuracy, Precision, Recall, F1
  - Confusion matrices
  - Bias metrics: sDIS, sAmbig, bias alignment
- Outputs per-category visualizations of prediction types

---

## 📁 Directory Structure

```
.
├── run_bbq.py                       # Main evaluation script for full-precision (FP) models
├── requirements.txt                 # Python dependencies
├── outputs/                         # Directory containing outputs from all model evaluations
│   ├── Microsoft_Phi4_Instruct/     # Outputs for Phi-4 Mini (FP)
│   │   ├── label_type/              # Label confusion matrices (A/B/C)
│   │   └── bias_type/               # Bias-type confusion matrices (Target/Non-target/Unknown)
│   ├── Phi4_mini_Instruct_AWQ/      # Outputs for Phi-4 Mini (AWQ)
│   │   ├── label_type/
│   │   └── bias_type/
│   ├── Qwen3B_Instruct/             # Outputs for Qwen 3B (FP)
│   │   ├── label_type/
│   │   └── bias_type/
│   ├── Qwen3B_Instruct_AWQ/         # Outputs for Qwen 3B (AWQ)
│   │   ├── label_type/
│   │   └── bias_type/
│   ├── Llama3B_Instruct/            # Outputs for LLaMA 3B (FP)
│   │   ├── label_type/
│   │   └── bias_type/
│   ├── Llama_3B_Instruct_AWQ/       # Outputs for LLaMA 3B (AWQ)
│   │   ├── label_type/
│   │   └── bias_type/
└── data/
│   └── bbq_data.csv                 # A sample of preprocessed BBQ dataset
│   └── crows_pairs_anonymized.csv   # CrowS-Pairs Dataset CSV File
│
└── crowspairs.ipynb                 # CrowS-Pairs Notebook
└── stereoSet_inter.ipynb            # StereoSet - Inter Notebook
└── stereoSet_intra.ipynb            # StereoSet - Intra Notebook
└── finetune_csqa.ipynb              # Finetuning Code on CommonSenseQA

```


## 📊 Dataset

This script expects a preprocessed CSV file with the following columns:

- `context`, `question`, `ans0`, `ans1`, `ans2`, `label`
- `category`, `context_condition`, `question_polarity`
- `target`: list of stereotypes (as stringified list)
- `answer_info`: dictionary mapping answers to stereotype annotations

You can use `datasets.load_dataset("heegyu/bbq", split="test")` to generate this format or load your own variant.
You can use `datasets.load_dataset("McGill-NLP/stereoset", "intersentence")` to access the StereoSet Inter dataset.
You can use `datasets.load_dataset("McGill-NLP/stereoset", "intrasentence")` to access the StereoSet Intra dataset.
You can use `datasets.load_dataset("tau/commonsense_qa")` to access the CommonSenseQA dataset.

## 🚀 Running the Script

Update `<data path>` in the script to point to your BBQ dataset CSV file, then run:

```bash
pip install -r requirements.txt
python run_bbq.py

