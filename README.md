# 🐟 AquaSense-Agent — Notebook Implementation Guide

> IEEE Research Implementation | Multimodal Fish Disease Diagnosis via IoT Sensors, Vision, and Continual RAG

---

## 📋 Table of Contents

- [Overview](#overview)
- [Disease Categories](#disease-categories)
- [Notebook Structure](#notebook-structure)
  - [Notebook 1 — Data Preparation & EDA](#notebook-1--data-preparation--eda)
  - [Notebook 2 — Visual Pathway (CNN Ensemble)](#notebook-2--visual-pathway-cnn-ensemble)
  - [Notebook 3 — Text Pathway (FastText)](#notebook-3--text-pathway-fasttext)
  - [Notebook 4 — Environmental-Disease Correlation (XGBoost)](#notebook-4--environmental-disease-correlation-xgboost)
  - [Notebook 5 — Late Bayesian Fusion](#notebook-5--late-bayesian-fusion)
  - [Notebook 6 — Continual RAG Pipeline](#notebook-6--continual-rag-pipeline)
  - [Notebook 7 — Supervisor Agent & End-to-End Pipeline](#notebook-7--supervisor-agent--end-to-end-pipeline)
- [Key Libraries](#key-libraries)
- [Data Flow Between Notebooks](#data-flow-between-notebooks)

---

## Overview

AquaSense-Agent is a multimodal diagnostic agent for fish disease detection in aquaculture.  
It fuses three input modalities — **visual (images)**, **textual (farmer descriptions)**, and **environmental (IoT sensors)** — through a Late Bayesian Fusion layer, backed by a Continual RAG knowledge base and a Supervisor Agent that routes queries intelligently.

---

## Disease Categories

The system classifies fish into **7 categories** from the AquaGPT dataset:

| # | Disease |
|---|---------|
| 1 | Bacterial Red Disease |
| 2 | Aeromoniasis |
| 3 | Bacterial Gill Disease |
| 4 | Saprolegniasis |
| 5 | Healthy Fish |
| 6 | Parasitic Diseases |
| 7 | White Tail Disease |

---

## Notebook Structure

### Notebook 1 — Data Preparation & EDA

**File:** `01_data_preparation_eda.ipynb`

**Purpose:** Prepare both image and sensor data, compute anomaly features, and build the 10-dimensional sensor feature vector used by XGBoost downstream.

**What it does:**

- Load the AquaGPT image dataset across all 7 disease categories
- Visualize class distribution and sample images per class
- Simulate / load the IoT sensor dataset — readings recorded every **60 seconds** for:
  - Dissolved Oxygen (DO), pH, Temperature, Total Ammonia Nitrogen (TAN), Turbidity
- Compute the **24-hour rolling z-score anomaly score** per sensor channel
- Apply a **5-minute median filter** to remove transient spikes
- Visualize sensor readings over time and flag detected anomalies
- Build the **10-dimensional feature vector:**

```
[do, ph, temp, tan, turbidity, a_do, a_ph, a_temp, a_tan, a_turb]
 ↑ raw readings ────────────── ↑ anomaly scores ─────────────────
```

**Outputs:**
- `sensor_features.npy` — 10-dim feature matrix (N × 10)
- `sensor_labels.npy` — disease labels aligned to sensor readings
- `eda_class_distribution.png`
- `sensor_anomaly_timeline.png`

---

### Notebook 2 — Visual Pathway (CNN Ensemble)

**File:** `02_visual_pathway_cnn_ensemble.ipynb`

**Purpose:** Train a stacking CNN ensemble on fish disease images and output `P_cnn` — a probability distribution over the 7 classes for each image.

**What it does:**

- Apply **YOLOv11n** for ROI detection on fish images (confidence threshold **θ = 0.7**), then crop the detected region
- If no detection exceeds the threshold, fall back to the full image
- Fine-tune three CNN backbones on the cropped ROIs:
  - **MobileNetV2**
  - **DenseNet121**
  - **VGG16**
- Build a **stacking meta-learner** (Logistic Regression) that takes the concatenated softmax outputs of all 3 models (21 features) and produces `P_cnn`
- Evaluate each model independently and as an ensemble:
  - Per-class Precision, Recall, F1
  - Macro-F1

**Outputs:**
- `P_cnn.npy` — ensemble probability output (N_test × 7) → **used in Notebook 5**
- `best_MobileNetV2.pth`, `best_DenseNet121.pth`, `best_VGG16.pth`
- `visual_pathway_summary.csv`
- `confusion_matrix_*.png`

---

### Notebook 3 — Text Pathway (FastText)

**File:** `03_text_pathway_fasttext.ipynb`

**Purpose:** Train a text classifier on informal farmer symptom descriptions and output `P_text` — a probability distribution over the 7 disease classes.

**What it does:**

- Train a **FastText** classifier on farmer symptom descriptions (informal language, regional phrasing, misspellings)
- Use **character n-gram embeddings** for robustness against out-of-vocabulary words
- Output `P_text` — probability distribution over 7 disease classes per text input
- Evaluate using F1-score, targeting **82–92%** macro-F1 (following SHREADS baseline)

**Outputs:**
- `P_text.npy` — text pathway probability output (N_test × 7) → **used in Notebook 5**
- `fasttext_model.bin`
- `text_pathway_report.txt`

---

### Notebook 4 — Environmental-Disease Correlation (XGBoost)

**File:** `04_environmental_xgboost.ipynb`

**Purpose:** Train an XGBoost classifier on the 10-dim sensor feature vector to produce `P_sensor` — a disease-risk prior conditioned on current water quality.

**What it does:**

- Train **XGBoost** on the 10-dimensional sensor feature vector from Notebook 1
- Requires a minimum of **50 confirmed sensor–disease co-occurrence events per class**
- Output `P_sensor` — disease-risk prior conditioned on current water quality readings
- Run **SHAP analysis** to rank feature importance
  - Paper expects `a_do` (DO anomaly) and `a_tan` (TAN anomaly) to rank highest
- Evaluate: macro-averaged accuracy on held-out sensor–disease test set
- Simulate **monthly retraining** on accumulated new tank data

**Outputs:**
- `P_sensor.npy` — sensor pathway probability output (N_test × 7) → **used in Notebook 5**
- `xgboost_model.json`
- `shap_feature_importance.png`
- `sensor_model_summary.csv`

---

### Notebook 5 — Late Bayesian Fusion

**File:** `05_late_bayesian_fusion.ipynb`

**Purpose:** Fuse `P_cnn`, `P_text`, and `P_sensor` using weighted Bayesian fusion and run ablation experiments across 4 configurations.

**What it does:**

- Implement the fusion formula **(Equation 1 from the paper):**

```
P̄_k = (w_cnn · P_cnn,k  +  w_text · P_text,k  +  w_sensor · P_sensor,k) / Z
```

where `Z` is a normalisation constant and weights are derived from each modality's independent validation Macro-F1.

- Determine `w_cnn`, `w_text`, `w_sensor` from validation-set F1 scores
- Run the **4 ablation configurations:**

| Config | Modalities Used |
|--------|----------------|
| **C1** | Image only |
| **C2** | Image + Text |
| **C3** | Image + Sensor |
| **C4** | Full Triple-Modal — AquaSense-Agent |

- Apply **McNemar's test** (scipy) for statistical significance of C4 vs C1
- Report per-class and macro-F1 for all 4 configs (reproducing **Table II** of the paper)

**Outputs:**
- `P_fused.npy` — final fused probability output → **used in Notebook 7**
- `ablation_results_table2.csv`
- `mcnemar_test_results.txt`

---

### Notebook 6 — Continual RAG Pipeline

**File:** `06_continual_rag_pipeline.ipynb`

**Purpose:** Build and incrementally update a hybrid retrieval knowledge base with EWC regularization to prevent catastrophic forgetting.

**What it does:**

- Build a **hybrid BM25 + FAISS** knowledge base from three sources:
  - AQUA public QA dataset
  - Peer-reviewed literature on tilapia disease
  - Lab-confirmed case records
- Implement **semantic contextual chunking:**
  - Split documents by cosine distance thresholds between sentence embeddings
  - Generate LLM-produced chunk summaries for each segment
- Implement **Reciprocal Rank Fusion (RRF)** — **(Equation 2)** — to merge BM25 and FAISS rankings:
```
RRF_score(d) = Σ_r  1 / (k + rank_r(d))
```
- Implement **EWC regularization** — **(Equation 3)** — to protect existing knowledge during updates:
```
L_EWC = L_new  +  (λ/2) · Σ_j  F_j · (θ_j − θ*_j)²
```
- Simulate **3 monthly update cycles** and measure:
  - **ROUGE-L** vs expert reference answers
  - **F1 retention** on old disease categories (catastrophic forgetting rate)
  - Expert-rated advisory quality (Likert scale simulation)

**Outputs:**
- `faiss_index.bin` — FAISS vector index
- `bm25_index.pkl` — BM25 index
- `rag_rouge_scores.csv`
- `forgetting_rate_over_cycles.png`

---

### Notebook 7 — Supervisor Agent & End-to-End Pipeline

**File:** `07_supervisor_agent_e2e.ipynb`

**Purpose:** Wire all modules together under a Supervisor Agent that routes queries, constructs LLM prompts, generates advisory output, and measures system latency.

**What it does:**

- Implement the **intent router** classifying incoming queries into:

| Intent | Trigger |
|--------|---------|
| `DIAGNOSIS` | Farmer uploads image and/or describes symptoms |
| `WATER_QUALITY` | Sensor anomaly fires automatically |
| `INFORMATION` | Farmer asks a general disease question |
| `GREETING` | Opening message / casual input |

- Wire all 4 modules into a single end-to-end pipeline:
  - Visual Pathway → `P_cnn`
  - Text Pathway → `P_text`
  - Sensor Pathway → `P_sensor`
  - Late Bayesian Fusion → `P_fused`
  - RAG Retriever → top-k relevant chunks
- Construct the **structured LLM prompt** combining:
  - Fused diagnosis probabilities
  - Top-k RAG context chunks
  - Recent conversation history
- Generate the final **advisory output** containing:
  - Primary diagnosis
  - Top-3 differential diagnoses
  - Sensor reading interpretation
  - Recommended treatment actions
- Measure **end-to-end latency over 100 trials** broken down by component, targeting **P95 < 30 seconds** (reproducing **Table IV** of the paper)

**Outputs:**
- `latency_breakdown_table4.csv`
- `latency_distribution.png`
- `sample_advisory_outputs.json`

---

## Key Libraries

| Task | Library |
|------|---------|
| CNN fine-tuning | `torch`, `torchvision` |
| YOLO ROI detection | `ultralytics` |
| FastText classifier | `fasttext` |
| XGBoost + SHAP | `xgboost`, `shap` |
| FAISS vector search | `faiss-cpu` |
| BM25 retrieval | `rank_bm25` |
| Sentence embeddings | `sentence-transformers` |
| EWC regularization | Custom PyTorch training loop |
| ROUGE-L evaluation | `rouge-score` |
| Statistical tests | `scipy.stats` (McNemar's) |
| IoT simulation | `paho-mqtt` (optional) |
| Data handling | `numpy`, `pandas` |
| Visualization | `matplotlib`, `seaborn` |

---

## Data Flow Between Notebooks

```
Notebook 1 ──► sensor_features.npy ──────────────────────► Notebook 4
               sensor_labels.npy   ──────────────────────► Notebook 4

Notebook 2 ──► P_cnn.npy ───────────────────────────────► Notebook 5
               test_labels.npy ─────────────────────────► Notebook 5

Notebook 3 ──► P_text.npy ──────────────────────────────► Notebook 5

Notebook 4 ──► P_sensor.npy ────────────────────────────► Notebook 5

Notebook 5 ──► P_fused.npy ─────────────────────────────► Notebook 7

Notebook 6 ──► faiss_index.bin ─────────────────────────► Notebook 7
               bm25_index.pkl  ─────────────────────────► Notebook 7

Notebook 7 ──► Final advisory output + latency report
```

---

> **Tip:** Run the notebooks in order (1 → 7). Each notebook saves `.npy` or model files that the next notebook expects as input.
