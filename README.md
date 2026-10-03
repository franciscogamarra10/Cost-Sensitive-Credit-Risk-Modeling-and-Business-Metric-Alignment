# Cost-Sensitive Credit Risk Modeling and Business Metric Alignment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end pipeline for **credit-card default prediction** that optimises a **business cost function** instead of generic classification metrics.

> False Negatives (missed defaulters) are penalised **3×** more than False Positives.  
> All model selection, calibration and threshold tuning are driven by this asymmetric cost.

**Repo:** [franciscogamarra10/Cost-Sensitive-Credit-Risk-Modeling-and-Business-Metric-Alignment](https://github.com/franciscogamarra10/Cost-Sensitive-Credit-Risk-Modeling-and-Business-Metric-Alignment)

---

## Problem Statement

The [Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) dataset contains 30 000 Taiwanese credit-card holders observed between April and September 2005.  
The binary target indicates whether the client defaulted on the next payment.

Standard accuracy / F1 optimisation is misaligned with real lending economics: failing to identify a future defaulter is far more expensive than investigating a healthy client.  
This project therefore minimises:

```
Financial Cost = FN × 3 + FP × 1
```

---

## Pipeline Overview

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1. Domain Feature Engineering | Utilisation ratios, payment ratios, delay aggregates, trends | Inject credit-risk domain knowledge |
| 2. Preprocessing | Custom transformers + target encoding | Leakage-safe, handles skew and rare labels |
| 3. Hyperparameter Search | Optuna + LightGBM / CatBoost (Brier score) | Probability quality first |
| 4. Calibration | Isotonic regression (`CalibratedClassifierCV`) | Reliable probability estimates |
| 5. Threshold Optimisation | `TunedThresholdClassifierCV` on business cost | Minimise expected financial loss |
| 6. Ensembles | Probability blending + Stacking + AutoGluon | Extra lift and strong automated baseline |
| 7. Explainability | TreeSHAP (global + local) | Model transparency |
| 8. Production Bundle | Joblib artefact + Gradio + Cloudflare demo | Ready-to-serve inference |

---

## Key Results

| Model | ROC-AUC | Brier | Financial Cost |
|-------|---------|-------|----------------|
| **LightGBM (calibrated + tuned)** | ~0.78 | ~0.14 | **lowest** |
| CatBoost | similar | similar | slightly higher |
| Blending (LGB + Cat) | competitive | competitive | very close |
| Stacking | competitive | competitive | close |
| AutoGluon (600 s, best_quality) | competitive | competitive | close to best |

LightGBM won on the business cost metric. AutoGluon, with almost no manual work, reached a result very close to the hand-crafted pipeline — a strong reminder of how powerful automated platforms have become.

---

## Repository Structure

```
├── Cost_Sensitive_Credit_Risk_Modeling.ipynb   # Full reproducible notebook (Colab-ready)
├── README.md
├── requirements.txt
└── artifacts_binary/                           # Production artefacts
    ├── binary_lgb_production_bundle.joblib     # Best model from previous run
    ├── run_meta.json
    └── model_comparison_*.csv
```

---

## Quick Start (Google Colab)

1. Open the notebook in Colab.  
2. Upload the `artifacts_binary` zip (contains the production bundle from the best run).  
3. Run the environment-setup cell.  
4. Execute all cells. The notebook loads the saved bundle; it does **not** overwrite it.  
5. Gradio + Cloudflare tunnel starts at the end for interactive testing.

Alternatively, clone and run locally:

```bash
git clone https://github.com/franciscogamarra10/Cost-Sensitive-Credit-Risk-Modeling-and-Business-Metric-Alignment.git
cd Cost-Sensitive-Credit-Risk-Modeling-and-Business-Metric-Alignment
pip install -r requirements.txt
jupyter notebook Cost_Sensitive_Credit_Risk_Modeling.ipynb
```

---

## Business Cost Definition

```python
C_FN = 3.0   # cost of a missed defaulter
C_FP = 1.0   # cost of an unnecessary investigation

def financial_cost(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn * C_FN + fp * C_FP
```

The optimal decision threshold is found by minimising this cost on out-of-fold predictions, then locked for production.

---

## Domain Feature Engineering

Raw bill and payment amounts are transformed into financially meaningful signals:

- **Credit utilisation** (`UTIL_1` … `UTIL_6`)
- **Payment-to-bill ratios** (`PAY_RATIO_1` … `PAY_RATIO_6`)
- **Bill & payment statistics** (mean, std, max, 6-month trend)
- **Delay aggregates** (max delay, average delay, count of severe delays)
- **Over-limit flag**

These engineered features consistently rank among the top SHAP contributors.

---

## Production Inference

A single joblib bundle contains:

- Feature-engineering function  
- Preprocessing pipeline  
- Calibrated + threshold-tuned LightGBM model  
- Optimal threshold  
- Reference statistics for drift monitoring  

```python
bundle = joblib.load("binary_lgb_production_bundle.joblib")
predictions = predict_batch_binary(raw_dataframe, bundle)
```

A Gradio UI (batch CSV upload + single-client form with realistic defaults) is provided for interactive testing.

---

## Citation

```
Lichman, M. (2013). UCI Machine Learning Repository.
Irvine, CA: University of California, School of Information and Computer Science.
http://archive.ics.uci.edu/ml
```

---

## License

MIT
