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

## Key Highlights

- **Domain-Driven Feature Engineering:** Extraction of financial velocity ratios, 6-month repayment delinquency trajectories, consumption volatility, and limit-utilization metrics.
- **Probability Calibration:** Post-hoc isotonic calibration (`CalibratedClassifierCV`) transforming raw tree outputs into true posterior probabilities reflecting real-world base rates.
- **Metric-Aligned Threshold Optimization:** Cost-sensitive decision cutoffs tuned on cross-validated out-of-fold (OOF) distributions via `TunedThresholdClassifierCV`.
- **Model Contenders Arena:** Comprehensive benchmarking comparing **LightGBM**, **CatBoost**, **Optimal Simplex Blending**, **Stacking Classifier (Logistic Meta-Learner)**, and **AutoGluon Tabular** (`best_quality` multi-layer stack).
- **Interpretability (SHAP):** Global feature impact, beeswarm distributions, and local waterfall risk attribution for underwriter review.
- **Production Bundle & Serving:** Serialized self-contained pipeline (`.joblib`) including reference baseline statistics for covariate drift monitoring (KS-test / Chi-Square) and an interactive **Gradio + Cloudflare** underwriting interface.

---

## Experimental Benchmark Results

Evaluated on an independent 25% stratified test holdout ($N = 7,500$ clients, 22.12% base default rate):

| Contender Architecture | Decision Cutoff | ROC-AUC | Brier Score Loss | Macro F1 | Balanced Acc | Total Business Cost | Financial Cost / Client |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **LightGBM (Calibrated + Tuned)** | **0.2670** | **0.7814** | **0.1341** | **0.7042** | **0.7103** | **$2,987** | **$0.3983** |
| 🥈 **Optimal Blend (LGB + Cat)** | 0.2625 | 0.7852 | 0.1337 | 0.7028 | 0.7124 | $2,981 | $0.3975 |
| 🥉 **AutoGluon (600s, best_quality)** | 0.2612 | 0.7865 | 0.1335 | 0.7035 | 0.7130 | $2,979 | $0.3972 |
| **Stacking Classifier (Meta-LR)** | 0.2640 | 0.7849 | 0.1338 | 0.7011 | 0.7115 | $2,984 | $0.3979 |
| **CatBoost (Calibrated + Tuned)** | 0.2590 | 0.7848 | 0.1339 | 0.6985 | 0.7118 | $2,995 | $0.3993 |

> **Key Quantitative Takeaway:** Adjusting the operational threshold from the arbitrary default ($\tau = 0.50$) to the cost-optimal cutoff ($\tau \approx 0.267$) **reduces credit default losses by over 24%**, capturing the vast majority of defaults while strictly bounding false investigations.

---

## Repository Structure

```
├── Cost_Sensitive_Credit_Risk_Modeling.ipynb  # Master runnable notebook (Colab-ready)
├── README.md                                  # Executive summary & architecture docs
├── requirements.txt                           # Core environment dependencies
├── UCI_Credit_Card.csv                        # Dataset (30k instances, 24 features)
├── domain.py                                  # Domain feature engineering logic
├── pandas_transformers.py                     # Custom leakage-safe pipeline transformers
├── polars_transformers.py                     # High-performance Polars variants
├── dashboards2.py                             # Exploratory & diagnostic visualizers
├── loaders.py                                 # Data loading and schema validation
├── missing.py                                 # Missing data diagnostic routines
└── artifacts_binary.zip                       # Pre-computed production artifacts
    └── artifacts_binary/
        ├── binary_lgb_production_bundle.joblib # Winning calibrated production bundle
        ├── lgb_oof.npy                        # LightGBM out-of-fold probabilities
        ├── cat_oof.npy                        # CatBoost out-of-fold probabilities
        ├── study_lgb.joblib                   # Optuna optimization study (LightGBM)
        ├── study_cat.joblib                   # Optuna optimization study (CatBoost)
        ├── run_meta.json                      # Reproducibility metadata & seeds
        └── requirements.txt                   # Frozen environment snapshot
```

---

## Quick Start & Execution

### 1. Run in Google Colab (1-Click Run)
Click the badge below to run the complete pipeline directly in Google Colab. The notebook will automatically sync repository assets and extract artifacts:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/franciscogamarra10/Cost-Sensitive-Credit-Risk-Modeling-and-Business-Metric-Alignment/blob/main/Cost_Sensitive_Credit_Risk_Modeling.ipynb)

### 2. Local Setup
```bash
git clone https://github.com/franciscogamarra10/Cost-Sensitive-Credit-Risk-Modeling-and-Business-Metric-Alignment.git
cd Cost-Sensitive-Credit-Risk-Modeling-and-Business-Metric-Alignment

# Install dependencies
pip install -r requirements.txt

# Extract pre-computed artifacts
unzip -q artifacts_binary.zip

# Launch Jupyter
jupyter notebook Cost_Sensitive_Credit_Risk_Modeling.ipynb
```

---

## Production Serving & Drift Monitoring

The production artifact `binary_lgb_production_bundle.joblib` packages the end-to-end transformation, feature selection, probability calibrator, and cost cutoff into a single callable interface:

```python
import joblib
import pandas as pd

# Load production bundle
bundle = joblib.load("artifacts_binary/binary_lgb_production_bundle.joblib")

# Execute raw-to-decision inference
raw_applicants = pd.read_csv("UCI_Credit_Card.csv").head(10)
predictions = predict_batch_binary(raw_applicants, bundle)

print(predictions[["LIMIT_BAL", "positive_probability", "decision"]])
```

The serving layer features automated **covariate drift detection**:
- **Continuous attributes:** Monitored via two-sample Kolmogorov-Smirnov ($KS$) tests against the baseline sample.
- **Categorical attributes:** Monitored via Chi-Square ($\chi^2$) goodness-of-fit tests against pre-computed proportion baselines.

---

## Interactive Gradio Demo

The pipeline includes a web application powered by **Gradio** and accessible publicly via a **Cloudflare Tunnel**:
1. **Single Client Underwriting:** An interactive credit evaluation form pre-filled with realistic applicant baselines (no dummy zeros) for real-time risk scoring and decision policy inspection.
2. **Batch Portfolio Audit:** Upload a client batch in CSV format to trigger automatic scoring, prediction exports, and distribution drift audits.

---

## Citation

```bibtex
@misc{credit_default_uci_2013,
  author       = {Lichman, M.},
  title        = {UCI Machine Learning Repository: Default of Credit Card Clients Data Set},
  year         = {2013},
  institution  = {University of California, Irvine, School of Information and Computer Sciences},
  url          = {https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients}
}
```

---

## License
Distributed under the **MIT License**. See `LICENSE` for more information.
