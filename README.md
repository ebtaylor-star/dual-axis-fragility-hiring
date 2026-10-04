# Dual-Axis Fragility in Algorithmic Hiring

**An Empirical Audit of Structural Exclusion and Adversarial Vulnerabilities in Automated Recruitment Pipelines**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-DPR%20Audit-orange)](https://fairlearn.org)
[![AIF360](https://img.shields.io/badge/AIF360-Reweighing-green)](https://aif360.mybluemix.net/)
[![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-red)](https://shap.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-FairCVdb%20N%3D24K-purple)](https://github.com/BiDAlab/FairCVdb)

> *"A tool that is fair but easy to trick, or hard to trick but discriminatory, is still a legal liability."*

---

## Overview

Most third-party audits of automated hiring tools examine only one dimension of risk: whether the system discriminates against protected groups. An equally consequential risk goes largely unexamined: **whether the same system can be gamed by candidates who manipulate their inputs.**

This research develops and validates a **dual-axis auditing framework** that measures both risks against a single dataset and a single trained model, so the findings are directly comparable — not artifacts of different data or different baselines.

**Research Question:** *To what extent can a dual-axis auditing framework measure both algorithmic fairness (structural exclusion) and system gameability (adversarial exploitability)?*

---

## The Two Axes

### Axis 1 — Structural Exclusion (Fairness)

Does the automated hiring tool discriminate against protected demographic groups?

**Method:** Fairlearn's `MetricFrame`, computing the **Demographic Parity Ratio (DPR)** — the selection rate of the least-favoured group divided by the most-favoured group's rate — benchmarked against the **EEOC four-fifths rule** (≥ 0.80 threshold).

### Axis 2 — Adversarial Exploitability (Gaming)

Can a rejected candidate reverse their outcome by making small, strategic edits to mutable qualifications — while protected attributes remain constant?

**Method:** Custom flip-threshold perturbation search over mutable features only, with a 2-step edit budget. Results expressed as the **Model Fragility Index (MFI)**: the share of rejected candidates whose decision flips within that budget.

---

## Key Findings

### Five-KPI Summary

| Metric | Run 1 (Biased Label) | Run 3 (Reweighed) |
|---|---|---|
| **Validation Accuracy** | 89.6% | 88.3% |
| **ROC-AUC** | 0.968 | — |
| **DPR (Axis 1)** | 0.329 ❌ FAILS | 0.813 ✅ PASSES |
| **MFI (Axis 2, single-step)** | 83.5% | 88.4% |
| **Gaming Share** | 9.3% of flips | 2.6% of flips |

*EEOC threshold: DPR ≥ 0.80 · HIGH RISK threshold: DPR < 0.70 OR MFI > 60%*

### The Core Finding

> **Both axes were re-tested on the same model after fairness mitigation. DPR rose (good). MFI also rose (bad). Fairness and fragility do not move together.**

Fixing structural exclusion via AIF360 Reweighing raised the DPR from 0.329 to 0.813, passing the EEOC threshold at a cost of only 1.3 percentage points in accuracy. But the same mitigation raised the Model Fragility Index from 83.5% to 88.4% — the fairer model is slightly *more* gameable, not less.

### Axis 1 Detail — Selection Rates by Group (Run 1)

| Group | Selection Rate |
|---|---|
| Men | 35.7% |
| Women | 11.8% |
| **DPR** | **0.329** |

Threshold: ≥ 0.80. Result: **FAILS** — confirms the biased label signal was learned by the classifier.

### Axis 2 Detail — Gaming Rate by Feature (Run 1)

| Mutable Feature | Gaming Rate |
|---|---|
| `recommendation` | 90.0% |
| `prev_experience` | 81.0% |
| `lang_1`, `lang_2`, `lang_3` | — |
| `availability` | — |

*Gaming = flip achieved without genuine qualification improvement (ground-truth check against unbiased label)*

Overall: **9.3% of flips are gaming** (83.5% MFI total, 2-step budget). 90.7% of flips reflect candidates who could genuinely qualify — which is the intended use case. The system is still structurally fragile.

### Feature Importance (Run 1, Random Forest)

| Feature | Importance |
|---|---|
| `recommendation` | 0.254 |
| `suitability` | 0.181 |
| `gender` (protected) | 0.134 |
| `lang_avg` | 0.068 |

`gender` ranking third in feature importance is direct evidence the classifier learned the injected biased signal — the mechanism behind the DPR = 0.329 result.

---

## Procurement Risk Assessment

```
┌─────────────────────────────────────────────────────────┐
│             DUAL-AXIS RISK QUADRANT                     │
│                                                         │
│  HIGH │  Fair but       │  HIGH RISK                   │
│  MFI  │  Gameable       │  Both Axes                   │
│       │─────────────────│─────────────────             │
│  LOW  │  LOW RISK       │  Fair &                      │
│  MFI  │  Both Axes      │  Robust                      │
│       └─────────────────┘                              │
│            LOW DPR           HIGH DPR                   │
└─────────────────────────────────────────────────────────┘
```

**Verdict: HIGH RISK — Both Axes**

| Threshold | Run 1 | Result |
|---|---|---|
| DPR ≥ 0.70 | 0.329 | ❌ FAIL |
| MFI ≤ 60% | 83.5% | ❌ FAIL |

**Procurement recommendation: Do not procure** until both (a) fairness mitigation is implemented and independently validated, and (b) adversarial exploitability is reduced via mutable feature constraints, scoring transparency, or anomaly detection on submission patterns.

---

## Regulatory Context

| Framework | Requirement | Relevance |
|---|---|---|
| **Mobley v. Workday, Inc. (2024)** | Vendor liability for ATS discrimination; certified as nationwide class action (age, race, disability, sex) | Axis 1 — structural exclusion |
| **NYC Local Law 144 (2023)** | Mandatory bias audit + public disclosure for automated employment decision tools | Axis 1 — DPR audit |
| **EU AI Act (2024)** | High-risk classification for AI in hiring; requires conformity assessment and human oversight | Both axes |

A tool with DPR = 0.329 would face actionable exposure under all three frameworks.

---

## Dataset

**FairCVdb** — BiDA Lab, Universidad Autónoma de Madrid (Peña et al., 2023)

| Property | Value |
|---|---|
| Records | 24,000 synthetic candidate profiles |
| Features | Competency scores, demographics, language proficiency, experience, availability |
| Target variants | Blind (unbiased), Gender-Biased, Ethnicity-Biased |
| Demographic anchoring | U.S. Census Bureau education attainment distributions |
| Face identity | DiveFace database (Morales et al., 2021) |

This research used the **gender-biased label** (Run 1), the **blind label** (Run 2, baseline), and applied **AIF360 Reweighing** to the gender-biased training set (Run 3).

**Mutable features** (candidates can change): `prev_experience`, `lang_1`, `lang_2`, `lang_3`, `availability`, `recommendation`  
**Immutable / protected features** (held constant in Axis 2): `gender`, `ethnicity`

---

## Methodology

### Model

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
```

Train/test split: 80/20 stratified by `gender` + `ethnicity`

### Axis 1 — Fairlearn DPR Audit

```python
from fairlearn.metrics import MetricFrame, demographic_parity_ratio
import functools

metric_frame = MetricFrame(
    metrics={"selection_rate": functools.partial(selection_rate)},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

dpr = demographic_parity_ratio(y_true=y_test, y_pred=y_pred,
                                sensitive_features=sensitive_features)
```

### Axis 2 — Custom Flip-Threshold Search

```python
def flip_threshold_search(model, X_rejected, mutable_cols, budget=2):
    """
    For each rejected candidate, attempt to flip the decision
    by modifying up to `budget` mutable features.
    Returns: flipped_count, flip_indices
    """
    flipped = []
    for idx in range(len(X_rejected)):
        candidate = X_rejected[idx].copy()
        for col in mutable_cols:
            candidate_mod = candidate.copy()
            candidate_mod[col] = candidate_mod[col] * 1.1  # +10% perturbation
            if model.predict([candidate_mod])[0] == 1:
                flipped.append(idx)
                break
    return len(flipped), flipped
```

*MFI = flipped_count / len(X_rejected)*

### Axis 3 — Gaming vs. Improvement Ground-Truth Check

```python
# A flip is "gaming" if the unbiased (blind) label still rejects the candidate
# A flip is "improvement" if the unbiased label would have accepted them

gaming_rate = (
    flipped_still_rejected_by_blind_label / total_flipped
)
```

### Run 3 — AIF360 Reweighing Mitigation

```python
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

rw = Reweighing(unprivileged_groups=[{'gender': 0}],
                privileged_groups=[{'gender': 1}])
dataset_transf = rw.fit_transform(dataset_orig)
```

### SHAP Explainability

```python
import shap

def compute_shap_summary(model, X_train, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[1], X_test,
                      feature_names=feature_names, show=False)
```

---

## Three-Run Structure

| Run | Label | Purpose |
|---|---|---|
| **Run 1** | Gender-Biased | Primary diagnostic — confirms Axis 1 failure, establishes MFI baseline |
| **Run 2** | Blind (Unbiased) | Benchmark — confirms bias is in the label, not the features |
| **Run 3** | Reweighed (AIF360) | Mitigation — tests whether Axis 1 fix changes Axis 2 exposure |

---

## Repository Structure

```
dual-axis-fragility-hiring/
├── notebooks/
│   └── dual_axis_fragility_in_algorithmic_hiring_final.py   # Full Colab export, 3-run structure
├── reports/
│   ├── Utica_Capstone_Elizabeth_Taylor_2026_FINAL.pdf        # 60-page final paper
│   └── Dual-Axis_Fragility_One-Page_Findings.html            # Interactive findings summary
├── requirements.txt
└── README.md
```

---

## Requirements

```
scikit-learn>=1.3
fairlearn>=0.9
aif360>=0.5
shap>=0.43
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
```

Install: `pip install -r requirements.txt`

---

## References

- Peña, A., et al. (2023). *FairCVdb: A Database for Studying Fairness in Automated CV Analysis.* BiDA Lab, Universidad Autónoma de Madrid.
- Mobley v. Workday, Inc. (2024). N.D. Cal. No. 3:23-cv-00770.
- New York City Local Law 144 (2023). Automated Employment Decision Tools.
- European Parliament. (2024). *EU Artificial Intelligence Act.* Regulation (EU) 2024/1689.
- Morales, A., et al. (2021). *SensitiveNets: Learning Agnostic Representations with Application to Face Images.* IEEE TPAMI.
- An, R., et al. (2025). *Hiring Bias in Large Language Models.* PNAS Nexus.

---

## Author

**Dr. Elizabeth B. Taylor, DBA**  
M.S. Data Science, Utica University (August 2026)  
Advisor: Dr. Michael McCarthy  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-lizbtaylor-blue?style=flat&logo=linkedin)](https://linkedin.com/in/lizbtaylor/)  
[![GitHub](https://img.shields.io/badge/GitHub-ebtaylor--star-black?style=flat&logo=github)](https://github.com/ebtaylor-star)

*MSDS Capstone Project · Utica University · August 2026*

---

## Citation

```bibtex
@mastersthesis{taylor2026dualaxis,
  author    = {Taylor, Elizabeth B.},
  title     = {Dual-Axis Fragility in Algorithmic Hiring: An Empirical Audit of
               Structural Exclusion and Adversarial Vulnerabilities in Automated
               Recruitment Pipelines},
  school    = {Utica University},
  year      = {2026},
  month     = {August},
  advisor   = {McCarthy, Michael}
}
```
