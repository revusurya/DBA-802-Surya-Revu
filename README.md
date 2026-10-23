# LLM-DDA: Risk-conditioned delegation of LLM network diagnoses

Reproducible code and data for the Short Communication

> **Delegating diagnosis: A knowledge-based, risk-conditioned decision-support model for LLM-assisted network operations**
> Revu Surya. Prepared for submission to *Knowledge-Based Systems* (Elsevier), Short Communication.

LLM-DDA decides, for each LLM-proposed network remediation, whether to execute it autonomously or escalate it to an engineer. It uses four ingredients: confidence calibrated from verified outcomes, an explicit loss model of blast radius and reversibility, forcing-function human review, and per-fault-class boundary recalibration from outcome feedback.

## Reproduce everything

```bash
pip install -r requirements.txt
python llmdda.py            # ~10 min on 2 cores: tables, tests, sensitivity, Figures 1-2 -> results/
python graphical_abstract.py
```

All data are synthetic and generated from fixed seeds:

| Seeds | Used for |
|---|---|
| 0-29 | Test streams |
| 1000-1004 | Threshold tuning and the Fig. 1(a) frontier sweep (test seeds are never used for tuning) |
| 500-502 | Fig. 1(b) grid |

A from-scratch run reproduces the committed `results/` files exactly.

**No LLM is called.** A stochastic surrogate reproduces two properties of real models: accuracy that is bounded by evidence coverage, and verbal confidence that is fluent but overconfident.

## Files

| File | Content |
|---|---|
| `llmdda.py` | Incident generator, policies, threshold tuning, evaluation, paired tests, figures |
| `graphical_abstract.py` | Graphical abstract from `results/table1_policies.csv` |
| `results/table1_policies.csv` | Table 1: policy comparison (mean and SD over 30 seeds; `review_min` columns = review effort in CU per incident) |
| `results/table2_ablation.csv` | Table 2: component ablation with Holm-adjusted paired Wilcoxon tests |
| `results/tests_*.csv` | Paired tests against baselines (cost, severe escapes) and drift-escape tests |
| `results/per_seed.csv` | Every policy × seed result |
| `results/sensitivity.csv` | Supplementary Table S7: one-at-a-time sensitivity (loss scale, review cost, model capability) |
| `results/control_calibrated_forcing*.csv` | Supplementary Table S8: review-matched control (calibrated gate with the same forcing-function review as LLM-DDA) |
| `results/latency.csv` | Decision-layer runtime per incident |
| `results/figure*.csv`, `results/figure*.png/pdf` | Figure data and rendered figures |

## Key results (30 test seeds)

| Policy | Mean cost (CU) | Severe escaped errors /1000 | Autonomy |
|---|---|---|---|
| Calibrated-confidence gate, rubber-stamp review | 17.82 | 13.6 | 59.8% |
| Calibrated-confidence gate, forcing-function review (control) | 16.74 | 7.6 | 59.8% |
| **LLM-DDA** | **15.66** | **5.2** | 60.1% |

All baseline gates use rubber-stamp review; the control row gives the gate the same forcing-function review as LLM-DDA, so the remaining gap (6.5% cost, 31% severe escapes) comes from the risk-conditioned boundary itself.

Boundary recalibration does not lower average cost, but it contains an unannounced reliability drift. In the drifted fault class:

- autonomous execution falls from 59.0% to 17.0%;
- escaped errors fall from 381 to 201 per 1000.

Sensitivity: LLM-DDA had lower mean cost than the calibrated-confidence gate in 69 of 70 seed-condition pairs. The gain ranges from 3.3% (losses halved) to 25.0% (losses doubled), and the reduction in severe escaped errors stays between 58% and 65% in every condition. The decision layer adds about 0.04 ms per incident.

## License

MIT
