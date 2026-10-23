# LLM-DDA: Risk-conditioned delegation of LLM network diagnoses

Reproducible code and data for the Short Communication

> **Delegating diagnosis: A knowledge-based, risk-conditioned decision-support model for LLM-assisted network operations**
> Revu Surya. Submitted to *Knowledge-Based Systems* (Elsevier).

LLM-DDA decides, for each LLM-proposed network remediation, whether to execute it autonomously or escalate it to an engineer. It uses four ingredients: confidence calibrated from verified outcomes, an explicit loss model of blast radius and reversibility, engagement-forcing human review, and per-fault-class boundary recalibration from outcome feedback.

## Reproduce everything

```bash
pip install -r requirements.txt
python llmdda.py            # ~5 min on 2 cores: tables, tests, Figures 1-2 -> results/
python graphical_abstract.py
```

All data are synthetic and generated from fixed seeds:

| Seeds | Used for |
|---|---|
| 0-29 | Test streams |
| 1000-1004 | Threshold tuning |
| 500-502 | Fig. 1(b) grid |

A from-scratch run reproduces the committed `results/` files exactly.

**No LLM is called.** A stochastic surrogate reproduces two properties of real models: accuracy that is bounded by evidence coverage, and verbal confidence that is fluent but overconfident.

## Files

| File | Content |
|---|---|
| `llmdda.py` | Incident generator, policies, threshold tuning, evaluation, paired tests, figures |
| `graphical_abstract.py` | Graphical abstract from `results/table1_policies.csv` |
| `results/table1_policies.csv` | Table 1: policy comparison (mean and SD over 30 seeds) |
| `results/table2_ablation.csv` | Table 2: component ablation with Holm-adjusted paired Wilcoxon tests |
| `results/tests_*.csv` | Paired tests against baselines (cost, severe escapes) and drift-escape tests |
| `results/per_seed.csv` | Every policy × seed result |
| `results/figure*.csv`, `results/figure*.png/pdf` | Figure data and rendered figures |

## Key results (30 test seeds)

| Policy | Mean cost (CU) | Severe escapes /1000 | Autonomous |
|---|---|---|---|
| Calibrated confidence gate (selective deferral) | 17.82 | 13.6 | 59.8% |
| **LLM-DDA** | **15.66** | **5.2** | 60.1% |

Boundary recalibration does not lower average cost, but it contains an unannounced reliability drift. In the drifted fault class:

- autonomous execution falls from 59.0% to 17.0%;
- escaped errors fall from 381 to 201 per 1000.

## License

MIT
