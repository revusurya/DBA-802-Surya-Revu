"""Graphical abstract for the LLM-DDA Short Communication (reads results/table1_policies.csv)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
t1 = pd.read_csv(os.path.join(HERE, "results", "table1_policies.csv"), index_col=0)
short = {"Human review of all (rubber-stamp)": "Review all (rubber-stamp)",
         "Full autonomy": "Full autonomy",
         "Self-reported confidence threshold": "Verbal-confidence gate",
         "Calibrated confidence threshold (selective deferral)": "Calibrated gate (rubber-stamp)",
         "LLM-DDA (full)": "LLM-DDA"}
t1 = t1.loc[list(short)]
cc = pd.read_csv(os.path.join(HERE, "results", "control_calibrated_forcing.csv")).iloc[0]
cper = pd.read_csv(os.path.join(HERE, "results", "control_calibrated_forcing_per_seed.csv"))
ctrl = pd.DataFrame({"mean_cost_mean": [cc.ctrl_mean_cost], "mean_cost_sd": [cper.ctrl_mean_cost.std()],
                     "severe_k_mean": [cc.ctrl_severe_k], "severe_k_sd": [cper.ctrl_severe_k.std()]}, index=["ctrl"])
t1 = pd.concat([t1.iloc[:4][["mean_cost_mean", "mean_cost_sd", "severe_k_mean", "severe_k_sd"]], ctrl,
                t1.iloc[4:][["mean_cost_mean", "mean_cost_sd", "severe_k_mean", "severe_k_sd"]]])
labels = list(short.values())[:4] + ["Calibrated gate (forcing fn.)", "LLM-DDA"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9})
fig = plt.figure(figsize=(13.28, 5.31), dpi=150)
ax = fig.add_axes([0.0, 0.0, 0.56, 1.0]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0.3, 9.35, "Risk-conditioned delegation of LLM network diagnoses", fontsize=13, weight="bold")

steps = [("Telemetry + evidence\ncoverage e", "#e8eef6"),
         ("LLM ranked\nhypotheses", "#e8eef6"),
         ("Calibrated\nconfidence $\\hat p$\n(verified outcomes)", "#fde9c9"),
         ("Risk $R=(1-\\hat p)\\,\\ell(b,r)$\nb: blast radius\nr: reversibility", "#fde9c9"),
         ("Delegation boundary\n$R \\leq \\tau_f$ per fault class", "#f6d2d2")]
y = 7.6
for i, (txt, c) in enumerate(steps):
    x = 0.4 + i * 1.9
    ax.add_patch(FancyBboxPatch((x, y), 1.7, 1.25, boxstyle="round,pad=0.05", fc=c, ec="#555", lw=0.8))
    ax.text(x + 0.85, y + 0.62, txt, ha="center", va="center", fontsize=7.2)
    if i < len(steps) - 1:
        ax.annotate("", (x + 1.9, y + 0.62), (x + 1.75, y + 0.62),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1))
for x, txt, c in [(5.0, "Execute autonomously", "#d9ecd9"),
                  (7.3, "Escalate: forcing-function review\n(engineer states the\ndiscriminating evidence)", "#f6d2d2")]:
    ax.add_patch(FancyBboxPatch((x, 4.9), 2.1, 1.3, boxstyle="round,pad=0.05", fc=c, ec="#555", lw=0.8))
    ax.text(x + 1.05, 5.55, txt, ha="center", va="center", fontsize=7.2)
    ax.annotate("", (x + 1.05, 6.25), (8.85, 7.55), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1))
ax.add_patch(FancyBboxPatch((0.4, 2.6), 9.0, 1.4, boxstyle="round,pad=0.05", fc="#f4f4f4", ec="#999", lw=0.8, ls="--"))
ax.text(4.9, 3.3, "Outcome feedback: recalibrate $\\hat p$ (L5) and contract or expand $\\tau_f$ per fault class (L6)\n"
        "drifted class: autonomous share 59% → 17%, escaped errors 381 → 201 per 1000",
        ha="center", va="center", fontsize=7.6)
ax.text(0.4, 1.4, "Simulation benchmark: 6,000-incident streams x 30 seeds, 6 fault classes,\n"
        "overconfident LLM surrogate (verbal confidence 0.95 vs accuracy 0.64), mid-stream drift.",
        fontsize=7.4, color="#333")

cols = ["#9a9a9a"] * 4 + ["#6f6f6f", "#b2182b"]
for k, (m, lab) in enumerate([("mean_cost", "Mean cost per incident (CU)"),
                              ("severe_k", "Severe escaped errors per 1000")]):
    a = fig.add_axes([0.665 + k * 0.18, 0.2, 0.13, 0.62])
    a.barh(range(6), t1[m + "_mean"], xerr=t1[m + "_sd"], color=cols, height=0.65, error_kw=dict(lw=0.7))
    a.set_yticks(range(6)); a.set_yticklabels(labels if k == 0 else [], fontsize=8)
    a.invert_yaxis(); a.set_title(lab, fontsize=8.5, weight="bold")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for i, v in enumerate(t1[m + "_mean"]):
        a.text(v + t1[m + "_sd"].iloc[i] + 0.6, i, f"{v:.1f}", va="center", fontsize=7.5)
fig.text(0.58, 0.05, "LLM-DDA vs calibrated gate: 12.1% lower cost and 61% fewer severe escapes (rubber-stamp review);\n6.5% and 31% when the gate uses the same forcing-function review (30 seeds; all paired p < 1e-5)",
         fontsize=7.6)
fig.savefig(os.path.join(HERE, "results", "graphical_abstract.png"), dpi=300)
fig.savefig(os.path.join(HERE, "results", "graphical_abstract.pdf"))
