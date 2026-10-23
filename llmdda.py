"""
LLM-DDA: risk-conditioned delegation of LLM-generated network diagnoses.

Single-file, reproducible simulation benchmark for the Short Communication
"Delegating diagnosis: A knowledge-based, risk-conditioned decision-support model
for LLM-assisted network operations" (Knowledge-Based Systems submission).

The LLM is represented by a stochastic surrogate (no LLM is called): its
diagnostic accuracy is bounded by evidence coverage, and its self-reported
confidence is high and only weakly informative. Every reported number is
produced by `python llmdda.py`, which writes CSV files and figures to ./results.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# --------------------------------------------------------------------------
# 1. Incident generator (abstracts layers L1-L4 of LLM-DDA)
# --------------------------------------------------------------------------
CLASSES = ["Interface/physical", "OSPF adjacency", "BGP policy",
           "MTU/fragmentation", "ACL/firewall", "Config drift"]
FREQ = np.array([0.25, 0.15, 0.12, 0.10, 0.18, 0.20])
COV = [(5, 2), (4, 2.5), (3, 3), (2.5, 3), (4, 2), (5, 1.8)]      # evidence coverage e
BLAST = [(1.5, 6), (2, 4), (4, 2), (2, 5), (2.5, 3), (2, 3)]       # blast radius b
REV = [(6, 1.5), (5, 2), (2, 3), (5, 2), (4, 2), (7, 1.5)]         # reversibility r
APT = np.array([1.00, 0.95, 0.85, 0.80, 0.95, 1.00])               # class aptitude

H = 4                    # competing root-cause hypotheses per incident
P_COMPROMISED = 0.06     # incidents whose telemetry integrity is degraded
P_FLAG = 0.80            # integrity monitor detection probability
DRIFT_APT = 0.55         # aptitude of the drifted class after an unseen vendor release


def loss_if_wrong(b, r):
    """Organizational loss model l(b, r) for executing a wrong remediation (CU)."""
    return 15.0 + 185.0 * b * (1.0 - 0.7 * r)


def make_incidents(seed, n_hist=2000, n_test=6000, q=0.90, drift_at=3000, loss_scale=1.0):
    """History block (for calibration) followed by a test block with drift."""
    rng = np.random.default_rng(seed)
    drift_cls = int(rng.integers(len(CLASSES)))       # drifted class drawn per seed
    n = n_hist + n_test
    cls = rng.choice(len(CLASSES), size=n, p=FREQ)
    a = np.array([COV[c] for c in cls]); e = rng.beta(a[:, 0], a[:, 1])
    a = np.array([BLAST[c] for c in cls]); b = rng.beta(a[:, 0], a[:, 1])
    a = np.array([REV[c] for c in cls]); r = rng.beta(a[:, 0], a[:, 1])
    comp = rng.random(n) < P_COMPROMISED
    flag = comp & (rng.random(n) < P_FLAG)
    apt = APT[cls].copy()
    t = np.arange(n) - n_hist
    apt[(t >= drift_at) & (cls == drift_cls)] = DRIFT_APT
    e_eff = np.where(comp, 0.4 * e, e)
    p_true = 1.0 / H + (q * apt - 1.0 / H) * e_eff ** 0.8           # grounding bound
    correct = rng.random(n) < p_true
    s = np.clip(0.80 + 0.15 * q + 0.04 * correct + rng.normal(0, 0.06, n), 0, 0.999)
    u = rng.random((n, 3))                                          # human-review draws
    return dict(cls=cls, e=e, b=b, r=r, flag=flag.astype(float), s=s,
                correct=correct, p_true=p_true, L=loss_scale * loss_if_wrong(b, r),
                u=u, n_hist=n_hist, comp=comp, drift_cls=drift_cls)


def features(inc, idx, use_e=True, use_flag=True):
    oh = np.eye(len(CLASSES))[inc["cls"][idx]]
    cols = [oh, inc["s"][idx, None]]
    if use_e:
        cols += [inc["e"][idx, None], inc["e"][idx, None] ** 2]
    if use_flag:
        cols.append(inc["flag"][idx, None])
    return np.hstack(cols)


def fit_calibrator(inc, idx, use_e=True, use_flag=True):
    """Layer L5: calibrated confidence from verified outcomes (Eq. 2), not verbal self-report."""
    m = LogisticRegression(C=1.0, max_iter=500)
    m.fit(features(inc, idx, use_e, use_flag), inc["correct"][idx])
    return m


# --------------------------------------------------------------------------
# 2. Policies (layers L5-L7; Eqs. 3-6)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Policy:
    name: str
    gate: str = "risk"          # 'none_auto' | 'none_human' | 'self' | 'cal' | 'risk'
    theta: float = 0.0          # threshold (confidence for self/cal, risk for risk)
    review: str = "rs"          # 'rs' rubber-stamp | 'ff' forcing function
    use_e: bool = True          # evidence coverage in the calibrator
    use_flag: bool = True       # integrity signal in the calibrator
    consequence: bool = True    # condition on blast radius and reversibility
    recal_L5: bool = False      # periodic recalibration of confidence
    recal_L6: bool = False      # per-class boundary recalibration
    z: float = 3.0              # alarm threshold (standard errors) for L6


REVIEW = {  # d = d0 * exp(-phi * trust) * fatigue ; cost v; false-override prob
    "rs": dict(d0=0.95, phi=1.6, v=6.0, fo=0.03),
    "ff": dict(d0=0.92, phi=0.35, v=12.0, fo=0.02),
}
C_AUTO_OK, C_MANUAL, C_OVERRIDE = 1.0, 10.0, 10.0
MEAN_L = float(loss_if_wrong(0.30, 0.68))


def run(inc, pol: Policy, phi_override=None, v_ff=None):
    nh = inc["n_hist"]; n = len(inc["cls"]); test = np.arange(nh, n)
    rv = dict(REVIEW[pol.review])
    if phi_override is not None:
        rv["phi"] = phi_override
    if v_ff is not None and pol.review == "ff":
        rv["v"] = v_ff
    cal = None
    if pol.gate in ("cal", "risk"):
        cal = fit_calibrator(inc, np.arange(nh), pol.use_e, pol.use_flag)
        phat = cal.predict_proba(features(inc, test, pol.use_e, pol.use_flag))[:, 1]
    tau = np.full(len(CLASSES), pol.theta)
    win = [[] for _ in CLASSES]          # (wrong, predicted error) of autonomous acts
    trust, load = 0.75, 0.5
    cls, s, L, cor, u, b, r = (inc[k] for k in ("cls", "s", "L", "correct", "u", "b", "r"))
    cost = np.zeros(len(test)); auto = np.zeros(len(test), bool)
    escaped = np.zeros(len(test), bool)
    for j, i in enumerate(test):
        if pol.recal_L5 and j > 0 and j % 500 == 0:
            idx = np.arange(max(0, i - 3000), i)
            cal = fit_calibrator(inc, idx, pol.use_e, pol.use_flag)
            rest = test[j:]
            phat[j:] = cal.predict_proba(features(inc, rest, pol.use_e, pol.use_flag))[:, 1]
        c = cls[i]
        if pol.gate == "none_auto":
            go = True
        elif pol.gate == "none_human":
            go = False
        elif pol.gate == "self":
            go = s[i] >= pol.theta
        elif pol.gate == "cal":
            go = phat[j] >= pol.theta
        else:
            lhat = L[i] if pol.consequence else MEAN_L
            go = (1.0 - phat[j]) * lhat <= tau[c]
        if go:
            auto[j] = True
            if cor[i]:
                cost[j] = C_AUTO_OK
            else:
                cost[j] = L[i]; escaped[j] = True
            if pol.recal_L6:
                w = win[c]; w.append((0.0 if cor[i] else 1.0, 1.0 - phat[j]))
                if len(w) > 100:
                    w.pop(0)
                k = len(w)
                if k >= 30:
                    o = sum(x[0] for x in w) / k; m = sum(x[1] for x in w) / k
                    if o > m + pol.z * math.sqrt(max(m * (1 - m), 1e-4) / k):
                        tau[c] *= 0.5; win[c] = []
                    elif k >= 100 and o <= m:
                        tau[c] = min(pol.theta, tau[c] * 1.1)
        else:
            fatigue = 1.0 - 0.5 * max(0.0, load - 0.5)
            d = rv["d0"] * math.exp(-rv["phi"] * trust) * fatigue
            cst = rv["v"]
            if cor[i]:
                if u[i, 0] < rv["fo"]:
                    cst += C_OVERRIDE
            elif u[i, 1] < d:
                cst += C_MANUAL
            else:
                cst += L[i]; escaped[j] = True
            cost[j] = cst
            trust += 0.02 * ((1.0 if cor[i] else 0.0) - trust)
        load += 0.01 * ((0.0 if go else 1.0) - load)
    bt, rt = b[test], r[test]
    severe = escaped & (bt >= 0.6) & (rt <= 0.4)
    q95 = np.quantile(cost, 0.95)
    m = (np.arange(len(test)) >= 3000) & (cls[test] == inc["drift_cls"])   # post-drift, drifted class
    return dict(drift_esc_k=1000 * escaped[m].mean(), drift_auto=auto[m].mean(),cost=cost, auto=auto, escaped=escaped, cls=cls[test], drift_cls=inc["drift_cls"],
                mean_cost=cost.mean(), automation=auto.mean(),
                escapes_k=1000 * escaped.mean(), severe_k=1000 * severe.mean(),
                cvar95=cost[cost >= q95].mean(),
                review_min=rv["v"] * (1 - auto.mean()))


# --------------------------------------------------------------------------
# 3. Threshold tuning on validation seeds (no test-seed information)
# --------------------------------------------------------------------------
VAL_SEEDS = range(1000, 1005)
TEST_SEEDS = range(30)
GRID = {"self": np.round(np.linspace(0.90, 0.999, 23), 4),
        "cal": np.round(np.linspace(0.50, 0.99, 25), 3),
        "risk": np.round(np.geomspace(0.5, 60, 25), 3)}
_CACHE = {}


def incidents(seed, **kw):
    key = (seed, tuple(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = make_incidents(seed, **kw)
    return _CACHE[key]


def tune(pol: Policy):
    if pol.gate not in GRID:
        return pol
    best = None
    zs = (2.0, 3.0) if pol.recal_L6 else (pol.z,)
    for th, z in [(t, z) for t in GRID[pol.gate] for z in zs]:
        p = replace(pol, theta=float(th), z=z)
        m = np.mean([run(incidents(sd), p)["mean_cost"] for sd in VAL_SEEDS])
        if best is None or m < best[0]:
            best = (m, p)
    return best[1]


BASELINES = [
    Policy("Human review of all (rubber-stamp)", gate="none_human", review="rs"),
    Policy("Human review of all (forcing function)", gate="none_human", review="ff"),
    Policy("Full autonomy", gate="none_auto"),
    Policy("Self-reported confidence threshold", gate="self"),
    Policy("Calibrated confidence threshold (selective deferral)", gate="cal"),
]
FULL = Policy("LLM-DDA (full)", gate="risk", review="ff", recal_L5=True, recal_L6=True)
ABLATIONS = [
    FULL,
    replace(FULL, name="- evidence coverage in calibration", use_e=False),
    replace(FULL, name="- blast radius / reversibility", consequence=False),
    replace(FULL, name="- forcing function (rubber-stamp review)", review="rs"),
    replace(FULL, name="- integrity signal", use_flag=False),
    replace(FULL, name="- confidence recalibration (L5)", recal_L5=False),
    replace(FULL, name="- boundary recalibration (L6)", recal_L6=False),
    replace(FULL, name="- both feedback loops (static)", recal_L5=False, recal_L6=False),
]

METRICS = ["mean_cost", "automation", "escapes_k", "severe_k", "cvar95", "review_min",
           "drift_esc_k", "drift_auto"]


def evaluate(pols):
    rows = []
    for p in pols:
        for sd in TEST_SEEDS:
            res = run(incidents(sd), p)
            rows.append({"policy": p.name, "seed": sd, "theta": p.theta, "z": p.z,
                         **{k: res[k] for k in METRICS}})
    return pd.DataFrame(rows)


def summarize(df, order):
    g = df.groupby("policy")[METRICS]
    mu, sd = g.mean(), g.std()
    out = mu.add_suffix("_mean").join(sd.add_suffix("_sd"))
    out["theta"] = df.groupby("policy")["theta"].first()
    out["z"] = df.groupby("policy")["z"].first()
    return out.loc[order]


def paired(df, ref, others, metric="mean_cost"):
    a = df[df.policy == ref].sort_values("seed")[metric].to_numpy()
    rows = []
    for o in others:
        x = df[df.policy == o].sort_values("seed")[metric].to_numpy()
        w = stats.wilcoxon(x, a)
        d = x - a
        rows.append({"comparison": o, "delta_mean": d.mean(),
                     "delta_pct": 100 * d.mean() / a.mean(),
                     "cohen_dz": d.mean() / d.std(ddof=1), "p_raw": w.pvalue})
    out = pd.DataFrame(rows)
    # Holm-Bonferroni
    order = np.argsort(out.p_raw.to_numpy()); m = len(out); adj = np.empty(m); run_max = 0
    for rank, k in enumerate(order):
        run_max = max(run_max, (m - rank) * out.p_raw.iloc[k]); adj[k] = min(1.0, run_max)
    out["p_holm"] = adj
    return out


# --------------------------------------------------------------------------
# 4. Figure experiments
# --------------------------------------------------------------------------
def frontier(seeds=VAL_SEEDS):
    """Fig. 1(a): threshold sweep on the validation seeds (never on test seeds)."""
    curves = {"Self-reported confidence": Policy("s", gate="self"),
              "Calibrated confidence": Policy("c", gate="cal"),
              "Risk-conditioned (static, rubber-stamp)": Policy("r", gate="risk"),
              "LLM-DDA (full)": FULL}
    rows = []
    for lab, p in curves.items():
        for th in GRID[p.gate]:
            q = replace(p, theta=float(th))
            rs = [run(incidents(sd), q) for sd in seeds]
            rows.append({"curve": lab, "theta": th,
                         "automation": np.mean([x["automation"] for x in rs]),
                         "mean_cost": np.mean([x["mean_cost"] for x in rs]),
                         "severe_k": np.mean([x["severe_k"] for x in rs])})
    return pd.DataFrame(rows)


def paradox(qs=np.round(np.linspace(0.60, 0.98, 13), 3),
            phis=np.round(np.linspace(0.0, 3.0, 13), 2), seeds=range(3)):
    """Escaped errors per 1000 when every diagnosis is human-reviewed (rubber-stamp)."""
    pol = Policy("h", gate="none_human", review="rs")
    rows = []
    for q in qs:
        incs = [make_incidents(sd + 500, n_hist=10, n_test=3000, q=float(q), drift_at=10**9)
                for sd in seeds]
        for ph in phis:
            v = np.mean([run(x, pol, phi_override=float(ph))["escapes_k"] for x in incs])
            acc = np.mean([x["correct"][x["n_hist"]:].mean() for x in incs])
            rows.append({"q": q, "phi": ph, "accuracy": acc, "escapes_k": v})
    return pd.DataFrame(rows)


def timeseries(pols, seeds=TEST_SEEDS, win=300):
    rows = {}
    for p in pols:
        cs, bg = [], []
        for sd in seeds:
            res = run(incidents(sd), p)
            cs.append(pd.Series(res["cost"]).rolling(win, min_periods=win).mean().to_numpy())
            a = pd.Series(np.where(res["cls"] == res["drift_cls"], res["auto"], np.nan))
            v = a.rolling(win * 2, min_periods=1).mean().to_numpy().copy(); v[:win * 2] = np.nan
            bg.append(v)
        with np.errstate(all="ignore"):
            rows[p.name] = (np.nanmean(cs, 0), np.nanmean(bg, 0))
    return rows


# --------------------------------------------------------------------------
# 5. Figures
# --------------------------------------------------------------------------
def figures(fr, px, ts, drift_at=3000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 9.5, "axes.titleweight": "bold",
                         "legend.frameon": False, "savefig.dpi": 600})
    col = {"Self-reported confidence": "#9a9a9a", "Calibrated confidence": "#4c78a8",
           "Risk-conditioned (static, rubber-stamp)": "#e8a33d", "LLM-DDA (full)": "#b2182b"}
    flab = {"Self-reported confidence": "Verbal-confidence gate",
            "Calibrated confidence": "Calibrated-confidence gate",
            "Risk-conditioned (static, rubber-stamp)": "Risk rule, rubber-stamp review, no feedback",
            "LLM-DDA (full)": "LLM-DDA (full)"}

    # ---- Figure 1 -------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    for lab, g in fr.groupby("curve", sort=False):
        g = g.sort_values("automation")
        ax[0].plot(100 * g.automation, g.mean_cost, "-o", ms=2.6, lw=1.4,
                   color=col[lab], label=flab[lab])
        k = g.mean_cost.idxmin()
        ax[0].plot(100 * g.automation[k], g.mean_cost[k], "*", ms=10,
                   color=col[lab], mec="k", mew=0.4)
    ax[0].set_xlabel("Autonomy: diagnoses executed without review (%)")
    ax[0].set_ylabel("Mean operational cost per incident (CU)")
    ax[0].set_title("(a) Cost–autonomy frontier")
    ax[0].legend(fontsize=7, loc="upper right")
    piv = px.pivot(index="phi", columns="q", values="escapes_k")
    accs = px.groupby("q").accuracy.mean().reindex(piv.columns)
    im = ax[1].imshow(piv.values, origin="lower", aspect="auto", cmap="magma_r",
                      extent=[accs.iloc[0], accs.iloc[-1], piv.index[0], piv.index[-1]])
    ax[1].axhline(REVIEW["rs"]["phi"], color="w", lw=0.6, ls=":")
    ax[1].axhline(REVIEW["ff"]["phi"], color="w", lw=0.6, ls=":")
    ax[1].text(accs.iloc[0] + 0.005, REVIEW["rs"]["phi"] + 0.05, "rubber-stamp", color="w", fontsize=6.5)
    ax[1].text(accs.iloc[0] + 0.005, REVIEW["ff"]["phi"] + 0.05, "forcing function", color="w", fontsize=6.5)
    ax[1].set_xlabel("Mean diagnostic accuracy of the LLM")
    ax[1].set_ylabel("Complacency sensitivity $\\varphi$")
    ax[1].set_title("(b) Escaped errors under full human review")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.05, pad=0.02)
    cb.set_label("Escaped errors per 1000", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure1.png")); fig.savefig(os.path.join(OUT, "figure1.pdf"))
    plt.close(fig)

    # ---- Figure 2 -------------------------------------------------------
    tcol = {"Calibrated confidence threshold (selective deferral)": "#4c78a8",
            "- both feedback loops (static)": "#e8a33d", "LLM-DDA (full)": "#b2182b"}
    tlab = {"Calibrated confidence threshold (selective deferral)": "Calibrated-confidence gate",
            "- both feedback loops (static)": "LLM-DDA without feedback loops",
            "LLM-DDA (full)": "LLM-DDA (full)"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for k, (c, a) in ts.items():
        x = np.arange(len(c))
        ax[0].plot(x, c, color=tcol[k], lw=1.3, label=tlab[k])
        ax[1].plot(x, 100 * a, color=tcol[k], lw=1.3, label=tlab[k])
    for a_ in ax:
        a_.axvline(drift_at, color="k", lw=0.8, ls="--")
        a_.set_xlabel("Incident index (test stream)")
    ax[0].text(drift_at + 80, ax[0].get_ylim()[0] + 0.15, "unseen vendor release:\none fault class drifts", fontsize=7, va="bottom")
    ax[0].set_ylabel("Rolling mean cost (CU, 300 incidents)")
    ax[0].set_title("(a) Operational cost under drift")
    ax[1].set_ylabel("Drifted-class autonomy (%)")
    ax[1].set_title("(b) Delegation boundary, drifted fault class")
    ax[1].legend(fontsize=7, loc="lower left", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    ax[1].set_ylim(0, 75)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure2.png")); fig.savefig(os.path.join(OUT, "figure2.pdf"))
    plt.close(fig)


def sensitivity(test_seeds=range(10)):
    """One-at-a-time sensitivity of the main comparison to the fixed design coefficients.
    Thresholds are re-tuned on the validation seeds for every condition."""
    conds = [("baseline", {}, None), ("loss scale x0.5", {"loss_scale": 0.5}, None),
             ("loss scale x2", {"loss_scale": 2.0}, None), ("forcing review 9 CU", {}, 9.0),
             ("forcing review 18 CU", {}, 18.0), ("capability q=0.80", {"q": 0.80}, None),
             ("capability q=0.95", {"q": 0.95}, None)]
    pols = {"cal": Policy("cal", gate="cal"), "llmdda": FULL}
    rows = []
    for name, kw, vff in conds:
        res = {}
        for k, p in pols.items():
            best = None
            for th in GRID[p.gate]:
                q_ = replace(p, theta=float(th))
                m = np.mean([run(incidents(sd, **kw), q_, v_ff=vff)["mean_cost"] for sd in VAL_SEEDS])
                if best is None or m < best[0]:
                    best = (m, q_)
            res[k] = [run(incidents(sd, **kw), best[1], v_ff=vff) for sd in test_seeds]
            res[k + "_theta"] = best[1].theta
        a = np.array([x["mean_cost"] for x in res["cal"]]); b = np.array([x["mean_cost"] for x in res["llmdda"]])
        sa = np.array([x["severe_k"] for x in res["cal"]]); sb = np.array([x["severe_k"] for x in res["llmdda"]])
        rows.append({"condition": name, "cal_cost": a.mean(), "llmdda_cost": b.mean(),
                     "cost_reduction_pct": 100 * (1 - b.mean() / a.mean()), "p_cost": stats.wilcoxon(a, b).pvalue,
                     "cal_severe_k": sa.mean(), "llmdda_severe_k": sb.mean(),
                     "severe_reduction_pct": 100 * (1 - sb.mean() / sa.mean()),
                     "cal_auto": np.mean([x["automation"] for x in res["cal"]]),
                     "llmdda_auto": np.mean([x["automation"] for x in res["llmdda"]]),
                     "wins_cost": int((b < a).sum()), "n_seeds": len(a),
                     "cal_theta": res["cal_theta"], "llmdda_tau0": res["llmdda_theta"]})
    return pd.DataFrame(rows)


def control_calibrated_forcing():
    """Control analysis: calibrated-confidence gate escalating to forcing-function review
    (same review design as LLM-DDA), tuned on validation seeds, compared on the 30 test seeds."""
    ctrl = tune(Policy("Calibrated gate + forcing-function review", gate="cal", review="ff"))
    full = tune(FULL)
    rows = []
    for sd in TEST_SEEDS:
        a, b = run(incidents(sd), ctrl), run(incidents(sd), full)
        rows.append({"seed": sd, **{f"ctrl_{k}": a[k] for k in METRICS}, **{f"full_{k}": b[k] for k in METRICS}})
    df = pd.DataFrame(rows)
    out = {"ctrl_theta": ctrl.theta}
    for k in ("mean_cost", "automation", "severe_k", "escapes_k", "cvar95", "review_min", "drift_esc_k"):
        d = df[f"ctrl_{k}"] - df[f"full_{k}"]
        out[f"ctrl_{k}"] = df[f"ctrl_{k}"].mean(); out[f"full_{k}"] = df[f"full_{k}"].mean()
        out[f"{k}_p"] = stats.wilcoxon(df[f"ctrl_{k}"], df[f"full_{k}"]).pvalue
        out[f"{k}_wins_full"] = int((d > 0).sum()) if k != "automation" else int((d < 0).sum())
        out[f"{k}_dz"] = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
    return pd.DataFrame([out]), df


def latency(n=6000, seed=0):
    """Wall-clock cost of the LLM-DDA decision layer per incident (excludes the LLM itself)."""
    import time
    x = incidents(seed); t0 = time.perf_counter(); run(x, replace(FULL, theta=18.128)); t1 = time.perf_counter()
    return pd.DataFrame([{"incidents": n, "total_s": t1 - t0, "ms_per_incident": 1000 * (t1 - t0) / n}])


def descriptives(seeds=TEST_SEEDS):
    rows = []
    for sd in seeds:
        x = incidents(sd); t = slice(x["n_hist"], None)
        y = x["correct"][t].astype(int)
        rows.append({"accuracy": y.mean(),
                     "auc_self": __import__("sklearn.metrics", fromlist=["x"]).roc_auc_score(y, x["s"][t]),
                     "mean_self_conf": x["s"][t].mean()})
        cal = fit_calibrator(x, np.arange(x["n_hist"]))
        ph = cal.predict_proba(features(x, np.arange(x["n_hist"], len(x["cls"]))))[:, 1]
        from sklearn.metrics import roc_auc_score
        rows[-1]["auc_cal"] = roc_auc_score(y, ph)
        bins = np.clip((ph * 10).astype(int), 0, 9)
        rows[-1]["ece_cal"] = sum(abs(y[bins == k].mean() - ph[bins == k].mean()) * (bins == k).mean()
                                  for k in range(10) if (bins == k).any())
        sb = np.clip((x["s"][t] * 10).astype(int), 0, 9)
        rows[-1]["ece_self"] = sum(abs(y[sb == k].mean() - x["s"][t][sb == k].mean()) * (sb == k).mean()
                                   for k in range(10) if (sb == k).any())
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    pols = [tune(p) for p in BASELINES + ABLATIONS]
    base, abl = pols[:len(BASELINES)], pols[len(BASELINES):]
    df = evaluate(base + abl)
    df.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)
    t1 = summarize(df, [p.name for p in base] + [FULL.name])
    t1.to_csv(os.path.join(OUT, "table1_policies.csv"))
    t2 = summarize(df, [p.name for p in abl])
    t2 = t2.join(paired(df, FULL.name, [p.name for p in abl[1:]]).set_index("comparison"))
    t2.to_csv(os.path.join(OUT, "table2_ablation.csv"))
    paired(df, FULL.name, [p.name for p in base]).to_csv(os.path.join(OUT, "tests_vs_baselines.csv"), index=False)
    paired(df, FULL.name, [p.name for p in base], "severe_k").to_csv(
        os.path.join(OUT, "tests_vs_baselines_severe.csv"), index=False)
    paired(df, FULL.name, [p.name for p in abl[1:]], "drift_esc_k").to_csv(
        os.path.join(OUT, "tests_ablation_drift_escapes.csv"), index=False)
    descriptives().describe().to_csv(os.path.join(OUT, "surrogate_descriptives.csv"))
    fr = frontier(); fr.to_csv(os.path.join(OUT, "figure1a_frontier.csv"), index=False)
    px = paradox(); px.to_csv(os.path.join(OUT, "figure1b_paradox.csv"), index=False)
    named = {p.name: p for p in pols}
    ts = timeseries([named["Calibrated confidence threshold (selective deferral)"],
                     named["- both feedback loops (static)"], named["LLM-DDA (full)"]])
    pd.DataFrame({f"{k}|{m}": v[i] for k, v in ts.items() for i, m in enumerate(["cost", "drift_auto"])}
                 ).to_csv(os.path.join(OUT, "figure2_timeseries.csv"), index=False)
    figures(fr, px, ts)
    sensitivity().to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
    cs, cd = control_calibrated_forcing()
    cs.to_csv(os.path.join(OUT, "control_calibrated_forcing.csv"), index=False)
    cd.to_csv(os.path.join(OUT, "control_calibrated_forcing_per_seed.csv"), index=False)
    latency().to_csv(os.path.join(OUT, "latency.csv"), index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(t1.round(3)); print(t2.round(4))


if __name__ == "__main__":
    main()
