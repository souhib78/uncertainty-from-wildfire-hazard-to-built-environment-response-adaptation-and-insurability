"""UNBURNT prototype (Streamlit version): wildfire hazard -> loss, with uncertainty propagation.
Hazard link: bootstrap ensemble + Platt (sigmoid) calibration + split conformal.
Links 2-4 (response, adaptation, insurability): ILLUSTRATIVE parameters only.
Data: put your real data in data.csv (columns: temperature, wind, humidity, drought, fire)
otherwise a SYNTHETIC dataset is generated.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FEATURES = ["temperature", "wind", "humidity", "drought"]
RNG = np.random.default_rng(42)
B = 50          # bootstrap members
ALPHA = 0.10    # conformal miscoverage


# ---------------- data ----------------
def load_data():
    if os.path.exists("data.csv"):
        return pd.read_csv("data.csv")[FEATURES + ["fire"]], False
    n = 3000
    t = RNG.normal(28, 8, n)
    w = RNG.gamma(3, 4, n)
    h = np.clip(RNG.normal(45, 18, n), 5, 100)
    d = np.clip(RNG.normal(50, 20, n), 0, 100)
    z = -6 + 0.12 * t + 0.05 * w - 0.04 * h + 0.03 * d + RNG.normal(0, 0.8, n)
    y = (RNG.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"temperature": t, "wind": w, "humidity": h,
                         "drought": d, "fire": y}), True


# ---------------- hazard model ----------------
def train():
    df, synthetic = load_data()
    X, y = df[FEATURES].values, df["fire"].values
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0, stratify=ytmp)

    members = []
    for b in range(B):
        idx = RNG.integers(0, len(Xtr), len(Xtr))
        m = RandomForestClassifier(n_estimators=30, max_depth=8, min_samples_leaf=5,
                                   random_state=b)
        m.fit(Xtr[idx], ytr[idx])
        members.append(m)

    def raw(Xq):
        return np.array([m.predict_proba(Xq)[:, 1] for m in members])  # (B, n)

    def logit(p):
        p = np.clip(p, 1e-3, 1 - 1e-3)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit_platt(p_raw, y):
        return LogisticRegression(C=1.0).fit(logit(p_raw), y)

    def conf_q(p, y):
        sc = 1 - np.where(y == 1, p, 1 - p)
        n = len(sc)
        return np.quantile(sc, min(1, np.ceil((n + 1) * (1 - ALPHA)) / n))

    # pool = calibration + test (never used to train the forests)
    Xpool = np.vstack([Xcal, Xte])
    ypool = np.concatenate([ycal, yte])
    pm_raw = raw(Xpool).mean(0)

    # deployed calibrator (Platt / sigmoid) and conformal threshold, fitted on the whole pool
    plat_all = fit_platt(pm_raw, ypool)

    def member_probs(Xq):
        r = raw(Xq)
        return plat_all.predict_proba(logit(r.ravel()))[:, 1].reshape(B, -1)

    q = conf_q(plat_all.predict_proba(logit(pm_raw))[:, 1], ypool)

    # conformal coverage over 200 random 50/50 splits of the pool (mean, sd)
    rs = np.random.default_rng(7)
    covs = []
    for _ in range(200):
        idx = rs.permutation(len(ypool))
        h = len(idx) // 2
        ia, ib = idx[:h], idx[h:]
        pl = fit_platt(pm_raw[ia], ypool[ia])
        pa = pl.predict_proba(logit(pm_raw[ia]))[:, 1]
        pb = pl.predict_proba(logit(pm_raw[ib]))[:, 1]
        qq = conf_q(pa, ypool[ia])
        covs.append(np.mean(1 - np.where(ypool[ib] == 1, pb, 1 - pb) <= qq))
    cover = (float(np.mean(covs)), float(np.std(covs)))

    # calibration curve / Brier on one fixed split
    plat_c = fit_platt(raw(Xcal).mean(0), ycal)
    pt = plat_c.predict_proba(logit(raw(Xte).mean(0)))[:, 1]
    brier = float(np.mean((pt - yte) ** 2))
    bins = np.linspace(0, 1, 11)
    ids = np.clip(np.digitize(pt, bins) - 1, 0, 9)
    bx = [pt[ids == i].mean() for i in range(10) if (ids == i).sum() >= 10]
    by = [yte[ids == i].mean() for i in range(10) if (ids == i).sum() >= 10]
    brier = float(np.mean((pt - yte) ** 2))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(bx, by, "o-", label="Calibrated ensemble")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    ax.set_title("Calibration (held-out test)")
    fig.tight_layout()
    return member_probs, q, cover, brier, fig, synthetic


@st.cache_resource(show_spinner='Training bootstrap ensemble...')
def get_model():
    return train()


member_probs, Q, COVER, BRIER, CALIB_FIG, SYNTHETIC = get_model()
DATA_NOTE = ("⚠️ **Synthetic training data** (demo only). Replace with real data via data.csv."
             if SYNTHETIC else "Trained on user-supplied data.csv.")


def hazard(t, w, h, d):
    p = member_probs(np.array([[t, w, h, d]]))[:, 0]  # (B,)
    return p


# ---------------- tab 1 ----------------
def run_hazard(t, w, h, d):
    p = hazard(t, w, h, d)
    pm, lo, hi = p.mean(), *np.percentile(p, [5, 95])
    alea = float(np.mean(p * (1 - p)))
    epi = float(np.var(p))
    cs = []
    if pm >= 1 - Q:
        cs.append("fire")
    if 1 - pm >= 1 - Q:
        cs.append("no fire")
    cs_txt = "{" + ", ".join(cs) + "}" if cs else "{fire, no fire} (ambiguous)"
    if not cs:
        cs_txt = "{fire, no fire} (model cannot decide at this confidence)"
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(p, bins=15, color="#d9622b", alpha=0.8)
    ax.axvline(pm, color="k")
    ax.set_xlabel("Fire probability (per ensemble member)")
    ax.set_title("Spread of the bootstrap ensemble")
    fig.tight_layout()
    txt = (f"### Fire probability: **{pm:.0%}**  (90% ensemble interval {lo:.0%} – {hi:.0%})\n\n"
           f"- Conformal prediction set (≥{1-ALPHA:.0%} coverage): **{cs_txt}**\n"
           f"- Aleatoric (data noise) ≈ {alea:.3f} | Epistemic (model disagreement) ≈ {epi:.4f}\n\n"
           f"Check over 200 random splits: conformal coverage = {COVER[0]:.1%} ± {COVER[1]:.1%} "
           f"(target {1-ALPHA:.0%}); Brier = {BRIER:.3f}. "
           f"Small sample: treat these figures as indicative.\n\n{DATA_NOTE}")
    return txt, fig


# ---------------- tabs 2-4 (illustrative) ----------------
BASE_DAMAGE = {"Masonry": 0.15, "Timber": 0.45, "Mixed": 0.30}
ADAPT = {"None": 0.0, "Vegetation buffer": 0.25,
         "Fire-resistant materials": 0.40, "Both": 0.55}


def simulate(p_members, btype, dist, adapt, value, vary, n=5000):
    r = np.random.default_rng(1)
    p = r.choice(p_members, n) if "hazard" in vary else np.full(n, p_members.mean())
    base = BASE_DAMAGE[btype] * np.exp(-dist / 150)
    red = ADAPT[adapt]
    if "adaptation" in vary:
        red = np.clip(red * r.lognormal(0, 0.25, n), 0, 0.9)
    m = np.clip(base * (1 - red), 0.001, 0.99)
    if "damage" in vary:
        k = 10
        dmg = r.beta(m * k, (1 - m) * k) if np.ndim(m) == 0 else r.beta(m * k, (1 - m) * k)
    else:
        dmg = np.full(n, m) if np.ndim(m) == 0 else m
    return p * dmg * value


def run_chain(t, w, h, d, btype, dist, adapt, value):
    pm = hazard(t, w, h, d)
    allv = {"hazard", "damage", "adaptation"}
    loss = simulate(pm, btype, dist, adapt, value, allv)
    base = simulate(pm, btype, dist, "None", value, allv)
    prem = loss.mean() + 0.3 * loss.std()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(base, bins=50, alpha=0.5, label="No adaptation")
    ax.hist(loss, bins=50, alpha=0.6, label=f"With: {adapt}")
    ax.set_xlabel("Expected loss per event-year (currency units)")
    ax.legend()
    fig.tight_layout()
    lo, hi = np.percentile(loss, [5, 95])
    txt = (f"**Expected loss:** {loss.mean():,.0f} (90% range {lo:,.0f} – {hi:,.0f})\n\n"
           f"**Without adaptation:** {base.mean():,.0f}  → saving ≈ {base.mean()-loss.mean():,.0f}\n\n"
           f"**Indicative premium** (mean + 0.3·σ loading): {prem:,.0f}\n\n"
           f"Coefficient of variation: {loss.std()/max(loss.mean(),1e-9):.2f}\n\n"
           "_ILLUSTRATIVE: damage curves, adaptation effects and loading are placeholder values, "
           "not calibrated to any dataset._")
    return txt, fig


def run_budget(t, w, h, d, btype, dist, adapt, value):
    pm = hazard(t, w, h, d)
    vs = {k: simulate(pm, btype, dist, adapt, value, {k}).var()
          for k in ["hazard", "damage", "adaptation"]}
    tot = sum(vs.values()) or 1
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(list(vs), [v / tot for v in vs.values()], color=["#d9622b", "#4a7fb5", "#6aa84f"])
    ax.set_ylabel("Share of loss variance")
    ax.set_title("Uncertainty budget (one-at-a-time)")
    fig.tight_layout()
    return fig


LIMITS = """
### Limitations
- Very small dataset (244 daily records, one summer): calibration and coverage are unstable.
- Hazard model trained on tabular weather variables only; no fuel, topography or ignition sources.
- Conformal guarantees assume exchangeability (no temporal/spatial shift handled yet).
- Links 2–4 use **placeholder** parameters; the one-at-a-time budget ignores interactions.
- Ensemble spread captures only part of epistemic uncertainty.

### Roadmap (PhD)
1. Calibrate damage and adaptation curves on literature + UK/Mediterranean loss data.
2. Spatio-temporal validation and transferability (Algeria → UK).
3. Global sensitivity analysis (Sobol) for the uncertainty budget.
4. Insurability metrics beyond the simple premium loading.
"""


st.set_page_config(page_title="UNBURNT prototype", page_icon="🔥", layout="wide")
st.title("🔥 UNBURNT: uncertainty from wildfire hazard to insurability")
st.caption("Prototype: the hazard link is fully uncertainty-quantified; later links are illustrative.")

with st.sidebar:
    st.header("Weather inputs")
    t = st.slider("Temperature (°C)", 0, 55, 35)
    w = st.slider("Wind speed (km/h)", 0, 100, 25)
    h = st.slider("Relative humidity (%)", 0, 100, 25)
    d = st.slider("Dryness index (0-100)", 0, 100, 70)
    st.header("Built environment")
    bt = st.selectbox("Building type", list(BASE_DAMAGE), index=2)
    ds = st.slider("Distance to vegetation (m)", 0, 500, 50)
    ad = st.selectbox("Adaptation", list(ADAPT))
    vl = st.number_input("Asset value", value=200000, step=10000)

tab1, tab2, tab3, tab4 = st.tabs(["1. Hazard", "2-4. Response → Insurability (illustrative)",
                                  "5. Uncertainty budget", "6. Limitations & Roadmap"])
with tab1:
    txt, fig = run_hazard(t, w, h, d)
    st.markdown(txt)
    c1, c2 = st.columns(2)
    c1.pyplot(fig)
    c2.pyplot(CALIB_FIG)
with tab2:
    txt, fig = run_chain(t, w, h, d, bt, ds, ad, vl)
    st.markdown(txt)
    st.pyplot(fig)
with tab3:
    st.pyplot(run_budget(t, w, h, d, bt, ds, ad, vl))
with tab4:
    st.markdown(LIMITS)
