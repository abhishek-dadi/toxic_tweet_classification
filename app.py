"""
app.py: Streamlit demo for the APS toxicity project
=====================================================
Tab 1  Live classifier       : type a comment, both models classify it side by side
Tab 2  Statistical results   : decision, CIs, hypothesis tests and figures from the experiment
Tab 3  Live statistics       : re-sample the test set live (sampling variability, power, bootstrap CI)

Needs the output folder created by toxicity_stats.py (models/, results.json, predictions_test.csv,
thresholds.json, figures/).

Run in Colab:   see README ("Streamlit demo")
Run locally :   streamlit run app.py -- --out_dir path/to/aps_toxicity
"""
import argparse
import json
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Toxic Comment Classifier: APS Demo", page_icon="🛡️", layout="wide")

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default=os.environ.get("APS_OUT_DIR", "/content/drive/MyDrive/aps_toxicity"))
ARGS, _ = parser.parse_known_args()

MODELS = {"baseline": ("BERT-base (baseline)", "bert-base-uncased"),
          "improved": ("Twitter-RoBERTa (improved)", "twitter-roberta-base-2022-154m")}

EXAMPLES = {
    "Friendly": "Congrats on the launch, the new feature looks great!",
    "Disagreement": "I completely disagree with this article, the author ignores the data.",
    "Rude": "This take is dumb, honestly.",
    "Insult": "You are a worthless idiot and nobody cares what you think.",
    "Threat": "I know where you live and I will make you regret this.",
    "Slang": "lmao this ref is trash, worst call of the season",
}


def clean(text):
    t = str(text)
    t = re.sub(r"https?://\S+|www\.\S+", "http", t)
    t = re.sub(r"@\w+", "@user", t)
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------------------- #
# Loading (cached so it happens once per session)
# ----------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading both models ...")
def load_models(out_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {}
    for role, (_, folder) in MODELS.items():
        path = f"{out_dir}/models/{folder}"
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        loaded[role] = (tok, model)
    return loaded, device


@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_predictions(path):
    return pd.read_csv(path)


def predict(loaded, device, texts):
    import torch
    out = {}
    for role, (tok, model) in loaded.items():
        t0 = time.perf_counter()
        with torch.inference_mode():
            enc = tok([clean(t) for t in texts], truncation=True, max_length=128,
                      padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy()
        out[role] = (probs, (time.perf_counter() - t0) * 1000)
    return out


# ----------------------------------------------------------------------------- #
# Fast statistics helpers (same formulas as toxicity_stats.py)
# ----------------------------------------------------------------------------- #
def f1_rows(Y, P):
    tp = (Y & P).sum(1)
    fp = (~Y & P).sum(1)
    fn = (Y & ~P).sum(1)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def mcnemar_pvalues(b, c):
    b, c = np.atleast_1d(b).astype(int), np.atleast_1d(c).astype(int)
    nd = b + c
    p = np.ones(len(b))
    exact = (nd > 0) & (nd < 25)
    p[exact] = np.minimum(1.0, 2 * stats.binom.cdf(np.minimum(b, c)[exact], nd[exact], 0.5))
    big = nd >= 25
    p[big] = stats.chi2.sf((np.abs(b - c)[big] - 1) ** 2 / nd[big], df=1)
    return p


def resample(y, ya, yb, n, reps, rng, chunk=200):
    ca, cb = ya == y, yb == y
    deltas, pvals, f1a_all, f1b_all = [], [], [], []
    for s in range(0, reps, chunk):
        idx = rng.integers(0, len(y), size=(min(chunk, reps - s), n))
        Y = y[idx]
        fa, fb = f1_rows(Y, ya[idx]), f1_rows(Y, yb[idx])
        f1a_all.append(fa); f1b_all.append(fb); deltas.append(fb - fa)
        pvals.append(mcnemar_pvalues((ca[idx] & ~cb[idx]).sum(1), (~ca[idx] & cb[idx]).sum(1)))
    cat = np.concatenate
    return cat(f1a_all), cat(f1b_all), cat(deltas), cat(pvals)


# ----------------------------------------------------------------------------- #
# Sidebar
# ----------------------------------------------------------------------------- #
st.sidebar.title("🛡️ APS Demo")
out_dir = st.sidebar.text_input("Output folder", ARGS.out_dir)
paths = {
    "results": f"{out_dir}/results.json",
    "preds": f"{out_dir}/predictions_test.csv",
    "thr": f"{out_dir}/thresholds.json",
    "figs": f"{out_dir}/figures",
}
thresholds = load_json(paths["thr"]) if os.path.exists(paths["thr"]) else {"baseline": 0.5, "improved": 0.5}
st.sidebar.markdown("**Decision thresholds** (tuned on validation)")
st.sidebar.write({MODELS[r][0]: thresholds[r] for r in MODELS})
st.sidebar.markdown("---")
st.sidebar.caption("Task: binary toxic-comment classification on Civil Comments. "
                   "Question: is the improved model really better, or is the gap test-sample noise?")

st.title("Toxic Comment Classification: Is the Improvement Real?")
tab1, tab2, tab3 = st.tabs(["🔎 Live classifier", "📊 Statistical results", "🎲 Live statistics"])

# ----------------------------------------------------------------------------- #
# Tab 1: Live classifier
# ----------------------------------------------------------------------------- #
with tab1:
    models_ok = all(os.path.exists(f"{out_dir}/models/{folder}/config.json") for _, folder in MODELS.values())
    if not models_ok:
        st.error(f"Trained models not found in {out_dir}/models/. Run toxicity_stats.py first, "
                 "or fix the output folder in the sidebar.")
    else:
        loaded, device = load_models(out_dir)
        st.caption(f"Both models loaded on **{device.upper()}**.")

        if "text" not in st.session_state:
            st.session_state.text = EXAMPLES["Insult"]

        def set_text(t):
            st.session_state.text = t

        st.write("Try an example:")
        cols = st.columns(len(EXAMPLES))
        for col, (name, ex) in zip(cols, EXAMPLES.items()):
            col.button(name, on_click=set_text, args=(ex,), use_container_width=True)

        text = st.text_area("Comment or tweet", key="text", height=100)
        if st.button("Classify", type="primary") and text.strip():
            res = predict(loaded, device, [text])
            c1, c2 = st.columns(2)
            verdicts = {}
            for col, role in zip((c1, c2), MODELS):
                p, ms = res[role][0][0], res[role][1]
                toxic = p >= thresholds[role]
                verdicts[role] = toxic
                with col:
                    st.subheader(MODELS[role][0])
                    st.metric("P(toxic)", f"{p:.1%}", "TOXIC" if toxic else "not toxic",
                              delta_color="inverse" if toxic else "normal")
                    st.progress(float(p))
                    st.caption(f"Threshold {thresholds[role]:.2f} · {ms:.0f} ms")
            if verdicts["baseline"] == verdicts["improved"]:
                st.info("Both models agree.")
            else:
                st.warning("The models disagree on this comment. Disagreements like this are exactly what "
                           "McNemar's test counts (b and c).")

        st.markdown("---")
        st.subheader("Batch mode")
        batch = st.text_area("One comment per line", "\n".join(list(EXAMPLES.values())[:4]), height=130)
        if st.button("Classify all"):
            lines = [l for l in batch.splitlines() if l.strip()]
            if lines:
                res = predict(loaded, device, lines)
                df = pd.DataFrame({"comment": lines})
                for role in MODELS:
                    df[f"P(toxic) {MODELS[role][0]}"] = res[role][0].round(3)
                    df[f"Toxic? {MODELS[role][0]}"] = res[role][0] >= thresholds[role]
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(" · ".join(f"{MODELS[r][0]}: {res[r][1]:.0f} ms for {len(lines)} comments" for r in MODELS))

# ----------------------------------------------------------------------------- #
# Tab 2: Results of the experiment
# ----------------------------------------------------------------------------- #
with tab2:
    if not os.path.exists(paths["results"]):
        st.error(f"{paths['results']} not found. Run toxicity_stats.py first.")
    else:
        r = load_json(paths["results"])
        a = r["config"]["alpha"]
        t = r["test_set"]
        st.caption(f"Held-out test set: n = {t['n']:,} comments, {t['positives']:,} toxic "
                   f"({t['prevalence']:.2%}). Significance level α = {a}.")
        if r["decision"].startswith("Reject"):
            st.success("**Decision:** " + r["decision"])
        else:
            st.warning("**Decision:** " + r["decision"])

        st.subheader("Performance with 95% bootstrap confidence intervals")
        rows = []
        for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            row = {"Metric": m}
            for role in MODELS:
                v = r["point_metrics"][role][m]
                ci = r["confidence_intervals"][role].get(m)
                row[MODELS[role][0]] = f"{v:.4f}" + (f"  [{ci['ci'][0]:.4f}, {ci['ci'][1]:.4f}]" if ci else "")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Paired difference (improved − baseline)")
        drows = []
        for m, d in r["paired_difference"].items():
            sig = d["ci"][0] > 0 or d["ci"][1] < 0
            drows.append({"Metric": m, "Δ": f"{d['observed_delta']:+.4f}",
                          "95% CI": f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]",
                          "Bootstrap SE": f"{d['bootstrap_se']:.4f}", "z": f"{d['z']:.2f}",
                          "p-value": f"{d['p_value_bootstrap']:.4g}",
                          "Significant?": "✅ yes" if sig else "❌ no (CI contains 0)"})
        st.dataframe(pd.DataFrame(drows), use_container_width=True, hide_index=True)

        st.subheader("Hypothesis tests")
        mc, pt = r["mcnemar"], r["permutation_test"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Permutation test p (F1)", f"{pt['p_value']:.3f}")
        c2.metric("McNemar p (error rate)", f"{mc['p_value']:.3f}")
        c3.metric("Only baseline correct (b)", f"{mc['b_baseline_only_correct']:,}")
        c4.metric("Only improved correct (c)", f"{mc['c_improved_only_correct']:,}")

        st.subheader("Figures")
        figs = [("bootstrap_delta_f1.png", "Bootstrap distribution of ΔF1"),
                ("permutation_test.png", "Permutation test null distribution"),
                ("metric_confidence_intervals.png", "Metrics with confidence intervals"),
                ("confusion_matrices.png", "Confusion matrices"),
                ("roc_curves.png", "ROC curves")]
        figs = [(f, c) for f, c in figs if os.path.exists(f"{paths['figs']}/{f}")]
        for i in range(0, len(figs), 2):
            cols = st.columns(2)
            for col, (f, cap) in zip(cols, figs[i:i + 2]):
                col.image(f"{paths['figs']}/{f}", caption=cap, use_container_width=True)
        if os.path.exists(f"{paths['figs']}/sampling_experiment.png"):
            st.image(f"{paths['figs']}/sampling_experiment.png",
                     caption="Sampling-variability experiment", use_container_width=True)

# ----------------------------------------------------------------------------- #
# Tab 3: Live statistics on the saved test predictions
# ----------------------------------------------------------------------------- #
with tab3:
    if not os.path.exists(paths["preds"]):
        st.error(f"{paths['preds']} not found. Run toxicity_stats.py first.")
    else:
        pred = load_predictions(paths["preds"])
        y = pred["label"].values.astype(bool)
        ya = pred["p_baseline"].values >= thresholds["baseline"]
        yb = pred["p_improved"].values >= thresholds["improved"]
        N = len(y)
        st.caption(f"Uses the saved predictions of both models on all {N:,} test comments. "
                   "Nothing is retrained: we only change WHICH comments are in the test set.")

        # A. one random test set at a time
        st.subheader("A. Draw a random test set")
        st.write("Each click draws a new random test set. Watch how the 'winner' changes when the test set is small.")
        cA1, cA2 = st.columns([1, 2])
        with cA1:
            nA = st.select_slider("Test-set size", [100, 250, 500, 1000, 2500, 5000, 10000, 20000], value=500, key="nA")
            if st.button("🎲 Draw test set"):
                idx = np.random.default_rng().integers(0, N, size=(1, nA))
                fa, fb = f1_rows(y[idx], ya[idx])[0], f1_rows(y[idx], yb[idx])[0]
                st.session_state.setdefault("draws", []).append({"n": nA, "F1 baseline": fa, "F1 improved": fb,
                                                                 "ΔF1": fb - fa,
                                                                 "Winner": "Improved" if fb > fa else ("Baseline" if fa > fb else "Tie")})
            if st.button("Clear draws"):
                st.session_state.draws = []
        with cA2:
            draws = st.session_state.get("draws", [])
            if draws:
                d = pd.DataFrame(draws)
                last = d.iloc[-1]
                st.metric(f"Last draw (n = {int(last['n']):,})", f"ΔF1 = {last['ΔF1']:+.4f}", last["Winner"] + " wins",
                          delta_color="off")
                st.write(f"Improved model won **{(d['Winner'] == 'Improved').sum()} of {len(d)}** draws.")
                st.dataframe(d.iloc[::-1].round(4), use_container_width=True, hide_index=True, height=200)

        st.markdown("---")
        # B. many test sets
        st.subheader("B. Repeat many times: sampling distribution and power")
        cB1, cB2 = st.columns([1, 2])
        with cB1:
            nB = st.select_slider("Test-set size", [100, 250, 500, 1000, 2500, 5000, 10000, 20000], value=1000, key="nB")
            reps = st.slider("Number of random test sets", 100, 2000, 1000, step=100)
            alpha = st.select_slider("Significance level α", [0.01, 0.05, 0.10], value=0.05)
            runB = st.button("Run simulation", type="primary")
        if runB:
            fa, fb, d, pv = resample(y, ya, yb, nB, reps, np.random.default_rng())
            with cB2:
                m1, m2, m3 = st.columns(3)
                m1.metric("P(improved wins)", f"{np.mean(d > 0):.1%}")
                m2.metric("SD of ΔF1 (standard error)", f"{d.std(ddof=1):.4f}")
                m3.metric(f"Power (McNemar p < {alpha})", f"{np.mean(pv < alpha):.1%}")
                fig, ax = plt.subplots(figsize=(7, 3.2))
                ax.hist(d, bins=50, color="#4C78A8", alpha=0.8)
                ax.axvline(0, color="red", lw=2, label="no difference")
                ax.axvline(d.mean(), color="black", ls="--", label=f"mean ΔF1 = {d.mean():+.4f}")
                ax.set_xlabel("ΔF1 = F1(improved) − F1(baseline)"); ax.set_ylabel("count")
                ax.set_title(f"{reps:,} random test sets of size {nB:,}"); ax.legend(fontsize=8)
                st.pyplot(fig, use_container_width=True)
                st.caption("Area to the left of the red line = test sets where the BASELINE would have looked better.")

        st.markdown("---")
        # C. bootstrap CI on the full test set
        st.subheader("C. Bootstrap confidence interval for ΔF1 (full test set)")
        cC1, cC2 = st.columns([1, 2])
        with cC1:
            B = st.select_slider("Bootstrap resamples B", [500, 1000, 2000, 5000, 10000], value=2000)
            level = st.select_slider("Confidence level", [0.90, 0.95, 0.99], value=0.95)
            runC = st.button("Compute bootstrap CI")
        if runC:
            _, _, d, _ = resample(y, ya, yb, N, B, np.random.default_rng())
            lo, hi = np.percentile(d, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
            obs = f1_rows(y[None, :], yb[None, :])[0] - f1_rows(y[None, :], ya[None, :])[0]
            with cC2:
                st.metric(f"Observed ΔF1 with {int(level * 100)}% CI", f"{obs:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]",
                          delta_color="off")
                if lo > 0 or hi < 0:
                    st.success("The interval excludes 0: the difference is statistically significant.")
                else:
                    st.warning("The interval contains 0: the difference could be due to test-sample variation.")
                fig, ax = plt.subplots(figsize=(7, 3.2))
                ax.hist(d, bins=50, color="#4C78A8", alpha=0.8, density=True)
                xs = np.linspace(d.min(), d.max(), 200)
                ax.plot(xs, stats.norm.pdf(xs, d.mean(), d.std(ddof=1)), "k-", lw=1, label="normal approx. (CLT)")
                ax.axvline(lo, color="green", ls="--"); ax.axvline(hi, color="green", ls="--", label=f"{int(level * 100)}% CI")
                ax.axvline(0, color="red", lw=2, label="0")
                ax.set_xlabel("bootstrap ΔF1"); ax.legend(fontsize=8)
                st.pyplot(fig, use_container_width=True)
