#!/usr/bin/env python3
"""
Applying Probability and Statistics to Evaluate a Machine Learning Model
=========================================================================
Task      : Binary toxic-comment classification (toxic vs. non-toxic)
Dataset   : Civil Comments (Jigsaw / Google), loaded from Hugging Face `google/civil_comments`
Baseline  : BERT-base (bert-base-uncased), fine-tuned
Improved  : Twitter-RoBERTa (cardiffnlp/twitter-roberta-base-2022-154m), fine-tuned.
            Same architecture family, but pretrained on 154M tweets (domain-adapted).

Research question
-----------------
Does the improved model perform better, or could the observed difference be due to
variation in the test sample?

Pipeline
--------
1. Data preparation and SAMPLING
   - train : disproportionate stratified sample from the official train split (25% toxic)
   - val   : simple random sample from the official validation split (threshold tuning)
   - test  : proportionate stratified random sample from the official test split
   - sampling check: one-sample z-test that sample prevalence matches population prevalence
2. Training both models with identical data, hyper-parameters and seed
3. Evaluation on the SAME held-out test set (paired design)
4. STATISTICAL ESTIMATION
   - Wilson score 95% CI for accuracy (binomial model)
   - Bootstrap 95% CIs for F1, accuracy, ROC-AUC of each model
   - Paired bootstrap 95% CI for the DIFFERENCE (improved - baseline)
5. HYPOTHESIS TESTING (alpha = 0.05)
   - McNemar's test on paired correctness (H0: equal error rates)
   - Paired permutation test on F1 (H0: models are exchangeable, delta F1 = 0)
   - Normal-approximation z-test using the bootstrap standard error (CLT)
6. SAMPLING-VARIABILITY EXPERIMENT
   - Repeatedly draw test sets of size n = 100 ... N and measure how often the
     improved model wins and how often the difference is significant (empirical power)

Usage (Google Colab, GPU runtime)
---------------------------------
    from google.colab import drive; drive.mount('/content/drive')
    !python toxicity_stats.py --out_dir /content/drive/MyDrive/aps_toxicity

    # statistics only (re-uses saved predictions, no GPU needed):
    !python toxicity_stats.py --out_dir /content/drive/MyDrive/aps_toxicity --skip_training
"""

import argparse
import glob
import inspect
import json
import math
import os
import random
import re
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

BASELINE_MODEL = "bert-base-uncased"
IMPROVED_MODEL = "cardiffnlp/twitter-roberta-base-2022-154m"


# ----------------------------------------------------------------------------- #
# 0. Configuration and reproducibility
# ----------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="APS project: statistical comparison of two toxicity classifiers")
    p.add_argument("--out_dir", default="./aps_toxicity_output")
    p.add_argument("--baseline", default=BASELINE_MODEL)
    p.add_argument("--improved", default=IMPROVED_MODEL)
    p.add_argument("--train_size", type=int, default=100_000)
    p.add_argument("--train_pos_frac", type=float, default=0.25, help="share of toxic comments in training sample")
    p.add_argument("--val_size", type=int, default=15_000)
    p.add_argument("--test_size", type=int, default=20_000)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--n_boot", type=int, default=10_000, help="bootstrap resamples for F1/accuracy")
    p.add_argument("--n_boot_auc", type=int, default=2_000, help="bootstrap resamples for ROC-AUC (slower)")
    p.add_argument("--n_perm", type=int, default=10_000, help="permutations for the permutation test")
    p.add_argument("--n_sub", type=int, default=1_000, help="repetitions per size in sampling experiment")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrain", action="store_true", help="ignore saved models and train again")
    p.add_argument("--skip_training", action="store_true", help="only run statistics on saved predictions")
    args, _ = p.parse_known_args()   # parse_known_args: safe inside notebooks too
    return args


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clean(text):
    t = str(text)
    t = re.sub(r"https?://\S+|www\.\S+", "http", t)
    t = re.sub(r"@\w+", "@user", t)
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------------------- #
# 1. Data preparation and sampling
# ----------------------------------------------------------------------------- #
def prevalence_z_test(y_sample, p_pop):
    """One-sample z-test for a proportion: does the sample prevalence match the population?"""
    n, k = len(y_sample), int(np.sum(y_sample))
    p_hat = k / n
    se = math.sqrt(p_pop * (1 - p_pop) / n)
    z = (p_hat - p_pop) / se
    return {"n": n, "positives": k, "sample_prevalence": p_hat, "population_prevalence": p_pop,
            "z": z, "p_value": 2 * stats.norm.sf(abs(z))}


def prepare_data(args, data_dir):
    paths = {s: f"{data_dir}/{s}.csv" for s in ("train", "val", "test")}
    summary_path = f"{data_dir}/data_summary.json"
    if all(os.path.exists(p) for p in paths.values()) and os.path.exists(summary_path):
        print(f"[data] Using cached samples from {data_dir}")
        data = {s: pd.read_csv(p, keep_default_na=False) for s, p in paths.items()}
        return data, json.load(open(summary_path))

    from datasets import load_dataset
    print("[data] Downloading google/civil_comments from Hugging Face ...")
    cc = load_dataset("google/civil_comments")

    def population(split):
        d = cc[split]
        d = d.remove_columns([c for c in d.column_names if c not in ("text", "toxicity")])
        df = d.to_pandas().reset_index().rename(columns={"index": "orig_idx"})
        df["label"] = (df["toxicity"] >= 0.5).astype(int)   # majority of annotators said toxic
        df["text"] = df["text"].map(clean)
        return df[df["text"].str.len() > 0]

    tr_pop, va_pop, te_pop = population("train"), population("validation"), population("test")

    # Train: disproportionate stratified sample (oversample the rare toxic class to 25%)
    n_pos = int(round(args.train_size * args.train_pos_frac))
    train = pd.concat([
        tr_pop[tr_pop.label == 1].sample(n=n_pos, random_state=args.seed),
        tr_pop[tr_pop.label == 0].sample(n=args.train_size - n_pos, random_state=args.seed),
    ]).sample(frac=1, random_state=args.seed)

    # Validation: simple random sample (natural class balance, used to tune the threshold)
    val = va_pop.sample(n=args.val_size, random_state=args.seed)

    # Test: proportionate stratified random sample (same class proportions as population)
    frac = args.test_size / len(te_pop)
    test = (te_pop.groupby("label", group_keys=False)
                  .sample(frac=frac, random_state=args.seed)
                  .sample(frac=1, random_state=args.seed))

    data = {"train": train, "val": val, "test": test}
    for s, df in data.items():
        df[["orig_idx", "text", "toxicity", "label"]].to_csv(paths[s], index=False)

    summary = {
        "population_sizes": {"train": len(tr_pop), "validation": len(va_pop), "test": len(te_pop)},
        "population_prevalence": {"train": tr_pop.label.mean(), "validation": va_pop.label.mean(),
                                  "test": te_pop.label.mean()},
        "sample_sizes": {s: len(df) for s, df in data.items()},
        "sample_prevalence": {s: df.label.mean() for s, df in data.items()},
        "sampling_check_val": prevalence_z_test(val.label.values, va_pop.label.mean()),
        "sampling_check_test": prevalence_z_test(test.label.values, te_pop.label.mean()),
    }
    json.dump(summary, open(summary_path, "w"), indent=2, default=float)
    data = {s: pd.read_csv(p, keep_default_na=False) for s, p in paths.items()}
    return data, summary


# ----------------------------------------------------------------------------- #
# 2. Model training and prediction
# ----------------------------------------------------------------------------- #
def train_or_load(model_name, train_df, args, model_root, ckpt_root):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    run = model_name.split("/")[-1]
    model_dir, ckpt_dir = f"{model_root}/{run}", f"{ckpt_root}/{run}"
    if os.path.exists(f"{model_dir}/config.json") and not args.retrain:
        print(f"[train] Saved model found for {run}, skipping training.")
        return model_dir

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "non_toxic", 1: "toxic"},
        label2id={"non_toxic": 0, "toxic": 1})

    total_steps = int(math.ceil(len(ds) / args.batch_size) * args.epochs)
    wanted = dict(
        output_dir=ckpt_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, learning_rate=args.lr,
        weight_decay=0.01, warmup_steps=int(0.06 * total_steps), lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(), eval_strategy="no",
        save_strategy="steps", save_steps=1000, save_total_limit=1,
        logging_steps=200, report_to="none", seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=2, group_by_length=True)
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    targs = TrainingArguments(**{k: v for k, v in wanted.items() if k in supported})

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorWithPadding(tok))
    ckpts = sorted(glob.glob(f"{ckpt_dir}/checkpoint-*"), key=lambda x: int(x.split("-")[-1]))
    print(f"[train] {run}: {len(ds):,} examples, {total_steps} steps, resume={ckpts[-1] if ckpts else None}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=ckpts[-1] if ckpts else None)
    print(f"[train] {run} finished in {(time.time() - t0) / 60:.1f} min")

    trainer.save_model(model_dir)
    tok.save_pretrained(model_dir)
    shutil.rmtree(ckpt_dir, ignore_errors=True)   # free Drive space
    del trainer, model
    torch.cuda.empty_cache()
    return model_dir


def predict_proba(model_dir, texts, args, batch_size=256):
    """Returns P(toxic) for every text."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    if device == "cuda":
        model.half()
    texts = [str(t) for t in texts]
    order = np.argsort([len(t) for t in texts])   # length-sorted batches = less padding = faster
    probs = np.empty(len(texts))
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            probs[idx] = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return probs


def tune_threshold(y, p):
    """Pick the decision threshold that maximises F1 on the validation sample."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


# ----------------------------------------------------------------------------- #
# 3-6. Statistics
# ----------------------------------------------------------------------------- #
def point_metrics(y, yhat, p):
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": roc_auc_score(y, p)}


def wilson_ci(k, n, alpha):
    """Wilson score interval for a binomial proportion (e.g. accuracy = k correct out of n)."""
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def _f1_acc_rows(Y, P):
    """Row-wise F1 and accuracy for boolean matrices (each row = one resampled test set)."""
    tp = (Y & P).sum(1)
    fp = (~Y & P).sum(1)
    fn = (Y & ~P).sum(1)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    acc = (Y == P).mean(1)
    return f1, acc


def mcnemar_pvalues(b, c):
    """McNemar test p-values (vectorised). Exact binomial if b+c < 25, else chi-square with
    continuity correction. b = baseline right & improved wrong; c = baseline wrong & improved right."""
    b, c = np.atleast_1d(b).astype(int), np.atleast_1d(c).astype(int)
    nd = b + c
    p = np.ones(len(b))
    exact = (nd > 0) & (nd < 25)
    p[exact] = np.minimum(1.0, 2 * stats.binom.cdf(np.minimum(b, c)[exact], nd[exact], 0.5))
    big = nd >= 25
    chi2 = (np.abs(b - c)[big] - 1) ** 2 / nd[big]
    p[big] = stats.chi2.sf(chi2, df=1)
    return p


def mcnemar_test(y, ya, yb):
    ca, cb = (ya == y), (yb == y)
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    table = [[int(np.sum(ca & cb)), b], [c, int(np.sum(~ca & ~cb))]]
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0.0
    return {"table_[base_correct,base_wrong]x[imp_correct,imp_wrong]": table,
            "b_baseline_only_correct": b, "c_improved_only_correct": c,
            "chi2": chi2, "method": "exact binomial" if b + c < 25 else "chi-square (continuity corrected)",
            "p_value": float(mcnemar_pvalues(b, c)[0]),
            "odds_ratio_c_over_b": c / b if b > 0 else float("inf")}


def paired_bootstrap(y, ya, yb, pa, pb, n_boot, n_boot_auc, rng, chunk=250):
    """Resample test examples WITH replacement; both models are scored on the same resample."""
    n = len(y)
    f1a, f1b, acca, accb = [], [], [], []
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - s), n))
        Y = y[idx]
        fa, aa = _f1_acc_rows(Y, ya[idx])
        fb, ab = _f1_acc_rows(Y, yb[idx])
        f1a.append(fa); f1b.append(fb); acca.append(aa); accb.append(ab)
    auca, aucb = [], []
    for _ in range(n_boot_auc):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        auca.append(roc_auc_score(yy, pa[idx]))
        aucb.append(roc_auc_score(yy, pb[idx]))
    cat = np.concatenate
    return {"f1": (cat(f1a), cat(f1b)), "accuracy": (cat(acca), cat(accb)),
            "roc_auc": (np.array(auca), np.array(aucb))}


def permutation_test_f1(y, ya, yb, n_perm, rng, chunk=250):
    """Paired permutation test. Under H0 the two models are exchangeable, so for every test
    example we randomly swap which model's prediction is called 'baseline' and 'improved'."""
    n = len(y)
    obs = f1_score(y, yb, zero_division=0) - f1_score(y, ya, zero_division=0)
    null = []
    for s in range(0, n_perm, chunk):
        m = min(chunk, n_perm - s)
        Y = np.broadcast_to(y, (m, n))
        swap = rng.random((m, n)) < 0.5
        A = np.where(swap, yb, ya)
        B = np.where(swap, ya, yb)
        null.append(_f1_acc_rows(Y, B)[0] - _f1_acc_rows(Y, A)[0])
    null = np.concatenate(null)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"observed_delta_f1": obs, "p_value": p, "null_mean": null.mean(), "null_sd": null.std(ddof=1)}, null


def sampling_experiment(y, ya, yb, sizes, reps, alpha, rng, chunk=250):
    """How would our conclusion change if the test set had been a different random sample of size n?"""
    ca, cb = (ya == y), (yb == y)
    rows, dists = [], {}
    for n in sizes:
        deltas, pvals = [], []
        for s in range(0, reps, chunk):
            idx = rng.integers(0, len(y), size=(min(chunk, reps - s), n))
            Y = y[idx]
            deltas.append(_f1_acc_rows(Y, yb[idx])[0] - _f1_acc_rows(Y, ya[idx])[0])
            b = (ca[idx] & ~cb[idx]).sum(1)
            c = (~ca[idx] & cb[idx]).sum(1)
            pvals.append(mcnemar_pvalues(b, c))
        d, pv = np.concatenate(deltas), np.concatenate(pvals)
        dists[n] = d
        rows.append({"n": n, "mean_delta_f1": d.mean(), "sd_delta_f1": d.std(ddof=1),
                     "P(improved_wins)": np.mean(d > 0), "P(baseline_wins)": np.mean(d < 0),
                     "power_mcnemar": np.mean(pv < alpha)})
    return pd.DataFrame(rows), dists


# ----------------------------------------------------------------------------- #
# Plots
# ----------------------------------------------------------------------------- #
def make_plots(fig_dir, y, preds, names, boot, cis, null, perm, sub_df, sub_dists, alpha):
    pa, pb, ya, yb = preds
    nb, ni = names

    # ROC curves
    plt.figure(figsize=(5.5, 5))
    for p, name in [(pa, nb), (pb, ni)]:
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves on the held-out test set"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(f"{fig_dir}/roc_curves.png", dpi=150); plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, yh, name in [(axes[0], ya, nb), (axes[1], yb, ni)]:
        cm = confusion_matrix(y, yh)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["non-toxic", "toxic"]); ax.set_yticks([0, 1], ["non-toxic", "toxic"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
    plt.tight_layout(); plt.savefig(f"{fig_dir}/confusion_matrices.png", dpi=150); plt.close()

    # Metric estimates with CIs
    metrics = ["accuracy", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    plt.figure(figsize=(7, 4.5))
    for k, (role, name, off) in enumerate([("baseline", nb, -0.15), ("improved", ni, 0.15)]):
        est = [cis[role][m]["estimate"] for m in metrics]
        lo = [cis[role][m]["ci"][0] for m in metrics]
        hi = [cis[role][m]["ci"][1] for m in metrics]
        plt.errorbar(x + off, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)],
                     fmt="o", capsize=5, label=name)
    plt.xticks(x, ["Accuracy", "F1 (toxic)", "ROC-AUC"])
    plt.ylabel("Score"); plt.title(f"Point estimates with {int((1 - alpha) * 100)}% bootstrap CIs")
    plt.legend(); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{fig_dir}/metric_confidence_intervals.png", dpi=150); plt.close()

    # Bootstrap distribution of delta F1
    d = boot["f1"][1] - boot["f1"][0]
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    plt.figure(figsize=(7, 4.5))
    plt.hist(d, bins=60, density=True, alpha=0.7, color="tab:blue", label="bootstrap ΔF1")
    xs = np.linspace(d.min(), d.max(), 300)
    plt.plot(xs, stats.norm.pdf(xs, d.mean(), d.std(ddof=1)), "k-", lw=1.2, label="normal approximation")
    plt.axvline(lo, color="tab:green", ls="--", label=f"{int((1 - alpha) * 100)}% CI [{lo:.4f}, {hi:.4f}]")
    plt.axvline(hi, color="tab:green", ls="--")
    plt.axvline(0, color="red", lw=1.5, label="no difference (0)")
    plt.xlabel("ΔF1 = F1(improved) − F1(baseline)"); plt.ylabel("Density")
    plt.title("Sampling distribution of ΔF1 (paired bootstrap)"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{fig_dir}/bootstrap_delta_f1.png", dpi=150); plt.close()

    # Permutation null distribution
    plt.figure(figsize=(7, 4.5))
    plt.hist(null, bins=60, density=True, alpha=0.7, color="gray", label="null distribution (H0 true)")
    plt.axvline(perm["observed_delta_f1"], color="red", lw=2,
                label=f"observed ΔF1 = {perm['observed_delta_f1']:.4f}\np = {perm['p_value']:.4g}")
    plt.xlabel("ΔF1 under random label swapping"); plt.ylabel("Density")
    plt.title("Paired permutation test"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{fig_dir}/permutation_test.png", dpi=150); plt.close()

    # Sampling experiment
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sizes = list(sub_df["n"])
    box_kw = "tick_labels" if "tick_labels" in inspect.signature(plt.Axes.boxplot).parameters else "labels"
    axes[0].boxplot([sub_dists[n] for n in sizes], showfliers=False, **{box_kw: [str(n) for n in sizes]})
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_xlabel("test-set size n"); axes[0].set_ylabel("ΔF1")
    axes[0].set_title("ΔF1 across random test samples")
    sd = sub_df["sd_delta_f1"].values
    c = sd[-1] * math.sqrt(sizes[-1])
    axes[1].loglog(sizes, sd, "o-", label="observed SD of ΔF1")
    axes[1].loglog(sizes, [c / math.sqrt(n) for n in sizes], "k--", label="c / √n (CLT)")
    axes[1].set_xlabel("test-set size n"); axes[1].set_ylabel("standard deviation")
    axes[1].set_title("Standard error shrinks like 1/√n"); axes[1].legend()
    axes[2].semilogx(sizes, sub_df["P(improved_wins)"], "o-", label="P(improved model wins)")
    axes[2].semilogx(sizes, sub_df["power_mcnemar"], "s-", label=f"P(McNemar p < {alpha})  (power)")
    axes[2].axhline(0.8, color="gray", ls=":", label="80% power")
    axes[2].set_ylim(0, 1.05); axes[2].set_xlabel("test-set size n"); axes[2].set_ylabel("probability")
    axes[2].set_title("How reliable is the conclusion?"); axes[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{fig_dir}/sampling_experiment.png", dpi=150); plt.close()


# ----------------------------------------------------------------------------- #
# Main statistical analysis
# ----------------------------------------------------------------------------- #
def run_statistics(y, pa, pb, thr_a, thr_b, names, args, out_dir, data_summary=None):
    rng = np.random.default_rng(args.seed)
    fig_dir = f"{out_dir}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    y = y.astype(bool)
    ya, yb = pa >= thr_a, pb >= thr_b
    n = len(y)
    a = args.alpha
    q = [100 * a / 2, 100 * (1 - a / 2)]

    print("[stats] point estimates ...")
    pm = {"baseline": point_metrics(y, ya, pa), "improved": point_metrics(y, yb, pb)}

    print(f"[stats] paired bootstrap (B={args.n_boot}, AUC B={args.n_boot_auc}) ...")
    boot = paired_bootstrap(y, ya, yb, pa, pb, args.n_boot, args.n_boot_auc, rng)
    cis = {"baseline": {}, "improved": {}}
    for m, (ba, bb) in boot.items():
        for role, arr in [("baseline", ba), ("improved", bb)]:
            cis[role][m] = {"estimate": pm[role][m], "ci": list(np.percentile(arr, q)),
                            "bootstrap_se": float(arr.std(ddof=1))}
    for role, yh in [("baseline", ya), ("improved", yb)]:
        k = int(np.sum(yh == y))
        cis[role]["accuracy"]["wilson_ci"] = list(wilson_ci(k, n, a))

    diff = {}
    for m, (ba, bb) in boot.items():
        d = bb - ba
        obs = pm["improved"][m] - pm["baseline"][m]
        se = d.std(ddof=1)
        z = obs / se if se > 0 else float("inf")
        p_boot = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        diff[m] = {"observed_delta": obs, "ci": list(np.percentile(d, q)), "bootstrap_se": se,
                   "z": z, "p_value_normal_approx": 2 * stats.norm.sf(abs(z)),
                   "p_value_bootstrap": max(p_boot, 1 / len(d)),
                   "P(delta>0)": float(np.mean(d > 0)),
                   "bootstrap_skewness": float(stats.skew(d))}

    print("[stats] McNemar test ...")
    mc = mcnemar_test(y, ya, yb)

    print(f"[stats] permutation test (R={args.n_perm}) ...")
    perm, null = permutation_test_f1(y, ya, yb, args.n_perm, rng)

    print(f"[stats] sampling-variability experiment ({args.n_sub} reps per size) ...")
    sizes = sorted({s for s in [100, 250, 500, 1000, 2500, 5000, 10000] if s < n} | {n})
    sub_df, sub_dists = sampling_experiment(y, ya, yb, sizes, args.n_sub, a, rng)
    sub_df.to_csv(f"{out_dir}/sampling_experiment.csv", index=False)

    print("[stats] plots ...")
    make_plots(fig_dir, y, (pa, pb, ya, yb), names, boot, cis, null, perm, sub_df, sub_dists, a)

    # Decision
    f1_ci = diff["f1"]["ci"]
    ci_excludes_zero = f1_ci[0] > 0 or f1_ci[1] < 0
    significant = ci_excludes_zero and perm["p_value"] < a
    direction = "better" if diff["f1"]["observed_delta"] > 0 else "worse"
    powered = sub_df[sub_df["power_mcnemar"] >= 0.8]
    n80 = int(powered["n"].min()) if len(powered) else None
    if significant:
        decision = (f"Reject H0 at alpha = {a}. The improved model is significantly {direction} than the "
                    f"baseline on F1 (ΔF1 = {diff['f1']['observed_delta']:+.4f}, "
                    f"{int((1 - a) * 100)}% CI [{f1_ci[0]:+.4f}, {f1_ci[1]:+.4f}], permutation p = {perm['p_value']:.4g}). "
                    f"The difference is unlikely to be explained by test-sample variation alone.")
    else:
        decision = (f"Fail to reject H0 at alpha = {a}. The observed ΔF1 = {diff['f1']['observed_delta']:+.4f} "
                    f"has a {int((1 - a) * 100)}% CI [{f1_ci[0]:+.4f}, {f1_ci[1]:+.4f}] that includes 0 "
                    f"(permutation p = {perm['p_value']:.4g}), so the difference could plausibly be due to "
                    f"variation in the test sample.")

    results = {
        "config": vars(args), "models": {"baseline": names[0], "improved": names[1]},
        "thresholds": {"baseline": thr_a, "improved": thr_b},
        "test_set": {"n": n, "positives": int(y.sum()), "prevalence": float(y.mean())},
        "data_summary": data_summary, "point_metrics": pm, "confidence_intervals": cis,
        "paired_difference": diff, "mcnemar": mc, "permutation_test": perm,
        "sampling_experiment": sub_df.to_dict(orient="records"),
        "n_for_80pct_power": n80, "decision": decision,
    }
    json.dump(results, open(f"{out_dir}/results.json", "w"), indent=2, default=float)
    write_summary(results, f"{out_dir}/results_summary.md")
    print("\n" + "=" * 80 + "\nDECISION: " + decision + "\n" + "=" * 80)
    print(f"Results: {out_dir}/results_summary.md | figures: {fig_dir}/")
    return results


def write_summary(r, path):
    nb, ni = r["models"]["baseline"], r["models"]["improved"]
    a = r["config"]["alpha"]
    pct = int((1 - a) * 100)
    L = [f"# Results: {nb} (baseline) vs {ni} (improved)\n"]
    t = r["test_set"]
    L.append(f"Test set: n = {t['n']:,}, toxic = {t['positives']:,} (prevalence {t['prevalence']:.3%}). "
             f"Thresholds tuned on validation: baseline {r['thresholds']['baseline']}, "
             f"improved {r['thresholds']['improved']}.\n")
    ds = r.get("data_summary")
    if ds:
        L.append("## Sampling check (one-sample z-test for prevalence)\n")
        L.append("| Sample | n | Sample prevalence | Population prevalence | z | p |\n|---|---|---|---|---|---|")
        for k in ("sampling_check_val", "sampling_check_test"):
            s = ds[k]
            L.append(f"| {k.split('_')[-1]} | {s['n']:,} | {s['sample_prevalence']:.4f} | "
                     f"{s['population_prevalence']:.4f} | {s['z']:.3f} | {s['p_value']:.3f} |")
        L.append("")
    L.append(f"## Point estimates and {pct}% bootstrap confidence intervals\n")
    L.append("| Metric | Baseline | Improved |\n|---|---|---|")
    for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        row = f"| {m} |"
        for role in ("baseline", "improved"):
            v = r["point_metrics"][role][m]
            ci = r["confidence_intervals"][role].get(m)
            row += f" {v:.4f}" + (f" [{ci['ci'][0]:.4f}, {ci['ci'][1]:.4f}]" if ci else "") + " |"
        L.append(row)
    for role in ("baseline", "improved"):
        w = r["confidence_intervals"][role]["accuracy"]["wilson_ci"]
        L.append(f"\nWilson {pct}% CI for {role} accuracy: [{w[0]:.4f}, {w[1]:.4f}]")
    L.append(f"\n## Paired difference (improved − baseline)\n")
    L.append(f"| Metric | Δ | {pct}% CI | Bootstrap SE | z (normal approx.) | p (normal) | p (bootstrap) | P(Δ>0) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m, d in r["paired_difference"].items():
        L.append(f"| {m} | {d['observed_delta']:+.4f} | [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] | "
                 f"{d['bootstrap_se']:.4f} | {d['z']:.2f} | {d['p_value_normal_approx']:.4g} | "
                 f"{d['p_value_bootstrap']:.4g} | {d['P(delta>0)']:.3f} |")
    mc, pt = r["mcnemar"], r["permutation_test"]
    L.append("\n## Hypothesis tests\n")
    L.append(f"McNemar's test (H0: equal error rates): b (only baseline correct) = {mc['b_baseline_only_correct']:,}, "
             f"c (only improved correct) = {mc['c_improved_only_correct']:,}, χ² = {mc['chi2']:.3f}, "
             f"p = {mc['p_value']:.4g} ({mc['method']}).\n")
    L.append(f"Paired permutation test on F1 (H0: ΔF1 = 0): observed ΔF1 = {pt['observed_delta_f1']:+.4f}, "
             f"null SD = {pt['null_sd']:.4f}, p = {pt['p_value']:.4g}.\n")
    L.append("## Sampling-variability experiment\n")
    L.append("| n | mean ΔF1 | SD ΔF1 | P(improved wins) | P(baseline wins) | Power (McNemar) |\n|---|---|---|---|---|---|")
    for s in r["sampling_experiment"]:
        L.append(f"| {s['n']:,} | {s['mean_delta_f1']:+.4f} | {s['sd_delta_f1']:.4f} | {s['P(improved_wins)']:.3f} | "
                 f"{s['P(baseline_wins)']:.3f} | {s['power_mcnemar']:.3f} |")
    L.append(f"\nSmallest test size with ≥80% power: {r['n_for_80pct_power']}\n")
    L.append(f"## Decision\n\n{r['decision']}\n")
    open(path, "w", encoding="utf-8").write("\n".join(L))


# ----------------------------------------------------------------------------- #
def main():
    args = parse_args()
    out = args.out_dir
    data_dir, model_root, ckpt_root = f"{out}/data", f"{out}/models", f"{out}/checkpoints"
    for d in (out, data_dir, model_root, ckpt_root):
        os.makedirs(d, exist_ok=True)
    set_all_seeds(args.seed)
    names = (args.baseline.split("/")[-1], args.improved.split("/")[-1])
    pred_path, thr_path = f"{out}/predictions_test.csv", f"{out}/thresholds.json"

    data, summary = prepare_data(args, data_dir)
    print(f"[data] train {len(data['train']):,} | val {len(data['val']):,} | test {len(data['test']):,}")

    if args.skip_training and os.path.exists(pred_path):
        print("[main] --skip_training: loading saved predictions")
        pred = pd.read_csv(pred_path)
        thr = json.load(open(thr_path))
    else:
        val_y = data["val"]["label"].values
        pred = data["test"][["orig_idx", "label"]].copy()
        val_pred = data["val"][["orig_idx", "label"]].copy()
        thr = {}
        for role, name in (("baseline", args.baseline), ("improved", args.improved)):
            set_all_seeds(args.seed)   # identical randomness for both models
            model_dir = train_or_load(name, data["train"], args, model_root, ckpt_root)
            val_pred[f"p_{role}"] = predict_proba(model_dir, data["val"]["text"].tolist(), args)
            pred[f"p_{role}"] = predict_proba(model_dir, data["test"]["text"].tolist(), args)
            thr[role] = tune_threshold(val_y, val_pred[f"p_{role}"].values)
            print(f"[eval] {role} threshold = {thr[role]}")
        pred.to_csv(pred_path, index=False)
        val_pred.to_csv(f"{out}/predictions_val.csv", index=False)
        json.dump(thr, open(thr_path, "w"), indent=2)

    run_statistics(pred["label"].values, pred["p_baseline"].values, pred["p_improved"].values,
                   thr["baseline"], thr["improved"], names, args, out, summary)


if __name__ == "__main__":
    main()
