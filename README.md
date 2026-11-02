# Applying Probability and Statistics to Evaluate a Machine Learning Model

**Question:** Does a domain-adapted model (Twitter-RoBERTa) really classify toxic comments better than a
baseline (BERT-base), or could the observed difference be due to variation in the test sample?

| | Model | Notes |
|---|---|---|
| Baseline | `bert-base-uncased` | General-purpose BERT, fine-tuned |
| Improved | `cardiffnlp/twitter-roberta-base-2022-154m` | RoBERTa pretrained on 154M tweets, fine-tuned |

Both models use identical training data, hyper-parameters, seed and the **same held-out test set** (paired design).

## Dataset
[Civil Comments](https://huggingface.co/datasets/google/civil_comments) (Jigsaw / Google, CC0).
A comment is labelled **toxic** if its toxicity score (fraction of annotators who rated it toxic) is ≥ 0.5.

## Statistical methods
- **Sampling:** stratified sample for training (25% toxic), simple random sample for validation,
  proportionate stratified random sample for the test set; one-sample z-test confirms sample prevalence matches the population.
- **Estimation:** Wilson score CI for accuracy; bootstrap CIs (10,000 resamples) for F1, accuracy and ROC-AUC.
- **Paired difference:** paired bootstrap CI for ΔF1, ΔAccuracy, ΔAUC; normal-approximation z-test (CLT).
- **Hypothesis tests (α = 0.05):** McNemar's test (error rates) and paired permutation test (F1).
- **Sampling-variability experiment:** repeated test samples of size 100 → N to show how often the
  conclusion would change, how the standard error shrinks like 1/√n, and the empirical power.

## How to run (Google Colab, GPU runtime)
```python
from google.colab import drive
drive.mount('/content/drive')
!git clone <YOUR_REPO_URL>
%cd <REPO_FOLDER>
!python toxicity_stats.py --out_dir /content/drive/MyDrive/aps_toxicity
```
Runtime on a T4 GPU: about 50–60 minutes (two models × ~25 min training + ~5 min statistics).
Colab already includes all required packages. Locally: `pip install -r requirements.txt`.

Re-run only the statistics (no GPU, uses saved predictions):
```
python toxicity_stats.py --out_dir /content/drive/MyDrive/aps_toxicity --skip_training
```
Training resumes automatically from the last checkpoint if Colab disconnects.

## Outputs (in `--out_dir`)
| File | Content |
|---|---|
| `results_summary.md` | All tables + final decision |
| `results.json` | Every number in machine-readable form |
| `figures/` | ROC curves, confusion matrices, CIs, bootstrap and permutation distributions, sampling experiment |
| `data/` | The exact sampled train/val/test sets (for reproducibility) |
| `predictions_test.csv` | Per-example probabilities from both models |
| `models/` | Fine-tuned weights |

## Reproducibility
Fixed seed (42) for sampling, training and resampling; sampled datasets and predictions are saved to disk.
