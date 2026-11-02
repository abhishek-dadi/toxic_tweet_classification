# Results: bert-base-uncased (baseline) vs twitter-roberta-base-2022-154m (improved)

Test set: n = 20,000, toxic = 1,598 (prevalence 7.990%). Thresholds tuned on validation: baseline 0.84, improved 0.86.

## Sampling check (one-sample z-test for prevalence)

| Sample | n | Sample prevalence | Population prevalence | z | p |
|---|---|---|---|---|---|
| val | 15,000 | 0.0782 | 0.0788 | -0.283 | 0.777 |
| test | 20,000 | 0.0799 | 0.0799 | -0.006 | 0.995 |

## Point estimates and 95% bootstrap confidence intervals

| Metric | Baseline | Improved |
|---|---|---|
| accuracy | 0.9502 [0.9471, 0.9532] | 0.9527 [0.9497, 0.9556] |
| precision | 0.6930 | 0.7217 |
| recall | 0.6752 | 0.6652 |
| f1 | 0.6840 [0.6649, 0.7022] | 0.6923 [0.6728, 0.7105] |
| roc_auc | 0.9617 [0.9579, 0.9653] | 0.9650 [0.9615, 0.9684] |

Wilson 95% CI for baseline accuracy: [0.9470, 0.9531]

Wilson 95% CI for improved accuracy: [0.9497, 0.9556]

## Paired difference (improved − baseline)

| Metric | Δ | 95% CI | Bootstrap SE | z (normal approx.) | p (normal) | p (bootstrap) | P(Δ>0) |
|---|---|---|---|---|---|---|---|
| f1 | +0.0083 | [-0.0043, +0.0210] | 0.0064 | 1.30 | 0.1936 | 0.195 | 0.902 |
| accuracy | +0.0026 | [+0.0006, +0.0046] | 0.0010 | 2.51 | 0.01207 | 0.0152 | 0.992 |
| roc_auc | +0.0032 | [+0.0011, +0.0055] | 0.0011 | 2.88 | 0.00397 | 0.004 | 0.998 |

## Hypothesis tests

McNemar's test (H0: equal error rates): b (only baseline correct) = 193, c (only improved correct) = 245, χ² = 5.938, p = 0.01481 (chi-square (continuity corrected)).

Paired permutation test on F1 (H0: ΔF1 = 0): observed ΔF1 = +0.0083, null SD = 0.0064, p = 0.1916.

## Sampling-variability experiment

| n | mean ΔF1 | SD ΔF1 | P(improved wins) | P(baseline wins) | Power (McNemar) |
|---|---|---|---|---|---|
| 100 | +0.0144 | 0.0983 | 0.462 | 0.334 | 0.002 |
| 250 | +0.0050 | 0.0608 | 0.504 | 0.422 | 0.014 |
| 500 | +0.0075 | 0.0413 | 0.560 | 0.406 | 0.035 |
| 1,000 | +0.0080 | 0.0289 | 0.613 | 0.377 | 0.048 |
| 2,500 | +0.0084 | 0.0186 | 0.672 | 0.326 | 0.116 |
| 5,000 | +0.0089 | 0.0129 | 0.751 | 0.249 | 0.211 |
| 10,000 | +0.0088 | 0.0091 | 0.837 | 0.163 | 0.407 |
| 20,000 | +0.0079 | 0.0063 | 0.905 | 0.095 | 0.663 |

Smallest test size with ≥80% power: None

## Decision

Fail to reject H0 at alpha = 0.05. The observed ΔF1 = +0.0083 has a 95% CI [-0.0043, +0.0210] that includes 0 (permutation p = 0.1916), so the difference could plausibly be due to variation in the test sample.
