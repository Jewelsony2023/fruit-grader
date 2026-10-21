# PRELIMINARY — test ground truth = auto-generated pseudo-masks, not human-verified

## Test results (227 images, split=test, MC Dropout 20 passes, seed 0)

| Metric | multitask | baseline |
|---|---|---|
| Grade accuracy | 79.7% [74.4, 84.6] | 78.4% [73.1, 83.3] |
| Grade macro-F1 | 0.599 [0.533, 0.677] | 0.602 [0.519, 0.680] |
| Mean defect IoU | 0.250 [0.228, 0.270] | 0.124 [0.113, 0.135] |
| Mean defect Dice | 0.340 [0.315, 0.364] | 0.181 [0.169, 0.192] |
| Any-defect IoU | 0.593 [0.549, 0.636] | 0.372 [0.339, 0.404] |
| Any-defect Dice | 0.745 [0.709, 0.778] | 0.543 [0.507, 0.575] |
| Severity MAE (mask, % pts) | 9.40 [7.73, 11.09] | 10.22 [8.74, 11.72] |
| Severity MAE (regression head, % pts) | 6.67 [5.72, 7.65] | — |
| Flagged for manual review | 17.6% | 26.4% |
| Wrong grades caught by flag | 45.7% | 63.3% |
| Flagged, all app reasons (+ rule mismatch) | 26.0% | 26.4% |
| Wrong grades caught, all app reasons | 69.6% | 63.3% |

Values in brackets are 95% bootstrap confidence intervals (images resampled with replacement).
Baseline IoU/Dice/MAE use its Grad-CAM ("Reject" class, >0.5) inside the fruit as a proxy defect mask, and only as a single 'defect' class (counted as rot) — so per-type IoU is not meaningful for it.

## Paired bootstrap: multitask - baseline (1000 resamples, same images for both models)

| Metric | Difference | 95% CI | CI excludes 0? |
|---|---|---|---|
| Grade accuracy | +0.0132 | [-0.0220, +0.0441] | no |
| Grade macro-F1 | -0.0027 | [-0.0708, +0.0603] | no |
| Any-defect Dice | +0.2022 | [+0.1692, +0.2379] | yes |
| Severity MAE (mask, % pts) | -0.8213 | [-2.3228, +0.6493] | no |
| Severity MAE (multitask head vs baseline mask, % pts) | -3.5572 | [-4.9122, -2.2615] | yes |

For MAE a negative difference means the multitask model is better.
