# PRELIMINARY — test ground truth is auto-generated pseudo-masks, not human-verified

## Test results (227 images, split=test)

| Metric | multitask | baseline |
|---|---|---|
| Grade accuracy | 78.4% [73.1, 83.3] | 78.0% [72.2, 82.8] |
| Grade macro-F1 | 0.585 [0.515, 0.660] | 0.573 [0.504, 0.648] |
| Mean defect IoU | 0.250 [0.228, 0.270] | 0.124 [0.113, 0.135] |
| Mean defect Dice | 0.340 [0.315, 0.364] | 0.181 [0.169, 0.192] |
| Any-defect IoU | 0.593 [0.549, 0.636] | 0.372 [0.339, 0.404] |
| Any-defect Dice | 0.745 [0.709, 0.778] | 0.543 [0.507, 0.575] |
| Severity MAE (mask, % pts) | 9.40 [7.73, 11.09] | 10.22 [8.74, 11.72] |
| Severity MAE (regression head, % pts) | 6.60 [5.70, 7.56] | — |
| Flagged for manual review | 18.9% | 25.6% |
| Wrong grades caught by flag | 49.0% | 60.0% |

Values in brackets are 95% bootstrap confidence intervals (images resampled with replacement).
Baseline IoU/Dice/MAE use its Grad-CAM ("Reject" class, >0.5) inside the fruit as a proxy defect mask, and only as a single 'defect' class (counted as rot) — so per-type IoU is not meaningful for it.
