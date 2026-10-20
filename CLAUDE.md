# CLAUDE.md: fruit-grader

## Project
VIT capstone: one multi-task model for fruit grading (grade A/B/C/Reject, defect segmentation, severity %, MC-Dropout confidence, Grad-CAM), compared against a classification-only baseline. See README.md for the full design and commands.

**The demo numbers in `runs/` and `reports/` come from SYNTHETIC data. Never report or quote them as results.**

## Machine
- Windows 11, NVIDIA MX350 (2 GB VRAM), `.venv` with torch 2.14.1 + CUDA 12.6.
- If the DataLoader hangs, set `data.num_workers: 0` in config.yaml.
- Do not train for real on this laptop. Real training runs in a Kaggle notebook on a GPU (Phase 4).

## Kaggle
- Auth: new-style token file at `C:\Users\HP\.kaggle\access_token`. **Never print, cat, echo or log its contents.**
- Dataset: `sriramr/fruits-fresh-and-rotten-for-classification`.
- The archive holds a duplicate nested copy (`dataset/dataset/...`). Use only one tree.

## Phases (each one ends at a gate: stop and show the user before moving on)
1. **Data + dedupe.** Download, keep one tree, group near-duplicates, split by group (see below).
2. **Tune pseudo-masks.** The user approves the contact sheets and the grade split.
3. **CVAT verified test set.** About 150 images, pre-annotated with pseudo-masks, then corrected by hand.
4. **Training** in a Kaggle notebook on a GPU, not the laptop.
5. **Evaluation.** The verified subset gives the headline numbers.
6. **App** running on the real models.
7. **`results/RESULTS.md`**, with every number traced to the file it came from.

### Before data prep (Phase 1)
- Perceptual hash that ignores rotation and flips: take the minimum hash over the 8 rotated/flipped versions of each image.
- Group near-duplicates at Hamming distance <= 6.
- Split train/val/test **by group**, so no group spans two splits.
- Use Kaggle's own test folder only if none of its groups overlap train.

### Before Phase 4
- Add AMP (autocast + GradScaler) and a `--grad-accum` option to `train.py`, with tests.
- Write `notebooks/train_kaggle.ipynb`.

## Rules
- Never invent numbers or round them up. Report every seed, not only the best one.
- Ask before changing the grading rules, deleting data, or starting any run longer than 2 h.
- Explain each step in simple terms.
- Make small, focused commits.
