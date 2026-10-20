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
- Dataset: `sriramr/fruits-fresh-and-rotten-for-classification`, extracted to `D:\datasets\fruits\dataset`.
- The archive holds a duplicate nested copy (`dataset/dataset/...`), byte-identical (same 13,599 files, same CRCs). Only the top tree is extracted.

## Dataset facts (measured, in `data/interim/dataset_audit.json`)
- 13,599 files = **1,511 originals x 9 versions**: original, rotated 15/30/45/60/75 deg, salt-and-pepper, translation, vertical flip.
- Kaggle's train/test split leaks: for **1,310 of 1,511 originals**, versions sit in both Kaggle-train and Kaggle-test. Kaggle's split is never used.
- **We use the 1,511 originals only.** All Kaggle augmented copies are dropped from every split, train included (noise and black rotation corners corrupt the pseudo-masks; `dataset.py` augments properly).
- Report the dataset as about 1,500 photos, not 13.6k.

## Phases (each one ends at a gate: stop and show the user before moving on)
1. **Data + dedupe.** Download, keep one tree, group near-duplicates, split by group (see below).
2. **Tune pseudo-masks.** The user approves the contact sheets and the grade split.
3. **CVAT verified test set.** Covers **ALL test originals** (227 images in the current split), pre-annotated with pseudo-masks, then corrected by hand.
4. **Training** in a Kaggle notebook on a GPU, not the laptop.
5. **Evaluation.** The verified subset gives the headline numbers.
6. **App** running on the real models.
7. **`results/RESULTS.md`**, with every number traced to the file it came from.

### Dedupe and split method (Phase 1, `fruitgrade/data/dedupe.py`)
- Group by original photo: strip the augmentation prefix from the file name. All 9 versions form one group, and only the original is kept.
- pHash (minimum over the 8 rotated/flipped versions) on the 1,511 originals only. List cross-original pairs at Hamming distance <= 6 as pairs, never chained.
- Make a contact sheet of those pairs. Merge **only** pairs that are visibly the same fruit or photo, recorded in `data/interim/merge_decisions.csv`. Never merge on the threshold alone.
- Split 70/15/15 by group, stratified by class (fresh/rotten x fruit), seed 42. No group spans two splits.
- (The first attempt, pHash with union-find over all 13,599 files, chained unrelated rotated images into a 1,053-image group. Don't use it.)

### Evaluation
- `evaluate.py` reports 95% bootstrap CIs (1,000 resamples, by image) as "value [low, high]" for accuracy, macro-F1, defect IoU/Dice and severity MAE.

### Before Phase 4
- Add AMP (autocast + GradScaler) and a `--grad-accum` option to `train.py`, with tests.
- Write `notebooks/train_kaggle.ipynb`.

## Rules
- Never invent numbers or round them up. Report every seed, not only the best one.
- Ask before changing the grading rules, deleting data, or starting any run longer than 2 h.
- Explain each step in simple terms.
- Make small, focused commits.
