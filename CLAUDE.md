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
- **When unsure, merge.** A wrong merge costs nothing; a missed one risks leakage.
- A cross-class pair that shows the same photo or fruit is a label conflict: mark it `conflict`, which drops both images and logs them as `label_conflict` in the audit. (None found: all 12 cross-class pairs were re-checked enlarged.)
- Tracked copies of the decisions, audit and splits live in `results/data_audit/`.
- Split 70/15/15 by group, stratified by class (fresh/rotten x fruit), seed 42. No group spans two splits.
- (The first attempt, pHash with union-find over all 13,599 files, chained unrelated rotated images into a 1,053-image group. Don't use it.)

### Pseudo-masks (Phase 2, `fruitgrade/data/pseudo_masks.py`, approved state = commit 1fe3361)
- Tuning changes kept: mechanical = small and thin only; long thin blobs hugging the outline dropped (contact shadow); green leaves/stems removed unless the fruit is green; stricter darkness threshold for fresh-labelled images (x0.8).
- Tried and reverted (measured worse): chroma/colour checks, skipping large bright-texture blobs, a texture cue for white mould. Keep the mould cue out.
- Known limits: white mould is often missed (rotten oranges graded A), deep-red apple skin can read as rot, several fruit in one picture keeps only the largest.
- Check any change with `python -m fruitgrade.data.pm_eval` (fresh/rotten agreement 197/240, synthetic defect IoU 0.594, severity MAE 3.58 at approval) and the before/after sheets from `fruitgrade.data.contact_sheet`.
- The 240-image tuning sample and the 144-image sheet sample were drawn from all splits (tuning: 167 train / 33 val / 40 test). Recorded in the audit; not fixed. Say so in the report.
- **Note for the report (weak supervision):** training labels use the image-level Kaggle fresh/rotten label as weak supervision (the x0.8 prior and the consistency filter). Inference never sees that label. Test ground truth comes from CVAT, not from pseudo-masks.

### Label-consistency filter (`fruitgrade/data/label_filter.py`)
- Train and val only; test is never touched. Fresh label + pseudo grade Reject, or rotten label + pseudo grade A, gets `excluded_reason` in labels.csv (flagged, not deleted). `FruitDataset` skips flagged rows.
- At approval: 71 rows excluded (62 train, 9 val). Counts are in the audit under `label_consistency_filter`.

### Evaluation
- `evaluate.py` reports 95% bootstrap CIs (1,000 resamples, by image) as "value [low, high]" for accuracy, macro-F1, defect IoU/Dice and severity MAE.

### Before Phase 4
- Add AMP (autocast + GradScaler) and a `--grad-accum` option to `train.py`, with tests.
- Write `notebooks/train_kaggle.ipynb`.

## Grading rules: LOCKED
- Grading rules locked at commit 0e11134, 2026-10-19 (lock recorded 2026-10-20), before any real test evaluation.
- Changing `grading_rules` in config.yaml needs the user's **explicit approval**. `test_grading_rules_locked` fails on any change; update it only together with an approved change.

## Git
- Remote: `origin` = https://github.com/Jewelsony2023/fruit-grader (branch `main`). Push after each batch of commits.

## Rules
- Never invent numbers or round them up. Report every seed, not only the best one.
- Ask before changing the grading rules (locked, see above), deleting data, or starting any run longer than 2 h.
- Explain each step in simple terms.
- Make small, focused commits.
