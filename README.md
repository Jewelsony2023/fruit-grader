# Automated Fruit Quality Grading & Surface Defect Quantification

One deep-learning model that looks at a fruit photo and returns:

| Output | How |
|---|---|
| **Grade** A / B / C / Reject | classification head |
| **Defect mask**: bruise, rot, mechanical damage (pixel level) | U-Net segmentation decoder |
| **Severity** = % of fruit surface damaged | from the mask + a regression head (0–100) |
| **Confidence** + "needs manual review" flag | Monte-Carlo Dropout (20 passes) |
| **Grad-CAM** heat-map | where the model looked |

A **classification-only baseline** (same backbone, grade head only) is trained the same way, and `evaluate.py` compares the two on **accuracy, macro-F1, IoU, Dice and severity MAE**.

### What's already in this folder
* `runs/` holds **demo checkpoints**. They were trained only on synthetic data, 8 epochs, 192 px, with no ImageNet weights (CPU-only machine), so they are for trying the app, not for the report.
* `reports/` holds the demo comparison (synthetic test set, 261 images):

| Metric | Multi-task | Baseline |
|---|---|---|
| Grade accuracy | 83.9% | 79.3% |
| Mean defect IoU / Dice | 0.704 / 0.823 | 0.095 / 0.148 (Grad-CAM proxy) |
| Severity MAE | 2.6 % pts | 12.7 % pts |

To open the app right away: `python -m fruitgrade.data.synthetic --n 1600` (rebuilds the same test images, seed 0), then `python -m app.server`.

---

## 1. Folder layout

```
fruit-grader/
├── config.yaml                 <- every setting (backbone, grade rules, loss weights ...)
├── fruitgrade/
│   ├── grading.py              severity % and grade rules
│   ├── models.py               MultiTaskFruitNet + BaselineClassifier
│   ├── losses.py               CE + (CE+Dice) + SmoothL1
│   ├── metrics.py              accuracy/F1, IoU/Dice, MAE
│   ├── train.py                training loop
│   ├── evaluate.py             model vs baseline report
│   ├── explain.py              Grad-CAM + MC Dropout
│   ├── inference.py            Predictor (used by the web app)
│   └── data/
│       ├── synthetic.py        synthetic fruit with exact masks (smoke test / demo)
│       ├── pseudo_masks.py     colour threshold + GrabCut (+ Grad-CAM) pseudo-masks
│       ├── prepare_kaggle.py   fresh/rotten folders -> images/ masks/ labels.csv
│       ├── coco_to_masks.py    COCO polygons (Roboflow / CVAT) -> verified masks
│       └── dataset.py          PyTorch Dataset + augmentations
├── app/
│   ├── server.py               FastAPI backend
│   └── static/                 index.html, style.css, app.js
├── tests/test_pipeline.py      pytest checks
└── runs/ reports/ data/        created when you run things
```

---

## 2. Setup (Windows 11, step by step)

```powershell
cd fruit-grader
python -m venv .venv
.venv\Scripts\activate

# 1) PyTorch — pick ONE line (GPU if you have an NVIDIA card)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) everything else
pip install -r requirements.txt

# 3) check that all parts work (~30 s)
pytest -q
```

> On Windows, if training freezes at the first batch, set `num_workers: 0` in `config.yaml`.
> Google Colab / Kaggle notebooks also work: upload the folder, `pip install timm`, run the same commands.

---

## 3. Quick demo in 10 minutes (synthetic data)

Checks the whole pipeline end-to-end before you touch the real data.

```powershell
python -m fruitgrade.data.synthetic --n 1600          # images + exact masks -> data/processed
python -m fruitgrade.train --model multitask --epochs 15
python -m fruitgrade.train --model baseline  --epochs 15
python -m fruitgrade.evaluate --multitask runs/multitask_efficientnet_b0/best.pt --baseline runs/baseline_efficientnet_b0/best.pt
python -m app.server                                   # open http://127.0.0.1:8000
```

**Do not put synthetic numbers in the report.** They only prove the code works.

---

## 4. Real data (what goes in the report)

### Step 4.1 — Download
Kaggle *"Fruits fresh and rotten for classification"* (apple / banana / orange, ~13.6k images). Unzip anywhere, e.g. `D:\datasets\fruits\dataset\train\freshapples\...`

### Step 4.2 — Make pseudo-masks and grades
The Kaggle data only says *fresh* or *rotten*. It has **no masks and no A/B/C grades**, so this script makes them:

```powershell
# start fresh: delete data/processed if it holds the synthetic set
python -m fruitgrade.data.prepare_kaggle --src D:\datasets\fruits\dataset --out data/processed
```

What it does for each image:
1. **Fruit vs background:** GrabCut, seeded by saturation + "different from the border colour".
2. **Shading correction:** a big grey *closing* estimates how bright each healthy pixel *should* be (fruit edges are naturally darker). A pixel is a defect candidate if it is < 72 % of that.
3. **Pale thin lines** (top-hat filter) = cuts/scratches. **Whitish low-colour blobs** = mould.
4. **Type each blob:** long/thin → *mechanical*; very dark, mouldy or big & blotchy → *rot*; otherwise → *bruise*.
5. **Severity** = defect pixels ÷ fruit pixels, then **grade** from the rules in `config.yaml`.

Check by eye: open `data/processed/preview/` (input | overlay). If a lot are wrong, tune `dark_ratio` in `pseudo_masks.py`.

Measured on synthetic images with exact ground truth: pseudo-mask **defect IoU 0.64, severity MAE 3.4 % points, fruit IoU 0.99**. Real photos will be worse — that is why step 4.4 matters.

### Step 4.3 — Train
```powershell
python -m fruitgrade.train --model baseline  --backbone efficientnet_b0
python -m fruitgrade.train --model multitask --backbone efficientnet_b0
# optional extra rows for the report:
python -m fruitgrade.train --model multitask --backbone resnet50
python -m fruitgrade.train --model multitask --backbone mobilenetv3_large_100
```
Optional second round, Grad-CAM refinement (as on the methodology slide): rebuild masks keeping only defect blobs the trained classifier actually looks at, then retrain.
```powershell
python -m fruitgrade.data.prepare_kaggle --src D:\datasets\fruits\dataset --out data/processed_r2 --cam-checkpoint runs/baseline_efficientnet_b0/best.pt
```
(then set `data.root: data/processed_r2` in config.yaml and train again)

### Step 4.4 — Hand-verified test set (important)
Pseudo-masks are approximate, so testing against them is weak. Take ~100–150 test images, fix their masks in **CVAT** (free) with labels `fruit`, `bruise`, `rot`, `cut`, export **COCO 1.0**, then:
```powershell
python -m fruitgrade.data.coco_to_masks --json annotations.json --images cvat_images/ --map "fruit=healthy,bruise=bruise,rot=rot,cut=mechanical"
python -m fruitgrade.evaluate --multitask runs/multitask_efficientnet_b0/best.pt --baseline runs/baseline_efficientnet_b0/best.pt --mask-quality verified
```
The same converter works for the 131-image Roboflow/HF segmentation set on your dataset slide (change `--map` to its category names).

### Step 4.5 — Report files
`reports/comparison.md` (table), `reports/metrics.json`, `reports/confusion_*.png`, `reports/samples_*.png`.

---

## 5. Web app

```powershell
python -m app.server
# or choose checkpoints explicitly
python -m app.server --multitask runs/multitask_resnet50/best.pt --baseline runs/baseline_efficientnet_b0/best.pt
```
Open **http://127.0.0.1:8000**:
* drag an image in (or click a test image), choose **Multi-task** or **Baseline**, press **Grade this fruit**
* see grade, confidence, severity gauge, defect breakdown, 4 images (input / defect overlay / mask / Grad-CAM — click to enlarge), grade probabilities and MC-Dropout uncertainty
* a yellow box explains **why** a fruit needs manual review
* **Model performance** tab shows the comparison table; **How it works** explains the pipeline and the grade rules
* **Download result (JSON)** saves the numbers

CLI alternative: `python -m fruitgrade.inference --ckpt runs/.../best.pt --image apple.jpg --save out.png`

### How to present (results dashboard, works offline)

The dashboard needs no internet: no web fonts, no CDN, Chart.js 4.4.1 is in `app/static/vendor/`.
Every number on it is read from `results/preliminary/` (made by `fruitgrade.predict_split`), the run folders and `results/data_audit/`.

**One command** (from the project folder):
```powershell
.venv\Scripts\python.exe -m app.server --multitask runs/prelim_multitask_efficientnet_b0/best.pt --baseline runs/prelim_baseline_efficientnet_b0/best.pt
```
Open **http://127.0.0.1:8000/dashboard**. Press F11 for full screen. Keys: ← / → change tab, 1–9 jump to a tab.

The orange banner (*PRELIMINARY — test ground truth = auto-generated pseudo-masks*) stays on every tab. Say it out loud once.

Click through in this order:
1. **Overview**: the four headline numbers with 95% CIs, then the paired-bootstrap table. Call a difference real only where the verdict says the CI excludes 0.
2. **Confusion**: click an off-diagonal cell (for example true C → predicted Reject) to show those images.
3. **Per grade**: point at the support column; the rarest grade has the weakest F1.
4. **Segmentation**: rot and any-defect; explain why the baseline scores 0 for bruise and mechanical (it has no mask).
5. **Severity**: scatter against the y = x line, then MAE by grade.
6. **Confidence**: reliability diagram, then the review-flag table (how many wrong grades the flag catches).
7. **Training**: curves with the best epoch marked.
8. **Data**: files = originals × versions, the Kaggle split leak, our group split.
9. **Errors**: the most confident mistakes. Check which ones are pseudo-mask misses (white mould): that is why the CVAT-verified test set comes next.
10. **Live demo →**: the Analyse page; click a test image and press *Grade this fruit*.

To rebuild the data (MC Dropout 20 passes, seed 0, so a re-run gives the same files):
```powershell
.venv\Scripts\python.exe -m fruitgrade.predict_split --title "PRELIMINARY — test ground truth = auto-generated pseudo-masks, not human-verified"
.venv\Scripts\python.exe scripts\dashboard_screenshots.py      # screenshots of every tab -> results/review/dashboard/
```

---

## 6. Why it is built this way (viva notes)

| Choice | Reason |
|---|---|
| **One shared encoder, three heads** | The grade depends on the damage; learning the mask forces the encoder to find the defects, which should help grading too. One forward pass instead of three models. |
| **U-Net decoder with skip connections** | Defects are small; skips carry high-resolution detail from early layers so edges stay sharp. |
| **CE + Dice for segmentation** | Defect pixels are a tiny fraction of the image; plain CE would get high accuracy by predicting "healthy" everywhere. Dice cares about overlap per class. |
| **Severity from the mask *and* a regression head** | The mask-based number is checkable by eye (that's the "quantification"). The head is a second opinion; disagreement is a warning sign. |
| **Class-weighted CE for grades** | Real data has many more A and Reject than B and C. |
| **Backbone LR × 0.1 + 2 frozen epochs** | Keeps pretrained ImageNet features from being wrecked by random-init heads early on. |
| **Grades from rules on the mask** | The public datasets have no grade labels. Rules are transparent and editable in `config.yaml`. The UI also flags when the classifier disagrees with the rule. |
| **MC Dropout** | Cheap uncertainty without training extra models. The encoder runs once; only the small heads repeat 20×. |
| **Baseline IoU via Grad-CAM** | Shows the deck's point: a classifier's heat-map is a coarse guess at "where", not a measured defect area. |

## 7. Honest limitations (say these before the panel does)
* **Grade labels are derived, not human-assigned.** Both models learn the rule from pseudo-masks. For a stronger claim, have 200–300 fruits graded by a person and test on those.
* **Pseudo-masks are noisy:** stems, stickers, dark natural spots and strong shadows can be mistaken for defects; bruise vs rot is decided by simple rules.
* **Single view:** only the visible side is graded (multi-view fusion = future work).
* **Synthetic data** is for testing the code only.
* Dense CRF was left out (the `pydensecrf` package is hard to install on Windows); GrabCut + morphology does the refinement.

## 8. Docker (optional, runs the web app on CPU)
```powershell
docker build -t fruit-grader .
docker run -p 8000:8000 -v ${PWD}/runs:/app/runs -v ${PWD}/data:/app/data -v ${PWD}/reports:/app/reports fruit-grader
```
