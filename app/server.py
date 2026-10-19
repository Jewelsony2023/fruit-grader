"""
Web app backend (FastAPI).

    python -m app.server                 # http://127.0.0.1:8000
    python -m app.server --port 8080 --multitask runs/.../best.pt --baseline runs/.../best.pt

Endpoints
    GET  /                 the web page (app/static/index.html)
    GET  /api/info         loaded models, grade names, mask colours, grading rules
    POST /api/predict      form-data: file=<image>, model=multitask|baseline
    GET  /api/samples      a few test images to try with one click
    GET  /api/report       latest comparison table (reports/comparison.md + metrics.json)
"""
import argparse
import base64
import json
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from fruitgrade.config import PROJECT_ROOT, load_config
from fruitgrade.data.dataset import read_labels
from fruitgrade.grading import MASK_COLOURS
from fruitgrade.inference import Predictor

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 15 * 1024 * 1024

app = FastAPI(title="Fruit Quality Grader")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
PREDICTORS: dict = {}
CFG = load_config()


def _latest(pattern):
    files = sorted(glob(str(PROJECT_ROOT / pattern)), key=os.path.getmtime)
    return files[-1] if files else None


def load_models(multitask=None, baseline=None):
    multitask = multitask or os.environ.get("FRUIT_MULTITASK_CKPT") or _latest("runs/multitask_*/best.pt")
    baseline = baseline or os.environ.get("FRUIT_BASELINE_CKPT") or _latest("runs/baseline_*/best.pt")
    for name, path in (("multitask", multitask), ("baseline", baseline)):
        if path and Path(path).exists():
            PREDICTORS[name] = Predictor(path)
            print(f"[server] loaded {name}: {path}")
    if not PREDICTORS:
        print("[server] WARNING: no checkpoints found in runs/. Train a model first (see README).")


def _b64png(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {
        "models": {k: {"backbone": p.meta["backbone"], "epoch": p.meta["epoch"]} for k, p in PREDICTORS.items()},
        "grades": CFG["grades"],
        "seg_classes": CFG["seg_classes"],
        "colours": {CFG["seg_classes"][k]: "#%02x%02x%02x" % v for k, v in MASK_COLOURS.items()},
        "rules": CFG["grading_rules"],
        "review": CFG["inference"],
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model: str = Form("multitask")):
    if model not in PREDICTORS:
        raise HTTPException(400, f"Model '{model}' not loaded. Available: {list(PREDICTORS)}")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image too large (max 15 MB)")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Could not read that file as an image (use JPG/PNG/BMP/WEBP).")
    r = PREDICTORS[model].predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    r["images"] = {k: _b64png(v) for k, v in r["images"].items()}
    r.pop("_mask", None)
    r["filename"] = file.filename
    return JSONResponse(r)


@app.get("/api/samples")
def samples(n: int = 8):
    try:
        rows = read_labels(CFG["data"]["root"], "test")
    except FileNotFoundError:
        return []
    # one or two per grade so the user sees the full range
    picked, per = [], {}
    for r in rows:
        if per.get(r["grade"], 0) < max(1, n // len(CFG["grades"])):
            per[r["grade"]] = per.get(r["grade"], 0) + 1
            picked.append({"url": f"/api/sample/{Path(r['image']).name}", "grade": r["grade"],
                           "severity": float(r["severity"]), "source": r["source"]})
    return picked[:n]


@app.get("/api/sample/{name}")
def sample(name: str):
    p = (Path(CFG["data"]["root"]) / "images" / Path(name).name).resolve()
    if not p.exists() or Path(CFG["data"]["root"]).resolve() not in p.parents:
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/report")
def report():
    rep = PROJECT_ROOT / "reports"
    out = {}
    if (rep / "metrics.json").exists():
        out["metrics"] = json.loads((rep / "metrics.json").read_text())
    if (rep / "comparison.md").exists():
        out["markdown"] = (rep / "comparison.md").read_text()
    return out


@app.get("/api/report-image/{name}")
def report_image(name: str):
    p = PROJECT_ROOT / "reports" / Path(name).name
    if p.suffix != ".png" or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--multitask", default=None)
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()
    load_models(args.multitask, args.baseline)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
