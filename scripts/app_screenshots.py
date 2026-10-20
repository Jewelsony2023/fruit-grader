"""
Screenshots of the web app's 3 tabs (Analyse, Model performance, How it works)
for 2 real test images: one fresh-labelled and one rotten-labelled photo, picked
with a fixed seed. Starts the app itself on the given checkpoints and stops it.

Needs: pip install playwright (uses the installed Google Chrome, no browser download).

    python scripts/app_screenshots.py --multitask runs/.../best.pt --baseline runs/.../best.pt \
        --report-dir results/preliminary --out results/review/app
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent


def pick_images(seed):
    rows = [r for r in csv.DictReader(open(ROOT / "data/processed/labels.csv", encoding="utf-8"))
            if r["split"] == "test" and r["mask_quality"] == "pseudo"]
    rng = random.Random(seed)
    out = []
    for src in ("kaggle_fresh", "kaggle_rotten"):
        cands = sorted((r for r in rows if r["source"] == src), key=lambda r: r["image"])
        out.append(rng.choice(cands))
    return out


def wait_up(url, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url + "/api/info", timeout=5)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("app did not start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multitask", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--report-dir", default="results/preliminary")
    ap.add_argument("--out", default="results/review/app")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "FRUIT_REPORT_DIR": str(ROOT / args.report_dir), "PYTHONUTF8": "1"}
    server = subprocess.Popen([sys.executable, "-m", "app.server", "--port", str(args.port),
                               "--multitask", args.multitask, "--baseline", args.baseline], cwd=ROOT, env=env)
    url = f"http://127.0.0.1:{args.port}"
    shots = []
    try:
        wait_up(url)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            for k, r in enumerate(pick_images(args.seed), 1):
                tag = f"img{k}_{r['source'].replace('kaggle_', '')}_{r['fruit']}"
                page.goto(url)
                page.wait_for_function("document.querySelector('#run') && "
                                       "!document.querySelector('[data-model=multitask]').disabled")
                page.set_input_files("#file", str(ROOT / "data/processed" / r["image"]))
                page.click("#run")
                page.wait_for_function("document.querySelector('#status').textContent.startsWith('Done') || "
                                       "document.querySelector('#status').textContent.startsWith('Error')",
                                       timeout=180_000)
                status = page.text_content("#status")
                for view, name in (("analyse", "1_analyse"), ("performance", "2_performance"),
                                   ("how", "3_how_it_works")):
                    page.click(f".tab[data-view={view}]")
                    if view == "performance":
                        page.wait_for_selector("#perf table, #perf p", timeout=30_000)
                    page.wait_for_timeout(800)
                    f = out / f"{tag}_{name}.png"
                    page.screenshot(path=str(f), full_page=True)
                    shots.append(f.name)
                page.click(".tab[data-view=analyse]")
                print(f"{tag}: {r['image']} (pseudo grade {r['grade']}) -> status '{status}'")
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(20)
        except subprocess.TimeoutExpired:
            server.kill()
    print(f"{len(shots)} screenshots -> {out}")


if __name__ == "__main__":
    main()
