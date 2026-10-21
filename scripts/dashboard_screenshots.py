"""
Screenshots of every dashboard tab at projector (1366x768) and full-HD (1920x1080) size,
with ALL non-local requests blocked, to prove the page works offline. Starts the app
itself and stops it. Also shots the Analyse page (tab 10 = live demo link).

Needs: pip install playwright (uses the installed Google Chrome, no browser download).

    python scripts/dashboard_screenshots.py [--out results/review/dashboard]

Prints, per shot, the page height (taller than the window = the audience must scroll),
plus any console errors and blocked/failed requests.
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
TABS = ["overview", "confusion", "pergrade", "segmentation", "severity", "confidence", "training", "data", "errors"]
SIZES = [(1366, 768), (1920, 1080)]


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
    ap.add_argument("--multitask", default="runs/prelim_multitask_efficientnet_b0/best.pt")
    ap.add_argument("--baseline", default="runs/prelim_baseline_efficientnet_b0/best.pt")
    ap.add_argument("--dashboard-dir", default="results/preliminary")
    ap.add_argument("--out", default="results/review/dashboard")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "FRUIT_DASHBOARD_DIR": str(ROOT / args.dashboard_dir),
           "FRUIT_REPORT_DIR": str(ROOT / args.dashboard_dir), "PYTHONUTF8": "1"}
    server = subprocess.Popen([sys.executable, "-m", "app.server", "--port", str(args.port),
                               "--multitask", args.multitask, "--baseline", args.baseline], cwd=ROOT, env=env)
    url = f"http://127.0.0.1:{args.port}"
    problems = []
    try:
        wait_up(url)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            for w, h in SIZES:
                page = browser.new_page(viewport={"width": w, "height": h})
                blocked = []

                def route(r, blocked=blocked):
                    if r.request.url.startswith(url):
                        r.continue_()
                    else:
                        blocked.append(r.request.url)
                        r.abort()
                page.route("**/*", route)
                page.on("console", lambda m: m.type == "error" and problems.append(f"{w}x{h} console: {m.text}"))
                page.on("pageerror", lambda e: problems.append(f"{w}x{h} page error: {e}"))
                page.on("requestfailed", lambda r: r.url.startswith(url) and problems.append(f"{w}x{h} failed: {r.url}"))
                page.goto(f"{url}/dashboard#overview")
                page.wait_for_selector("#kpis .kpi", timeout=60_000)
                for k, tab in enumerate(TABS, 1):
                    page.click(f"#tabs button[data-tab={tab}]")
                    if tab == "confusion":   # show the gallery of one off-diagonal cell (multitask, true C -> pred Reject)
                        page.click(".cm td[data-mo=multitask][data-i='2'][data-j='3']")
                    page.wait_for_timeout(700)
                    page.wait_for_load_state("networkidle")
                    f = out / f"{w}x{h}_{k:02d}_{tab}.png"
                    page.screenshot(path=str(f), full_page=True)
                    height = page.evaluate("document.documentElement.scrollHeight")
                    print(f"{f.name}: page height {height}px (window {h}px){'  <- scrolls' if height > h else ''}")
                page.goto(url + "/", wait_until="domcontentloaded")
                page.wait_for_selector("#samples button", timeout=30_000)
                page.wait_for_timeout(1500)
                page.screenshot(path=str(out / f"{w}x{h}_10_live_demo.png"), full_page=False)
                print(f"{w}x{h}_10_live_demo.png; blocked non-local requests: {sorted(set(blocked)) or 'none'}")
                page.close()
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(20)
        except subprocess.TimeoutExpired:
            server.kill()
    print("\n".join(problems) if problems else "no console errors, no failed local requests")


if __name__ == "__main__":
    main()
