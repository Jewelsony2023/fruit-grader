"""Load config.yaml into a plain dict (one place, used by every script)."""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # make data root absolute (relative paths are relative to the project folder)
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        cfg["data"]["root"] = str(PROJECT_ROOT / root)
    out = Path(cfg["train"]["out_dir"])
    if not out.is_absolute():
        cfg["train"]["out_dir"] = str(PROJECT_ROOT / out)
    return cfg
