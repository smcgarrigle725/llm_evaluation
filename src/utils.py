import json
from pathlib import Path
from typing import Any, Dict


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
