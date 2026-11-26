"""Configuration loader for RetinaNet-Pro YAML settings."""

from pathlib import Path
from typing import Any, Dict
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a single YAML configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
