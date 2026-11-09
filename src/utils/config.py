"""Configuration loader for RetinaNet-Pro YAML settings."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a single YAML configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_all_configs(config_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load and merge all YAML configuration files in the configs directory."""
    directory = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    merged: Dict[str, Any] = {}
    if not directory.exists():
        return merged
    for yaml_file in sorted(directory.glob("*.yaml")):
        data = load_yaml_config(yaml_file)
        merged.update(data)
    return merged
