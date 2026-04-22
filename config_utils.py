from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a YAML mapping: {config_path}")
    return data


def flatten_sections(config: dict, *sections: str) -> dict:
    flat: dict = {}
    for section in sections:
        value = config.get(section, {})
        if value is None:
            continue
        if not isinstance(value, dict):
            raise SystemExit(f"Config section '{section}' must be a mapping.")
        flat.update(value)
    return flat
