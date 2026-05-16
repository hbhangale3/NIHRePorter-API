from __future__ import annotations

import yaml

from .models import AppConfig


def load_config_from_yaml_str(config_yaml: str) -> AppConfig:
    data = yaml.safe_load(config_yaml) or {}
    return AppConfig.model_validate(data)
