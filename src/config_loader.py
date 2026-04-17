"""YAML configuration loader."""

import yaml


def load_config(path):
    """Load experiment configuration from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)
