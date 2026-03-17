from __future__ import annotations

import os

# from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no covers
    import tomli as tomllib  # type: ignore


def _find_config_toml() -> Path:
    # 1) Explicit override
    env = os.getenv("CONFIG_TOML")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"CONFIG_TOML points to missing file: {p}")
        return p

    # 2) Walk up from CWD looking for config.toml (project root)
    cur = Path.cwd().resolve()
    for base in (cur, *cur.parents):
        candidate = base / "config.toml"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find config.toml. "
        "Run from the project root (where config.toml lives) "
        "or set CONFIG_TOML=/absolute/path/to/config.toml."
    )


# lru_cache(maxsize=1)
def load_config() -> dict:
    cfg_path = _find_config_toml()
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


# lru_cache(maxsize=1)
def paths() -> dict[str, Path]:
    cfg = load_config()

    config_path = _find_config_toml()
    project_root = config_path.parent  # ← anchor here

    base_dir = project_root / cfg["paths"]["base_dir"]

    return {
        "BASE_DIR": base_dir,
        "RESULT_FIGURES_DATA_PATH": base_dir
        / cfg["paths"]["result_figures_data_path"],
        "RESULT_CSV_DATA_PATH": base_dir
        / cfg["paths"]["result_csv_data_path"],
        "SRC_PATH": base_dir / cfg["paths"]["src_path"],
        "DATA_PATH": base_dir / cfg["paths"]["data_path"],
        "PREFILTERED_DATA_PATH": base_dir
        / cfg["paths"]["prefiltered_data_path"],
        "CLINICAL_DATA_PATH": base_dir / cfg["paths"]["clinical_data_path"],
        "RAW_FEATURES_PATH": base_dir / cfg["paths"]["raw_features_path"],
        "PREFILTERED_DATA_PATH_VARIANCE": base_dir
        / cfg["paths"]["prefiltered_data_path_variance"],
        "PREFILTERED_DATA_PATH_DOMAIN": base_dir
        / cfg["paths"]["prefiltered_data_path_domain"],
        "RESULT_JSON_DIR": base_dir / cfg["paths"]["result_json_dir"],
        "RESULT_JSON_FILE": base_dir / cfg["paths"]["result_json_file"],
    }


CONFIG: dict[Any, Any] = load_config()
PATHS: dict[str, Path] = paths()
