#!/usr/bin/env python
"""Bootstrap Redis with recent cnt observations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ConfigLoader, ProjectPaths
from src.utils.redis_manager import RedisManager


def main() -> None:
    config = ConfigLoader()
    paths = ProjectPaths(config)
    redis_mgr = RedisManager()

    clean_path = paths.clean_data_file
    df_clean = pd.read_csv(clean_path)
    if 'dteday' not in df_clean.columns:
        raise ValueError("Column 'dteday' must be present in the clean dataset")

    df_clean = df_clean.sort_values(["dteday", "hr"]).reset_index(drop=True)
    redis_mgr.bootstrap(df_clean)
    print(f"Bootstrap complete with {len(df_clean)} rows from {clean_path}")


if __name__ == "__main__":
    main()
