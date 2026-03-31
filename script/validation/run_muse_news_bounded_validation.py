#!/usr/bin/env python3
"""Run the checked-in bounded MUSE-News validation profile."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_CONFIG = REPO_ROOT / "config" / "validation" / "muse_news_bounded.yaml"


def main() -> None:
    command = [
        sys.executable,
        "script/validation/run_muse_news_smoke_validation.py",
        "--profile-config",
        str(PROFILE_CONFIG),
        *sys.argv[1:],
    ]
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
