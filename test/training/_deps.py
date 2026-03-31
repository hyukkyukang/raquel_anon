"""Helpers for dependency-aware training tests."""

from __future__ import annotations

import importlib.util
from typing import Iterable, List


def missing_optional_packages(*package_names: str) -> List[str]:
    """Return the subset of package names that are unavailable."""
    return [
        package_name
        for package_name in package_names
        if importlib.util.find_spec(package_name) is None
    ]


def require_optional_packages(testcase, *package_names: str) -> None:
    """Skip the current unittest when optional dependencies are unavailable."""
    missing = missing_optional_packages(*package_names)
    if missing:
        testcase.skipTest(
            "Missing optional dependency(s): " + ", ".join(sorted(missing))
        )
