"""Deprecated training entrypoint kept only as a migration stub.

This repository now uses the Lightning-based training system under
`script/train/` and `src/training/train.py`.
"""

from __future__ import annotations

import sys
import textwrap
import warnings


DEPRECATION_MESSAGE = textwrap.dedent(
    """
    script/evaluation/train.py has been removed as an executable training pipeline.

    Use one of the maintained entrypoints instead:
      python script/train/finetune_retain.py ...
      python script/train/finetune_full.py ...
      python script/train/unlearn.py ...

    The old script implemented a separate non-Lightning stack and was intentionally
    retired to avoid maintaining two incompatible training architectures.
    """
).strip()


warnings.warn(
    "script/evaluation/train.py is deprecated and now only provides migration guidance.",
    DeprecationWarning,
    stacklevel=2,
)


def main() -> None:
    """Exit with migration guidance for the maintained training entrypoints."""
    print(DEPRECATION_MESSAGE, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
