# Hotfix argparse for Python3.14 and Hydra.cc compatibility
import argparse as _argparse
import logging


def _get_logger():
    """Get logger for this module, handling early import case."""
    try:
        from src.utils.logging import get_logger

        return get_logger(__name__, __file__)
    except ImportError:
        # Fallback if src.utils.logging is not available yet
        return logging.getLogger(__name__)


def _patch_hf_dill_batch_setitems() -> None:
    """
    Hugging Face vendors a thin wrapper around dill.Pickler that overrides
    `_batch_setitems`. Python 3.13+ changed the C Pickler signature to pass the
    original dict alongside the items iterator. Older overrides that only accept
    `(self, items)` now crash with `TypeError: ... takes 2 positional arguments but 3 were given`.

    We monkey patch the wrapper so it gracefully handles both call styles.
    """
    logger = _get_logger()
    try:
        from datasets.utils import _dill as _hf_dill  # type: ignore[attr-defined]
        import dill  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Skipping HF dill patch: %s", exc)
        return

    original = getattr(_hf_dill.Pickler, "_batch_setitems", None)
    if original is None:
        logger.debug("HF dill Pickler has no _batch_setitems")
        return

    # Avoid double patching
    if getattr(original, "__patched_for_py313__", False):
        return

    _MISSING = object()

    def _call_with_optional_obj(fn, items, obj=_MISSING):
        try:
            if obj is _MISSING:
                return fn(items, None)
            return fn(items, obj)
        except TypeError:
            if obj is _MISSING:
                return fn(items)
            return fn(items)

    def _patched_batch_setitems(self, items, *rest):
        obj = rest[0] if rest else _MISSING
        if self._legacy_no_dict_keys_sorting:
            return _call_with_optional_obj(
                super(_hf_dill.Pickler, self)._batch_setitems, items, obj
            )

        try:
            items = sorted(items)
        except Exception:
            from datasets.fingerprint import Hasher

            items = sorted(items, key=lambda x: Hasher.hash(x[0]))

        return _call_with_optional_obj(
            dill.Pickler._batch_setitems.__get__(self, dill.Pickler),
            items,
            obj,
        )

    setattr(_patched_batch_setitems, "__patched_for_py313__", True)
    _hf_dill.Pickler._batch_setitems = _patched_batch_setitems


_orig_add_argument = _argparse.ArgumentParser.add_argument
_PATCHES_APPLIED = False


def _patched_add_argument(self, *args, **kwargs):
    if "help" in kwargs and not isinstance(kwargs["help"], str):
        kwargs["help"] = str(kwargs["help"])
    return _orig_add_argument(self, *args, **kwargs)


def apply_python314_compatibility_patches() -> None:
    """Apply the narrow Python 3.14 compatibility patches needed before Hydra imports."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    _argparse.ArgumentParser.add_argument = _patched_add_argument
    _patch_hf_dill_batch_setitems()
    _PATCHES_APPLIED = True
