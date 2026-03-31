"""Generator package exports with lazy loading."""

from importlib import import_module
from typing import Any, Dict, Tuple
import warnings

_ACTIVE_EXPORTS: Dict[str, Tuple[str, str]] = {
    "SQLGenerator": (".sql", "SQLGenerator"),
    "QuerySynthesizer": (".synthesizer", "QuerySynthesizer"),
    "AlignedDBPipeline": (".entity_pipeline", "AlignedDBPipeline"),
    "EntityTypeDiscoverer": (".entity_type_discoverer", "EntityTypeDiscoverer"),
    "AttributeDiscoverer": (".attribute_discoverer", "AttributeDiscoverer"),
    "AttributeNormalizer": (".attribute_normalizer", "AttributeNormalizer"),
    "DynamicSchemaGenerator": (".dynamic_schema_generator", "DynamicSchemaGenerator"),
    "PerQAExtractor": (".per_qa_extractor", "PerQAExtractor"),
    "ExtractionValidator": (".extraction_validator", "ExtractionValidator"),
    "RelationDiscoverer": (".relation_discoverer", "RelationDiscoverer"),
    "RoundTripVerifier": (".round_trip_verifier", "RoundTripVerifier"),
    "VerificationResult": (".round_trip_verifier", "VerificationResult"),
}

_LEGACY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "SchemaGenerator": (".schema", "SchemaGenerator"),
}

__all__ = list(_ACTIVE_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _LEGACY_EXPORTS:
        module_path, attr_name = _LEGACY_EXPORTS[name]
        warnings.warn(
            "src.generator.SchemaGenerator is legacy and kept only for backwards "
            "compatibility. Prefer src.generator.DynamicSchemaGenerator.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    if name not in _ACTIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _ACTIVE_EXPORTS[name]
    module = import_module(module_path, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
