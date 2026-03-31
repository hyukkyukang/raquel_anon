import pytest

from src.generator.template_spec import (
    BindGroupSpec,
    PlaceholderSpec,
    TemplateConstraints,
    TemplateSpec,
)


def make_spec_dict():
    return {
        "type_name": "where",
        "description": "Filter by gender and nationality",
        "sql_template": "SELECT p.name FROM person p WHERE p.gender = {P1} LIMIT 25;",
        "placeholders": {
            "P1": {
                "name": "P1",
                "source_column": "p.gender",
                "operator_kind": "equals",
                "bind_group": "G1",
            }
        },
        "bind_groups": {
            "G1": {
                "group_id": "G1",
                "from_join_sql": "FROM person p",
                "required_columns": ["p.gender", "p.name"],
            }
        },
        "constraints": {"limit": 25, "disallow_destructive": True},
    }


def test_template_spec_validation_passes_with_matching_placeholders():
    spec = TemplateSpec.from_dict(make_spec_dict())
    errors = spec.validate()
    assert errors == []


def test_template_spec_validation_detects_missing_placeholder():
    bad_spec_dict = make_spec_dict()
    bad_spec_dict["sql_template"] = "SELECT p.name FROM person p WHERE p.gender = {P2};"
    spec = TemplateSpec.from_dict(bad_spec_dict)
    errors = spec.validate()
    assert any("placeholders" in err for err in errors)
