from src.generator.template_instantiator import TemplateInstantiator
from src.generator.template_spec import (
    BindGroupSpec,
    PlaceholderSpec,
    TemplateConstraints,
    TemplateSpec,
)


def build_simple_spec():
    placeholders = {
        "P1": PlaceholderSpec(
            name="P1",
            source_column="p.gender",
            operator_kind="equals",
            bind_group="G1",
        )
    }
    bind_groups = {
        "G1": BindGroupSpec(
            group_id="G1",
            from_join_sql="FROM person p",
            required_columns=["p.gender", "p.name"],
            anchor_key="p.name",
        )
    }
    return TemplateSpec(
        type_name="where",
        description="Simple filter",
        sql_template="SELECT p.name FROM person p WHERE p.gender = {P1} LIMIT 5;",
        placeholders=placeholders,
        bind_groups=bind_groups,
        constraints=TemplateConstraints(limit=5),
    )


def test_instantiator_renders_sql_with_literal():
    spec = build_simple_spec()
    witness_pool = {
        "G1": [
            {"p.gender": "female", "p.name": "Alice"},
        ]
    }
    instantiator = TemplateInstantiator(seed=42)
    result = instantiator.instantiate(spec, witness_pool)
    assert result is not None
    assert "female" in result.sql.lower()
    assert result.placeholder_values["P1"] == "'female'"
