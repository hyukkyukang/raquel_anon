from src.generator.template_spec import BindGroupSpec, PlaceholderSpec, TemplateSpec
from src.generator.template_witness import TemplateWitnessSampler


class FakePGClient:
    def __init__(self, rows):
        self.rows = rows
        self.last_sql = None

    def execute_and_fetchall_with_col_names(self, sql: str):
        self.last_sql = sql
        return self.rows


def test_witness_sampler_builds_query_and_remaps_rows():
    bind_group = BindGroupSpec(
        group_id="G1",
        from_join_sql="FROM person p",
        required_columns=["p.name"],
        filters=["p.name <> ''"],
    )
    placeholder = PlaceholderSpec(
        name="P1",
        source_column="p.name",
        operator_kind="equals",
        bind_group="G1",
    )
    fake_rows = [{"p_name": "Alice"}]
    sampler = TemplateWitnessSampler(FakePGClient(fake_rows))

    rows = sampler.sample_bind_group(bind_group, [placeholder], max_candidates=10)

    assert rows == [{"p.name": "Alice"}]
    assert "FROM person p" in sampler.pg_client.last_sql
    assert "p.name IS NOT NULL" in sampler.pg_client.last_sql
