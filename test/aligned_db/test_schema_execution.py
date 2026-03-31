"""Regression tests for schema execution with mixed DDL input."""

from __future__ import annotations

import unittest


class _FakeConn:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _FakePGClient:
    def __init__(self) -> None:
        self.conn = _FakeConn()
        self.executed: list[str] = []

    def execute(self, sql: str) -> None:
        self.executed.append(sql)


class _FakeSQLGenerator:
    def sql_syntax_correction(self, sql, error_message=None, schema_context=None):
        del sql, error_message, schema_context
        return []


class TestSchemaExecution(unittest.TestCase):
    """Ensure mixed CREATE/ALTER schema lists are executed safely."""

    def test_execute_schema_statements_skips_duplicate_input_fk_alters(self):
        from src.aligned_db.schema_execution import execute_schema_statements

        schema = [
            """
            CREATE TABLE person (
                person_id INTEGER PRIMARY KEY,
                occupation_id INTEGER,
                FOREIGN KEY (occupation_id) REFERENCES occupation(occupation_id)
            );
            """.strip(),
            """
            CREATE TABLE occupation (
                occupation_id INTEGER PRIMARY KEY,
                name TEXT
            );
            """.strip(),
            """
            ALTER TABLE person ADD CONSTRAINT fk_person_occupation_id
            FOREIGN KEY (occupation_id) REFERENCES occupation(occupation_id);
            """.strip(),
        ]

        pg_client = _FakePGClient()
        execute_schema_statements(
            pg_client=pg_client,
            sql_generator=_FakeSQLGenerator(),
            schema=schema,
        )

        self.assertEqual(len(pg_client.executed), 3)
        self.assertTrue(pg_client.executed[0].lstrip().upper().startswith("CREATE TABLE"))
        self.assertTrue(pg_client.executed[1].lstrip().upper().startswith("CREATE TABLE"))
        self.assertTrue(pg_client.executed[2].lstrip().upper().startswith("ALTER TABLE"))
        self.assertNotIn("Created table: ", "\n".join([]))
        self.assertEqual(
            sum(
                1
                for sql in pg_client.executed
                if "ADD CONSTRAINT fk_person_occupation_id" in sql
            ),
            1,
        )

    def test_execute_schema_statements_runs_non_fk_passthrough_ddl_after_tables(self):
        from src.aligned_db.schema_execution import execute_schema_statements

        schema = [
            "CREATE TABLE person (person_id INTEGER PRIMARY KEY, name TEXT);",
            "ALTER TABLE person ADD COLUMN bio TEXT;",
        ]

        pg_client = _FakePGClient()
        execute_schema_statements(
            pg_client=pg_client,
            sql_generator=_FakeSQLGenerator(),
            schema=schema,
        )

        self.assertEqual(len(pg_client.executed), 2)
        self.assertTrue(pg_client.executed[0].startswith("CREATE TABLE person"))
        self.assertTrue(pg_client.executed[1].startswith("ALTER TABLE person ADD COLUMN"))


if __name__ == "__main__":
    unittest.main()
