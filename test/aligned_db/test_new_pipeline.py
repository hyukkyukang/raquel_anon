"""Tests for the new AlignedDBPipeline data structures and components."""

import pytest
from typing import Any, Dict, List
from omegaconf import OmegaConf


class TestQAExtraction:
    """Tests for QAExtraction dataclass."""

    def test_empty_extraction(self):
        """Test creating an empty extraction."""
        from src.aligned_db.qa_extraction import QAExtraction

        extraction = QAExtraction(
            qa_index=0,
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
        )

        assert extraction.qa_index == 0
        assert extraction.is_empty
        assert extraction.entity_count == 0
        assert extraction.relation_count == 0
        assert extraction.validation_status == "pending"

    def test_add_entity(self):
        """Test adding entities to extraction."""
        from src.aligned_db.qa_extraction import QAExtraction

        extraction = QAExtraction(
            qa_index=0,
            question="Who wrote Hamlet?",
            answer="William Shakespeare wrote Hamlet.",
        )

        extraction.add_entity("person", {"full_name": "William Shakespeare"})
        extraction.add_entity("work", {"title": "Hamlet"})

        assert not extraction.is_empty
        assert extraction.entity_count == 2
        assert "person" in extraction.relevant_tables
        assert "work" in extraction.relevant_tables

    def test_add_relation(self):
        """Test adding relations to extraction."""
        from src.aligned_db.qa_extraction import QAExtraction

        extraction = QAExtraction(
            qa_index=0,
            question="Who wrote Hamlet?",
            answer="William Shakespeare wrote Hamlet.",
        )

        extraction.add_relation(
            {
                "type": "person_work",
                "source": "William Shakespeare",
                "target": "Hamlet",
            }
        )

        assert extraction.relation_count == 1
        assert "person_work" in extraction.relevant_tables

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        from src.aligned_db.qa_extraction import QAExtraction

        extraction = QAExtraction(
            qa_index=5,
            question="Test question",
            answer="Test answer",
            entities={"person": [{"name": "Test"}]},
            relations=[{"type": "test_relation"}],
            entity_attribute_metadata={
                "person": [
                    {
                        "name": {
                            "raw_value": "Test",
                            "normalized_value": "Test",
                            "canonical_candidate": "Test",
                            "role_hint": "scalar",
                            "target_table": None,
                            "confidence": 1.0,
                            "evidence": [],
                        }
                    }
                ]
            },
            relation_metadata=[{"canonical_type": "test_relation"}],
        )

        data = extraction.to_dict()
        restored = QAExtraction.from_dict(data)

        assert restored.qa_index == extraction.qa_index
        assert restored.question == extraction.question
        assert restored.entities == extraction.entities
        assert restored.entity_attribute_metadata == extraction.entity_attribute_metadata
        assert restored.relation_metadata == extraction.relation_metadata

    def test_add_entity_and_relation_metadata_align(self):
        """Metadata containers should stay aligned with entity/relation payloads."""
        from src.aligned_db.qa_extraction import QAExtraction

        extraction = QAExtraction(qa_index=0, question="Q", answer="A")
        extraction.add_entity(
            "person",
            {"name": "Ada"},
            {"name": {"raw_value": "Ada", "role_hint": "scalar"}},
        )
        extraction.add_relation(
            {"type": "person_work", "source": "Ada", "target": "Book"},
            {"canonical_type": "person_work"},
        )

        assert extraction.get_entity_attribute_metadata("person", 0)["name"]["raw_value"] == "Ada"
        assert extraction.get_relation_metadata(0)["canonical_type"] == "person_work"


class TestQAExtractionRegistry:
    """Tests for QAExtractionRegistry."""

    def test_empty_registry(self):
        """Test creating empty registry."""
        from src.aligned_db.qa_extraction import QAExtractionRegistry

        registry = QAExtractionRegistry.empty()

        assert registry.count == 0
        assert len(list(registry)) == 0

    def test_add_and_get(self):
        """Test adding and retrieving extractions."""
        from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry

        registry = QAExtractionRegistry.empty()

        extraction = QAExtraction(
            qa_index=0,
            question="Q1",
            answer="A1",
        )
        registry.add(extraction)

        assert registry.count == 1
        assert registry.get(0) is not None
        assert registry.get(1) is None

    def test_merge_entities(self):
        """Test merging entities from multiple extractions."""
        from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry

        registry = QAExtractionRegistry.empty()

        e1 = QAExtraction(qa_index=0, question="Q1", answer="A1")
        e1.add_entity("person", {"name": "Alice"})

        e2 = QAExtraction(qa_index=1, question="Q2", answer="A2")
        e2.add_entity("person", {"name": "Bob"})

        registry.add(e1)
        registry.add(e2)

        merged = registry.merge_entities()
        assert "person" in merged
        assert len(merged["person"]) == 2

    def test_get_relevant_tables(self):
        """Test getting relevant tables for a QA pair."""
        from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry

        registry = QAExtractionRegistry.empty()

        extraction = QAExtraction(qa_index=0, question="Q", answer="A")
        extraction.add_entity("person", {"name": "Test"})
        extraction.add_entity("work", {"title": "Book"})
        registry.add(extraction)

        tables = registry.get_relevant_tables(0)
        assert "person" in tables
        assert "work" in tables

    def test_entity_registry_relation_merge_preserves_metadata_endpoint_types(self):
        """Merged junction records should use relation metadata endpoint types, including multi-word types."""
        from src.aligned_db.entity_registry import EntityRegistry
        from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry

        extraction = QAExtraction(qa_index=0, question="Q", answer="A")
        extraction.add_relation(
            {
                "type": "newspaper_article_person",
                "source": "Daily Gazette",
                "target": "Ada Lovelace",
            },
            {
                "source_entity_type": "newspaper_article",
                "target_entity_type": "person",
            },
        )

        registry = QAExtractionRegistry.empty()
        registry.add(extraction)
        entity_registry = EntityRegistry.from_qa_extractions(registry)

        self.assertEqual(
            entity_registry.get_relationships("newspaper_article_person"),
            [
                {
                    "newspaper_article_name": "Daily Gazette",
                    "person_name": "Ada Lovelace",
                }
            ],
        )


class TestPipelineFlow:
    """Tests for pipeline stage wiring."""

    def test_extraction_stage_can_use_different_qa_pairs_than_discovery(self):
        """Stage 4 should accept normalized QA pairs without changing discovery inputs."""
        from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry
        from src.aligned_db.schema_registry import SchemaRegistry
        from src.aligned_db.type_registry import TypeRegistry
        from src.generator.pipeline_flow import run_extraction_and_registry_stages

        class StubExtractor:
            def __init__(self):
                self.seen_qa_pairs = None

            async def extract_all(
                self,
                qa_pairs,
                schema_registry,
                type_registry,
                max_concurrency=None,
            ):
                del schema_registry, type_registry, max_concurrency
                self.seen_qa_pairs = list(qa_pairs)
                registry = QAExtractionRegistry.empty()
                for idx, (question, answer) in enumerate(qa_pairs):
                    registry.add(
                        QAExtraction(
                            qa_index=idx,
                            question=question,
                            answer=answer,
                        )
                    )
                return registry

        captured_stage4 = {}
        extractor = StubExtractor()
        original_pairs = [("Who is this celebrated author?", "The author's full name is Ada.")]
        normalized_pairs = [("Who is the author?", "Ada.")]

        artifacts = run_extraction_and_registry_stages(
            qa_pairs=original_pairs,
            extraction_qa_pairs=normalized_pairs,
            qa_sources=["retain"],
            schema_registry=SchemaRegistry(),
            type_registry=TypeRegistry(),
            per_qa_extractor=extractor,
            extraction_max_concurrency=1,
            validation_enabled=False,
            run_extraction_validation_fn=lambda *args, **kwargs: args[1],
            save_stage4_results_fn=lambda qa_pairs, qa_extractions, quality_gate_summary: captured_stage4.update(
                {
                    "qa_pairs": list(qa_pairs),
                    "quality_gate_summary": quality_gate_summary,
                    "qa_extractions": qa_extractions,
                }
            ),
            save_stage4_5_results_fn=lambda qa_extractions: None,
            save_stage5_results_fn=lambda entity_registry: None,
        )

        assert extractor.seen_qa_pairs == normalized_pairs
        assert captured_stage4["qa_pairs"] == original_pairs
        assert artifacts.qa_extractions.get(0).question == normalized_pairs[0][0]
        assert artifacts.qa_extractions.get(0).source == "retain"

    def test_aligned_db_build_can_separate_discovery_and_extraction_corpora(self, monkeypatch):
        """AlignedDB.build should preserve explicit discovery and extraction corpora."""
        from src.aligned_db.db import AlignedDB

        captured = {}

        class StopBuild(RuntimeError):
            pass

        def fake_prepare_build_artifacts(**kwargs):
            captured.update(kwargs)
            raise StopBuild()

        monkeypatch.setattr("src.aligned_db.db.prepare_build_artifacts", fake_prepare_build_artifacts)

        cfg = OmegaConf.create(
            {
                "project_path": "/tmp",
                "model": {"dir_path": "aligned_db_test"},
                "database": {
                    "db_id": "test_db",
                    "host": "localhost",
                    "port": 5433,
                    "user_id": "postgres",
                    "passwd": "postgres",
                },
                "llm": {},
            }
        )
        aligned_db = AlignedDB(cfg)

        canonical_pairs = [("Who is this celebrated author?", "The author's full name is Ada.")]
        discovery_pairs = [("Who is this celebrated author?", "The author's full name is Ada.")]
        normalized_pairs = [("Who is the author?", "Ada.")]
        extraction_pairs = [("Who is the author?", "Ada.")]

        with pytest.raises(StopBuild):
            aligned_db.build(
                canonical_pairs,
                discovery_qa_pairs=discovery_pairs,
                normalized_qa_pairs=normalized_pairs,
                extraction_qa_pairs=extraction_pairs,
                overwrite=True,
            )

        assert captured["qa_pairs"] == discovery_pairs
        assert captured["normalized_qa_pairs"] == normalized_pairs
        assert captured["extraction_qa_pairs"] == extraction_pairs


class TestTypeRegistry:
    """Tests for TypeRegistry and related dataclasses."""

    def test_entity_type(self):
        """Test EntityType dataclass."""
        from src.aligned_db.type_registry import EntityType

        et = EntityType(
            name="person",
            description="A human being",
            aliases=["individual", "human"],
        )

        assert et.name == "person"
        assert et.is_junction is False
        assert "individual" in et.aliases

    def test_attribute_type(self):
        """Test AttributeType dataclass."""
        from src.aligned_db.type_registry import AttributeType

        attr = AttributeType(
            name="full_name",
            data_type="TEXT",
            is_natural_key=True,
            is_unique=True,
        )

        assert attr.name == "full_name"
        assert attr.is_natural_key
        assert attr.is_unique

    def test_relation_type(self):
        """Test RelationType dataclass."""
        from src.aligned_db.type_registry import RelationType

        rel = RelationType(
            name="person_work",
            source_entity="person",
            target_entity="work",
        )

        assert rel.source_fk_name == "person_id"
        assert rel.target_fk_name == "work_id"

    def test_type_registry_operations(self):
        """Test TypeRegistry add and get operations."""
        from src.aligned_db.type_registry import (
            AttributeType,
            EntityType,
            RelationType,
            TypeRegistry,
        )

        registry = TypeRegistry.empty()

        # Add entity type
        registry.add_entity_type(EntityType(name="person", description="A person"))

        # Add attribute
        registry.add_attribute_type(
            "person",
            AttributeType(name="full_name", data_type="TEXT"),
        )

        # Add relation
        registry.add_relation_type(
            RelationType(
                name="person_work", source_entity="person", target_entity="work"
            )
        )

        assert "person" in registry.entity_type_names
        assert "person_work" in registry.relation_type_names
        assert len(registry.get_attributes_for("person")) == 1

    def test_schema_registry_treats_date_note_fields_as_text(self):
        """Date note/precision fields should not be inferred as DATE columns."""
        from src.aligned_db.schema_registry import SchemaRegistry

        registry = SchemaRegistry.empty()

        assert (
            registry._infer_type_from_value(
                "publication_date_precision_note",
                "Only the year 2020 is stated.",
            )
            == "TEXT"
        )


class TestAsyncRateLimiter:
    """Tests for AsyncRateLimiter."""

    def test_rate_limiter_basic(self):
        """Test basic rate limiter functionality."""
        import asyncio
        from src.utils.async_utils import AsyncRateLimiter

        async def _test():
            limiter = AsyncRateLimiter(rate=10.0)
            # Should acquire immediately (have tokens)
            await limiter.acquire(1)
            # Reset should restore tokens
            limiter.reset()
            assert limiter.tokens == limiter.max_tokens

        asyncio.run(_test())

    def test_rate_limiter_unlimited(self):
        """Test unlimited rate (rate=0)."""
        import asyncio
        from src.utils.async_utils import AsyncRateLimiter

        async def _test():
            limiter = AsyncRateLimiter(rate=0)
            # Should return immediately with no rate limiting
            await limiter.acquire(100)

        asyncio.run(_test())


class TestLLMComponent:
    """Tests for LLMComponent base class."""

    def test_extract_json_from_markdown(self):
        """Test JSON extraction from markdown code blocks."""
        from src.utils.json_utils import extract_json_from_response

        # Test with ```json block
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = extract_json_from_response(text)
        assert result == '{"key": "value"}'

        # Test with plain ``` block
        text2 = '```\n{"key": "value2"}\n```'
        result2 = extract_json_from_response(text2)
        assert result2 == '{"key": "value2"}'

        # Test with raw JSON
        text3 = '{"key": "value3"}'
        result3 = extract_json_from_response(text3)
        assert result3 == '{"key": "value3"}'

    def test_fix_js_string_concat(self):
        """Test fixing JavaScript string concatenation."""
        from src.utils.json_utils import fix_js_string_concat

        # Test basic concatenation
        text = '{"value": "part1" + "part2"}'
        result = fix_js_string_concat(text)
        assert result == '{"value": "part1part2"}'

        # Test chained concatenation
        text2 = '{"value": "a" + "b" + "c"}'
        result2 = fix_js_string_concat(text2)
        assert result2 == '{"value": "abc"}'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
