"""Per-QA extraction data structures for entity-first pipeline.

This module provides data structures for storing extraction results
with full QA-to-entity attribution, enabling precise schema filtering
and verification.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("src.aligned_db.qa_extraction")


# =============================================================================
# Fact Extraction Data Structures
# =============================================================================


@dataclass
class AnswerFact:
    """A single fact extracted from an answer.

    This dataclass represents a discrete piece of information from a QA pair
    as a subject-predicate-object triple, used for validating extraction coverage.

    Attributes:
        subject: The entity the fact is about (e.g., "Jaime Vasquez")
        predicate: The attribute or relationship type (e.g., "received_award")
        object: The value or target entity (e.g., "Edgar Allan Poe Award")
        fact_text: Original text snippet containing this fact
        fact_type: Type of fact - 'attribute', 'relationship', or 'identifier'
        confidence: Confidence score for this fact (0.0 to 1.0)
    """

    subject: str
    predicate: str
    object: str
    fact_text: str
    fact_type: str  # 'attribute' | 'relationship' | 'identifier'
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "fact_text": self.fact_text,
            "fact_type": self.fact_type,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerFact":
        """Create AnswerFact from dictionary."""
        return cls(
            subject=data.get("subject", ""),
            predicate=data.get("predicate", ""),
            object=data.get("object", ""),
            fact_text=data.get("fact_text", ""),
            fact_type=data.get("fact_type", "attribute"),
            confidence=data.get("confidence", 1.0),
        )

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.subject}.{self.predicate} = {self.object}"


@dataclass
class FactCoverageResult:
    """Result of checking fact coverage in an extraction.

    This dataclass holds the results of comparing extracted answer facts
    against the entities and relations in a QAExtraction.

    Attributes:
        coverage_score: Fraction of facts found (0.0 to 1.0)
        found_facts: List of facts that were found in the extraction
        missing_facts: List of facts not found in the extraction
        match_details: Dict mapping fact_text to location where found
    """

    coverage_score: float
    found_facts: List[AnswerFact] = field(default_factory=list)
    missing_facts: List[AnswerFact] = field(default_factory=list)
    match_details: Dict[str, str] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if all facts were found."""
        return len(self.missing_facts) == 0

    @property
    def total_facts(self) -> int:
        """Get total number of facts checked."""
        return len(self.found_facts) + len(self.missing_facts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "coverage_score": self.coverage_score,
            "found_facts": [f.to_dict() for f in self.found_facts],
            "missing_facts": [f.to_dict() for f in self.missing_facts],
            "match_details": self.match_details,
        }


# =============================================================================
# QA Extraction Data Structures
# =============================================================================


@dataclass
class QAExtraction:
    """Extraction results for a single QA pair.

    This dataclass stores all entities, attributes, and relations extracted
    from a single QA pair, maintaining full attribution for downstream
    processing like schema filtering and verification.

    Attributes:
        qa_index: Index of the QA pair in the original list
        question: The original question text
        answer: The original answer text
        source: Source label for nullification ("retain", "forget", or "unknown")
        entities: Dict mapping entity_type to list of entity instances
        relations: List of relation records for junction tables
        relevant_tables: Set of table names this QA pair touches
        extraction_confidence: Confidence score from extraction (0.0 to 1.0)
        validation_status: Status from validation stage (pending/valid/invalid)
        missing_facts: List of facts identified as missing during validation
    """

    qa_index: int
    question: str
    answer: str
    source: str = "unknown"  # "retain", "forget", or "unknown"
    entities: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    entity_attribute_metadata: Dict[str, List[Dict[str, Dict[str, Any]]]] = field(
        default_factory=dict
    )
    relation_metadata: List[Dict[str, Any]] = field(default_factory=list)
    relevant_tables: Set[str] = field(default_factory=set)
    extraction_confidence: float = 1.0
    validation_status: str = "pending"  # pending, valid, invalid
    missing_facts: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._sync_entity_metadata()
        self._sync_relation_metadata()

    # =========================================================================
    # Property Methods
    # =========================================================================

    @property
    def is_valid(self) -> bool:
        """Check if extraction has been validated successfully."""
        return self.validation_status == "valid"

    @property
    def is_empty(self) -> bool:
        """Check if extraction has no entities or relations."""
        return len(self.entities) == 0 and len(self.relations) == 0

    @property
    def entity_count(self) -> int:
        """Get total number of entity instances across all types."""
        return sum(len(instances) for instances in self.entities.values())

    @property
    def relation_count(self) -> int:
        """Get total number of relation records."""
        return len(self.relations)

    # =========================================================================
    # Public Methods
    # =========================================================================

    def get_entity_types(self) -> Set[str]:
        """Get all entity types present in this extraction.

        Returns:
            Set of entity type names
        """
        return set(self.entities.keys())

    def get_entities_of_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all entity instances of a specific type.

        Args:
            entity_type: The type of entities to retrieve

        Returns:
            List of entity dictionaries, empty if type not found
        """
        return self.entities.get(entity_type, [])

    def add_entity(
        self,
        entity_type: str,
        entity: Dict[str, Any],
        attribute_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Add an entity to this extraction.

        Args:
            entity_type: The type of entity
            entity: The entity dictionary
        """
        if entity_type not in self.entities:
            self.entities[entity_type] = []
        if entity_type not in self.entity_attribute_metadata:
            self.entity_attribute_metadata[entity_type] = []
        self.entities[entity_type].append(entity)
        self.entity_attribute_metadata[entity_type].append(attribute_metadata or {})
        self.relevant_tables.add(entity_type)

    def add_relation(
        self,
        relation: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a relation record to this extraction.

        Args:
            relation: The relation dictionary with 'type' key indicating junction table
        """
        self.relations.append(relation)
        self.relation_metadata.append(metadata or {})
        # Add junction table to relevant tables
        if "type" in relation:
            self.relevant_tables.add(relation["type"])

    def get_entity_attribute_metadata(
        self,
        entity_type: str,
        entity_index: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Get metadata for a specific entity instance."""
        self._sync_entity_metadata()
        entity_metadata = self.entity_attribute_metadata.get(entity_type, [])
        if 0 <= entity_index < len(entity_metadata):
            return entity_metadata[entity_index]
        return {}

    def set_entity_attribute_metadata(
        self,
        entity_type: str,
        entity_index: int,
        attribute_name: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Set metadata for a specific entity attribute."""
        self._sync_entity_metadata()
        entity_metadata = self.entity_attribute_metadata.setdefault(entity_type, [])
        while len(entity_metadata) <= entity_index:
            entity_metadata.append({})
        entity_metadata[entity_index][attribute_name] = metadata

    def get_relation_metadata(self, relation_index: int) -> Dict[str, Any]:
        """Get metadata for a relation record."""
        self._sync_relation_metadata()
        if 0 <= relation_index < len(self.relation_metadata):
            return self.relation_metadata[relation_index]
        return {}

    def update_relevant_tables(self) -> None:
        """Recompute relevant_tables from entities and relations."""
        self.relevant_tables = set(self.entities.keys())
        for relation in self.relations:
            if "type" in relation:
                self.relevant_tables.add(relation["type"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all extraction data
        """
        return {
            "qa_index": self.qa_index,
            "question": self.question,
            "answer": self.answer,
            "source": self.source,
            "entities": self.entities,
            "relations": self.relations,
            "entity_attribute_metadata": self.entity_attribute_metadata,
            "relation_metadata": self.relation_metadata,
            "relevant_tables": list(self.relevant_tables),
            "extraction_confidence": self.extraction_confidence,
            "validation_status": self.validation_status,
            "missing_facts": self.missing_facts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QAExtraction":
        """Create QAExtraction from dictionary.

        Args:
            data: Dictionary with extraction data

        Returns:
            New QAExtraction instance
        """
        return cls(
            qa_index=data.get("qa_index", 0),
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            source=data.get("source", "unknown"),
            entities=data.get("entities", {}),
            relations=data.get("relations", []),
            entity_attribute_metadata=data.get("entity_attribute_metadata", {}),
            relation_metadata=data.get("relation_metadata", []),
            relevant_tables=set(data.get("relevant_tables", [])),
            extraction_confidence=data.get("extraction_confidence", 1.0),
            validation_status=data.get("validation_status", "pending"),
            missing_facts=data.get("missing_facts", []),
        )

    def _sync_entity_metadata(self) -> None:
        """Keep entity metadata aligned with entity instance lists."""
        for entity_type, entities in self.entities.items():
            metadata_list = self.entity_attribute_metadata.setdefault(entity_type, [])
            while len(metadata_list) < len(entities):
                metadata_list.append({})
            if len(metadata_list) > len(entities):
                del metadata_list[len(entities) :]

    def _sync_relation_metadata(self) -> None:
        """Keep relation metadata aligned with relation records."""
        while len(self.relation_metadata) < len(self.relations):
            self.relation_metadata.append({})
        if len(self.relation_metadata) > len(self.relations):
            del self.relation_metadata[len(self.relations) :]


@dataclass
class QAExtractionRegistry:
    """Registry holding extractions for all QA pairs.

    This class provides storage and lookup functionality for per-QA
    extraction results, supporting operations like merging, deduplication,
    and schema filtering.

    Attributes:
        extractions: Dictionary mapping qa_index to QAExtraction
    """

    extractions: Dict[int, QAExtraction] = field(default_factory=dict)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def empty(cls) -> "QAExtractionRegistry":
        """Create an empty registry.

        Returns:
            New empty QAExtractionRegistry
        """
        return cls(extractions={})

    @classmethod
    def from_list(cls, extraction_list: List[QAExtraction]) -> "QAExtractionRegistry":
        """Create registry from list of extractions.

        Args:
            extraction_list: List of QAExtraction objects

        Returns:
            New QAExtractionRegistry with all extractions
        """
        registry = cls.empty()
        for extraction in extraction_list:
            registry.add(extraction)
        return registry

    # =========================================================================
    # Property Methods
    # =========================================================================

    @property
    def count(self) -> int:
        """Get total number of extractions."""
        return len(self.extractions)

    @property
    def valid_count(self) -> int:
        """Get number of valid extractions."""
        return sum(1 for e in self.extractions.values() if e.is_valid)

    @property
    def all_entity_types(self) -> Set[str]:
        """Get all entity types across all extractions."""
        types: Set[str] = set()
        for extraction in self.extractions.values():
            types.update(extraction.get_entity_types())
        return types

    @property
    def all_relevant_tables(self) -> Set[str]:
        """Get all relevant tables across all extractions."""
        tables: Set[str] = set()
        for extraction in self.extractions.values():
            tables.update(extraction.relevant_tables)
        return tables

    # =========================================================================
    # Public Methods - Access
    # =========================================================================

    def get(self, qa_index: int) -> Optional[QAExtraction]:
        """Get extraction for a specific QA pair.

        Args:
            qa_index: Index of the QA pair

        Returns:
            QAExtraction if found, None otherwise
        """
        return self.extractions.get(qa_index)

    def get_relevant_tables(self, qa_index: int) -> Set[str]:
        """Get relevant tables for a specific QA pair.

        This method is used for schema filtering during verification.

        Args:
            qa_index: Index of the QA pair

        Returns:
            Set of table names, empty set if QA not found
        """
        extraction = self.extractions.get(qa_index)
        if extraction:
            return extraction.relevant_tables
        return set()

    def get_all_extractions(self) -> List[QAExtraction]:
        """Get all extractions as a list.

        Returns:
            List of all QAExtraction objects, sorted by qa_index
        """
        return sorted(self.extractions.values(), key=lambda e: e.qa_index)

    def get_invalid_extractions(self) -> List[QAExtraction]:
        """Get all extractions that failed validation.

        Returns:
            List of invalid QAExtraction objects
        """
        return [e for e in self.extractions.values() if e.validation_status == "invalid"]

    def get_forget_extractions(self) -> List[QAExtraction]:
        """Get all extractions from the forget set.

        Returns:
            List of QAExtraction objects where source == "forget"
        """
        return [e for e in self.extractions.values() if e.source == "forget"]

    def get_retain_extractions(self) -> List[QAExtraction]:
        """Get all extractions from the retain set.

        Returns:
            List of QAExtraction objects where source == "retain"
        """
        return [e for e in self.extractions.values() if e.source == "retain"]

    def get_extractions_by_source(self, source: str) -> List[QAExtraction]:
        """Get all extractions with a specific source label.

        Args:
            source: Source label ("retain", "forget", or "unknown")

        Returns:
            List of QAExtraction objects matching the source
        """
        return [e for e in self.extractions.values() if e.source == source]

    # =========================================================================
    # Public Methods - Modification
    # =========================================================================

    def add(self, extraction: QAExtraction) -> None:
        """Add an extraction to the registry.

        Args:
            extraction: The QAExtraction to add
        """
        self.extractions[extraction.qa_index] = extraction

    def update(self, extraction: QAExtraction) -> None:
        """Update an existing extraction in the registry.

        Args:
            extraction: The QAExtraction to update
        """
        self.extractions[extraction.qa_index] = extraction

    def remove(self, qa_index: int) -> Optional[QAExtraction]:
        """Remove an extraction from the registry.

        Args:
            qa_index: Index of the QA pair to remove

        Returns:
            The removed QAExtraction if found, None otherwise
        """
        return self.extractions.pop(qa_index, None)

    # =========================================================================
    # Public Methods - Aggregation
    # =========================================================================

    def merge_entities(self) -> Dict[str, List[Dict[str, Any]]]:
        """Merge all entities from all extractions.

        Returns:
            Dictionary mapping entity_type to list of all entity instances
        """
        merged: Dict[str, List[Dict[str, Any]]] = {}
        for extraction in self.extractions.values():
            for entity_type, entities in extraction.entities.items():
                if entity_type not in merged:
                    merged[entity_type] = []
                merged[entity_type].extend(entities)
        return merged

    def merge_relations(self) -> List[Dict[str, Any]]:
        """Merge all relations from all extractions.

        Returns:
            List of all relation records
        """
        merged: List[Dict[str, Any]] = []
        for extraction in self.extractions.values():
            merged.extend(extraction.relations)
        return merged

    def get_qa_indices_for_table(self, table_name: str) -> Set[int]:
        """Get all QA indices that touch a specific table.

        Args:
            table_name: Name of the table

        Returns:
            Set of QA indices
        """
        indices: Set[int] = set()
        for qa_index, extraction in self.extractions.items():
            if table_name in extraction.relevant_tables:
                indices.add(qa_index)
        return indices

    # =========================================================================
    # Public Methods - Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the extraction registry.

        Returns:
            Dictionary with various statistics
        """
        total_entities: int = sum(e.entity_count for e in self.extractions.values())
        total_relations: int = sum(e.relation_count for e in self.extractions.values())

        return {
            "total_qa_pairs": self.count,
            "valid_extractions": self.valid_count,
            "invalid_extractions": self.count - self.valid_count,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "entity_types": list(self.all_entity_types),
            "tables_touched": list(self.all_relevant_tables),
        }

    # =========================================================================
    # Public Methods - Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all registry data
        """
        return {
            "extractions": {
                str(idx): extraction.to_dict()
                for idx, extraction in self.extractions.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QAExtractionRegistry":
        """Create registry from dictionary.

        Args:
            data: Dictionary with registry data

        Returns:
            New QAExtractionRegistry instance
        """
        registry = cls.empty()
        extractions_data = data.get("extractions", {})
        for idx_str, extraction_data in extractions_data.items():
            extraction = QAExtraction.from_dict(extraction_data)
            registry.add(extraction)
        return registry

    def __iter__(self):
        """Iterate over extractions in qa_index order."""
        for qa_index in sorted(self.extractions.keys()):
            yield self.extractions[qa_index]

    def __len__(self) -> int:
        """Return the number of extractions."""
        return len(self.extractions)
