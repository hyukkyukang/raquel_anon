from .db_construction.schema_coverage_check import SchemaCoverageCheckPrompt
from .db_construction.table_analysis import TableAnalysisPrompt
from .db_construction.table_generation import TableGenerationPrompt
from .db_construction.schema_modification_duplicated_column import (
    SchemaColumnDeduplicationPrompt,
)
from .db_construction.schema_modification_foreign_key import (
    SchemaForeignKeyModificationPrompt,
)
from .db_construction.schema_modification_order import SchemaOrderModificationPrompt
from .db_construction.schema_modification_primary_key import (
    SchemaPrimaryKeyModificationPrompt,
)
from .db_construction.schema_modification_unique_key import (
    SchemaUniqueKeyModificationPrompt,
)
from .db_construction.schema_normalization import SchemaNormalizationPrompt
from .db_construction.schema_unification import SchemaUnificationPrompt
from .db_construction.sql_syntax_correction import SQLSyntaxCorrectionPrompt
from .db_construction.update_null_modification import UpdateNullModificationPrompt
from .db_construction.update_nullify_generation import UpdateNullifyGenerationPrompt
from .db_construction.upsert_null_modification import UpsertNullModificationPrompt
from .paraphrasing import ParaphrasingPrompt
from .perturb_answer import PerturbAnswerPrompt
from .sql_result_to_text.prompt import ResultToTextPrompt
from .sql_synthesis.text_to_sql_prompt import TextToSQLPrompt
from .sql_synthesis.query_generation.prompt import QueryGenerationPrompt
from .sql_synthesis.template_spec_generation import TemplateSpecGenerationPrompt
from .sql_translation.query_translation import QueryTranslationPrompt

# Six-Stage Pipeline Prompts
from .db_construction.entity_type_discovery import EntityTypeDiscoveryPrompt
from .db_construction.entity_type_consolidation import EntityTypeConsolidationPrompt
from .db_construction.attribute_discovery import AttributeDiscoveryPrompt
from .db_construction.attribute_normalization import AttributeNormalizationPrompt
from .db_construction.schema_from_attributes import SchemaFromAttributesPrompt
from .db_construction.round_trip_comparison import RoundTripComparisonPrompt
from .db_construction.round_trip_lookup import RoundTripLookupPrompt
from .db_construction.round_trip_answer import RoundTripAnswerPrompt
from .db_construction.round_trip_judgment import RoundTripJudgmentPrompt
from .db_construction.fix_generation import FixGenerationPrompt
from .db_construction.per_qa_extraction import PerQAExtractionPrompt
from .json_repair import JSONRepairPrompt


SCHEMA_UNIFICATION_PROMPT_REGISTRY = {
    "default": SchemaUnificationPrompt,
}


SCHEMA_FOREIGN_KEY_MODIFICATION_PROMPT_REGISTRY = {
    "default": SchemaForeignKeyModificationPrompt,
}

SCHEMA_UNIQUE_KEY_MODIFICATION_PROMPT_REGISTRY = {
    "default": SchemaUniqueKeyModificationPrompt,
}

SCHEMA_ORDER_MODIFICATION_PROMPT_REGISTRY = {
    "default": SchemaOrderModificationPrompt,
}

SCHEMA_PRIMARY_KEY_MODIFICATION_PROMPT_REGISTRY = {
    "default": SchemaPrimaryKeyModificationPrompt,
}

SCHEMA_NORMALIZATION_PROMPT_REGISTRY = {
    "default": SchemaNormalizationPrompt,
}

SCHEMA_COLUMN_DEDUPLICATION_PROMPT_REGISTRY = {
    "default": SchemaColumnDeduplicationPrompt,
}

SCHEMA_COVERAGE_CHECK_PROMPT_REGISTRY = {
    "default": SchemaCoverageCheckPrompt,
}

TABLE_ANALYSIS_PROMPT_REGISTRY = {
    "default": TableAnalysisPrompt,
}

TABLE_GENERATION_PROMPT_REGISTRY = {
    "default": TableGenerationPrompt,
}

UPSERT_NULLIFY_GENERATION_PROMPT_REGISTRY = {
    "default": UpdateNullifyGenerationPrompt,
}

UPSERT_NULL_MODIFICATION_PROMPT_REGISTRY = {
    "default": UpsertNullModificationPrompt,
}

UPDATE_NULLIFY_GENERATION_PROMPT_REGISTRY = {
    "default": UpdateNullifyGenerationPrompt,
}

UPDATE_NULL_MODIFICATION_PROMPT_REGISTRY = {
    "default": UpdateNullModificationPrompt,
}

SQL_SYNTAX_CORRECTION_PROMPT_REGISTRY = {
    "default": SQLSyntaxCorrectionPrompt,
}

# For Query Synthesis
# All types use the same QueryGenerationPrompt - the type_name is passed as extra_instruction
QUERY_GENERATION_PROMPT_REGISTRY = {
    "default": QueryGenerationPrompt,
    # Single-feature types
    "where": QueryGenerationPrompt,
    "groupby": QueryGenerationPrompt,
    "subquery": QueryGenerationPrompt,
    "join": QueryGenerationPrompt,
    "orderby": QueryGenerationPrompt,
    "having": QueryGenerationPrompt,
    "distinct": QueryGenerationPrompt,
    "like": QueryGenerationPrompt,
    "null_check": QueryGenerationPrompt,
    "between": QueryGenerationPrompt,
    "case_when": QueryGenerationPrompt,
    # Composed types
    "join_groupby": QueryGenerationPrompt,
    "join_where_orderby": QueryGenerationPrompt,
    "subquery_aggregation": QueryGenerationPrompt,
    "union_orderby": QueryGenerationPrompt,
    "groupby_having_orderby": QueryGenerationPrompt,
    "multi_join": QueryGenerationPrompt,
    "exists_subquery": QueryGenerationPrompt,
    "in_subquery": QueryGenerationPrompt,
    "comparison_subquery": QueryGenerationPrompt,
    # Legacy types (kept for backward compatibility)
    "intersection": QueryGenerationPrompt,
    "union": QueryGenerationPrompt,
}

# Template specification prompts
TEMPLATE_SPEC_GENERATION_PROMPT_REGISTRY = {
    "default": TemplateSpecGenerationPrompt,
}

# For SQL to Text
SQL_TO_TEXT_PROMPT_REGISTRY = {
    "default": QueryTranslationPrompt,
}

# For SQL result to Text
RESULT_TO_TEXT_PROMPT_REGISTRY = {
    "default": ResultToTextPrompt,
}

PARAPHRASING_PROMPT_REGISTRY = {
    "default": ParaphrasingPrompt,
}

PERTURB_ANSWER_PROMPT_REGISTRY = {
    "default": PerturbAnswerPrompt,
}

TEXT_TO_SQL_PROMPT_REGISTRY = {
    "default": TextToSQLPrompt,
}

# Six-Stage Pipeline Registries
ENTITY_TYPE_DISCOVERY_PROMPT_REGISTRY = {
    "default": EntityTypeDiscoveryPrompt,
}

ENTITY_TYPE_CONSOLIDATION_PROMPT_REGISTRY = {
    "default": EntityTypeConsolidationPrompt,
}

ATTRIBUTE_DISCOVERY_PROMPT_REGISTRY = {
    "default": AttributeDiscoveryPrompt,
}

ATTRIBUTE_NORMALIZATION_PROMPT_REGISTRY = {
    "default": AttributeNormalizationPrompt,
}

SCHEMA_FROM_ATTRIBUTES_PROMPT_REGISTRY = {
    "default": SchemaFromAttributesPrompt,
}

PER_QA_EXTRACTION_PROMPT_REGISTRY = {
    "default": PerQAExtractionPrompt,
}

ROUND_TRIP_COMPARISON_PROMPT_REGISTRY = {
    "default": RoundTripComparisonPrompt,
}

ROUND_TRIP_LOOKUP_PROMPT_REGISTRY = {
    "default": RoundTripLookupPrompt,
}

ROUND_TRIP_ANSWER_PROMPT_REGISTRY = {
    "default": RoundTripAnswerPrompt,
}

ROUND_TRIP_JUDGMENT_PROMPT_REGISTRY = {
    "default": RoundTripJudgmentPrompt,
}

FIX_GENERATION_PROMPT_REGISTRY = {
    "default": FixGenerationPrompt,
}

# JSON Repair
JSON_REPAIR_PROMPT_REGISTRY = {
    "default": JSONRepairPrompt,
}
