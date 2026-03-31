"""Query templates for complex SQL types.

This module provides structured templates for complex query types to reduce
LLM hallucination errors. Instead of free-form generation, the LLM fills
slots in predefined templates with valid table/column names.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("src.generator.query_templates")


@dataclass
class QueryTemplate:
    """A structured SQL query template with fillable slots."""

    name: str
    template: str
    slots: List[str]
    required_slots: List[str]
    description: str

    def fill(self, slot_values: Dict[str, str]) -> str:
        """Fill template slots with values.

        Args:
            slot_values: Dictionary mapping slot names to values

        Returns:
            Filled SQL query string

        Raises:
            ValueError: If required slots are missing
        """
        # Check required slots
        missing = set(self.required_slots) - set(slot_values.keys())
        if missing:
            raise ValueError(f"Missing required slots: {missing}")

        # Fill template
        result = self.template
        for slot, value in slot_values.items():
            placeholder = "{" + slot + "}"
            result = result.replace(placeholder, value)

        # Remove unfilled optional placeholders
        result = re.sub(r"\{[a-z_]+\}", "", result)

        # Clean up extra whitespace and empty lines
        result = "\n".join(line for line in result.split("\n") if line.strip())

        return result.strip()


# Templates for complex query types
QUERY_TEMPLATES: Dict[str, QueryTemplate] = {
    "multi_join": QueryTemplate(
        name="multi_join",
        template="""SELECT {select_columns}
FROM {table1} {alias1}
JOIN {table2} {alias2} ON {join_condition1}
JOIN {table3} {alias3} ON {join_condition2}
{where_clause}
{order_by_clause}
LIMIT {limit};""",
        slots=[
            "select_columns",
            "table1",
            "alias1",
            "table2",
            "alias2",
            "table3",
            "alias3",
            "join_condition1",
            "join_condition2",
            "where_clause",
            "order_by_clause",
            "limit",
        ],
        required_slots=[
            "select_columns",
            "table1",
            "alias1",
            "table2",
            "alias2",
            "table3",
            "alias3",
            "join_condition1",
            "join_condition2",
        ],
        description="Query joining 3+ tables with meaningful conditions",
    ),
    "subquery_aggregation": QueryTemplate(
        name="subquery_aggregation",
        template="""SELECT {select_columns}
FROM {main_table} {main_alias}
WHERE {compare_column} {operator} (
    SELECT {agg_function}({agg_column})
    FROM {subquery_table}
    {subquery_where}
)
{order_by_clause}
LIMIT {limit};""",
        slots=[
            "select_columns",
            "main_table",
            "main_alias",
            "compare_column",
            "operator",
            "agg_function",
            "agg_column",
            "subquery_table",
            "subquery_where",
            "order_by_clause",
            "limit",
        ],
        required_slots=[
            "select_columns",
            "main_table",
            "compare_column",
            "operator",
            "agg_function",
            "agg_column",
            "subquery_table",
        ],
        description="Query with subquery containing aggregation function",
    ),
    "exists_subquery": QueryTemplate(
        name="exists_subquery",
        template="""SELECT {select_columns}
FROM {main_table} {main_alias}
WHERE {exists_keyword} (
    SELECT 1
    FROM {subquery_table} {subquery_alias}
    WHERE {correlation_condition}
    {subquery_filter}
)
{order_by_clause}
LIMIT {limit};""",
        slots=[
            "select_columns",
            "main_table",
            "main_alias",
            "exists_keyword",
            "subquery_table",
            "subquery_alias",
            "correlation_condition",
            "subquery_filter",
            "order_by_clause",
            "limit",
        ],
        required_slots=[
            "select_columns",
            "main_table",
            "main_alias",
            "exists_keyword",
            "subquery_table",
            "subquery_alias",
            "correlation_condition",
        ],
        description="Query using EXISTS or NOT EXISTS with correlated subquery",
    ),
    "comparison_subquery": QueryTemplate(
        name="comparison_subquery",
        template="""SELECT {select_columns}
FROM {main_table} {main_alias}
WHERE {compare_column} {operator} {quantifier} (
    SELECT {subquery_column}
    FROM {subquery_table} {subquery_alias}
    {subquery_where}
)
{order_by_clause}
LIMIT {limit};""",
        slots=[
            "select_columns",
            "main_table",
            "main_alias",
            "compare_column",
            "operator",
            "quantifier",
            "subquery_column",
            "subquery_table",
            "subquery_alias",
            "subquery_where",
            "order_by_clause",
            "limit",
        ],
        required_slots=[
            "select_columns",
            "main_table",
            "compare_column",
            "operator",
            "quantifier",
            "subquery_column",
            "subquery_table",
        ],
        description="Query comparing values against subquery results using >, <, =, ALL, ANY",
    ),
}

# Types that should use templates instead of free-form generation
TEMPLATE_TYPES: Set[str] = {
    "multi_join",
    "subquery_aggregation",
    "exists_subquery",
    "comparison_subquery",
}


def get_template(type_name: str) -> Optional[QueryTemplate]:
    """Get template for a query type.

    Args:
        type_name: The query type name

    Returns:
        QueryTemplate if type uses templates, None otherwise
    """
    return QUERY_TEMPLATES.get(type_name)


def is_template_type(type_name: str) -> bool:
    """Check if a query type uses templates.

    Args:
        type_name: The query type name

    Returns:
        True if type should use template-based generation
    """
    return type_name in TEMPLATE_TYPES


def get_slot_filling_prompt(
    template: QueryTemplate,
    schema_info: str,
    table_data_sample: str,
) -> str:
    """Generate a prompt for LLM to fill template slots.

    Args:
        template: The query template to fill
        schema_info: Schema information with tables and columns
        table_data_sample: Sample data from tables

    Returns:
        Prompt string for LLM
    """
    prompt = f"""Fill in the slots for this SQL query template.

TEMPLATE TYPE: {template.name}
DESCRIPTION: {template.description}

TEMPLATE:
{template.template}

SLOTS TO FILL:
Required: {', '.join(template.required_slots)}
Optional: {', '.join(set(template.slots) - set(template.required_slots))}

SCHEMA:
{schema_info}

SAMPLE DATA:
{table_data_sample}

INSTRUCTIONS:
1. Use ONLY tables and columns that exist in the schema above
2. Create a meaningful analytical query
3. Return a JSON object with slot names as keys and values as strings
4. For optional slots you don't need, omit them from the JSON

EXAMPLE OUTPUT FORMAT:
{{
    "select_columns": "t1.name, COUNT(*) as count",
    "table1": "person",
    "alias1": "t1",
    ...
}}

Return ONLY the JSON object, no other text."""

    return prompt


def parse_slot_values(llm_response: str) -> Optional[Dict[str, str]]:
    """Parse slot values from LLM response.

    Args:
        llm_response: Raw LLM response containing JSON

    Returns:
        Dictionary of slot values, or None if parsing fails
    """
    try:
        # Try to extract JSON from response
        # Handle case where LLM wraps in markdown code blocks
        response = llm_response.strip()
        if response.startswith("```"):
            # Remove markdown code block
            lines = response.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            response = "\n".join(json_lines)

        # Parse JSON
        slot_values = json.loads(response)

        # Ensure all values are strings
        return {k: str(v) for k, v in slot_values.items()}

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse slot values from LLM response: {e}")
        return None


def validate_slot_values(
    slot_values: Dict[str, str],
    template: QueryTemplate,
    available_tables: List[str],
    available_columns: Dict[str, List[str]],
) -> Tuple[bool, Optional[str]]:
    """Validate that slot values reference existing tables/columns.

    Args:
        slot_values: Dictionary of slot values
        template: The template being filled
        available_tables: List of available table names
        available_columns: Dict mapping table names to column lists

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required slots are present
    missing = set(template.required_slots) - set(slot_values.keys())
    if missing:
        return False, f"Missing required slots: {missing}"

    # Check table references
    table_slots = ["table1", "table2", "table3", "main_table", "subquery_table"]
    for slot in table_slots:
        if slot in slot_values:
            table_name = slot_values[slot].lower()
            if table_name not in [t.lower() for t in available_tables]:
                return False, f"Invalid table '{slot_values[slot]}' in slot '{slot}'"

    # Basic validation passed
    return True, None
