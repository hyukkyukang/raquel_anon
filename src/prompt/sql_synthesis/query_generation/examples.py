"""Few-shot examples for SQL query generation by type.

Each query type has 1-2 example queries that demonstrate the expected
SQL pattern. These examples help the LLM understand the structure
without being tied to any specific schema.
"""

from typing import Dict, List

# Examples use generic table/column names that can apply to any schema
# The LLM should adapt these patterns to the actual schema provided

QUERY_EXAMPLES: Dict[str, List[str]] = {
    # Single-feature types
    "where": [
        "SELECT name, email FROM users WHERE status = 'active';",
        "SELECT title, price FROM products WHERE price > 100 AND category = 'electronics';",
    ],
    "groupby": [
        "SELECT category, COUNT(*) as count FROM products GROUP BY category;",
        "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department;",
    ],
    "subquery": [
        "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees);",
        "SELECT title FROM books WHERE author_id IN (SELECT id FROM authors WHERE country = 'USA');",
    ],
    "join": [
        "SELECT u.name, o.order_date FROM users u JOIN orders o ON u.id = o.user_id;",
        "SELECT p.title, c.name as category FROM products p LEFT JOIN categories c ON p.category_id = c.id;",
    ],
    "orderby": [
        "SELECT name, score FROM students ORDER BY score DESC LIMIT 10;",
        "SELECT title, publication_date FROM articles ORDER BY publication_date DESC LIMIT 5;",
    ],
    "having": [
        "SELECT department, COUNT(*) as emp_count FROM employees GROUP BY department HAVING COUNT(*) > 5;",
        "SELECT category, SUM(sales) as total FROM products GROUP BY category HAVING SUM(sales) > 10000;",
    ],
    "distinct": [
        "SELECT DISTINCT category FROM products;",
        "SELECT COUNT(DISTINCT author_id) as unique_authors FROM books;",
    ],
    "like": [
        "SELECT name, email FROM users WHERE email LIKE '%@gmail.com';",
        "SELECT title FROM books WHERE title ILIKE '%python%';",
    ],
    "null_check": [
        "SELECT name FROM users WHERE phone IS NULL;",
        "SELECT title, description FROM products WHERE description IS NOT NULL;",
    ],
    "between": [
        "SELECT title, price FROM products WHERE price BETWEEN 50 AND 100;",
        "SELECT name, hire_date FROM employees WHERE hire_date BETWEEN '2020-01-01' AND '2023-12-31';",
    ],
    "case_when": [
        "SELECT name, CASE WHEN score >= 90 THEN 'A' WHEN score >= 80 THEN 'B' ELSE 'C' END as grade FROM students;",
        "SELECT title, CASE WHEN price < 20 THEN 'budget' WHEN price < 100 THEN 'mid-range' ELSE 'premium' END as tier FROM products;",
    ],
    # Composed types
    "join_groupby": [
        "SELECT c.name, COUNT(p.id) as product_count FROM categories c JOIN products p ON c.id = p.category_id GROUP BY c.name;",
        "SELECT d.name as department, AVG(e.salary) as avg_salary FROM departments d JOIN employees e ON d.id = e.department_id GROUP BY d.name;",
    ],
    "join_where_orderby": [
        "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id WHERE o.total > 100 ORDER BY o.total DESC;",
        "SELECT a.name, b.title FROM authors a JOIN books b ON a.id = b.author_id WHERE b.year > 2020 ORDER BY b.year DESC LIMIT 10;",
    ],
    "subquery_aggregation": [
        "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees WHERE department_id = employees.department_id);",
        "SELECT category FROM products WHERE price > (SELECT AVG(price) * 1.5 FROM products);",
    ],
    "union_orderby": [
        "SELECT name, 'customer' as type FROM customers UNION SELECT name, 'supplier' as type FROM suppliers ORDER BY name;",
        "SELECT title, 'book' as item_type FROM books UNION SELECT title, 'article' as item_type FROM articles ORDER BY title;",
    ],
    "groupby_having_orderby": [
        "SELECT category, COUNT(*) as cnt FROM products GROUP BY category HAVING COUNT(*) > 3 ORDER BY cnt DESC;",
        "SELECT author_id, COUNT(*) as book_count FROM books GROUP BY author_id HAVING COUNT(*) >= 2 ORDER BY book_count DESC LIMIT 5;",
    ],
    "multi_join": [
        "SELECT a.name, b.title, p.name as publisher FROM authors a JOIN books b ON a.id = b.author_id JOIN publishers p ON b.publisher_id = p.id;",
        "SELECT e.name, d.name as dept, m.name as manager FROM employees e JOIN departments d ON e.department_id = d.id JOIN employees m ON e.manager_id = m.id;",
    ],
    "exists_subquery": [
        "SELECT name FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.total > 500);",
        "SELECT title FROM books b WHERE NOT EXISTS (SELECT 1 FROM reviews r WHERE r.book_id = b.id);",
    ],
    "in_subquery": [
        "SELECT name FROM employees WHERE department_id IN (SELECT id FROM departments WHERE location = 'NYC');",
        "SELECT title FROM books WHERE author_id NOT IN (SELECT id FROM authors WHERE country = 'UK');",
    ],
    "comparison_subquery": [
        "SELECT name, salary FROM employees WHERE salary > ALL (SELECT salary FROM employees WHERE department_id = 2);",
        "SELECT title FROM products WHERE price < ANY (SELECT price FROM products WHERE category = 'premium');",
    ],
}


def get_examples_for_type(type_name: str) -> List[str]:
    """Get example queries for a specific type.

    Args:
        type_name: The query type (e.g., 'join', 'groupby')

    Returns:
        List of example SQL queries for this type
    """
    return QUERY_EXAMPLES.get(type_name, [])


def format_examples_for_prompt(type_name: str) -> str:
    """Format examples for inclusion in a prompt.

    Args:
        type_name: The query type

    Returns:
        Formatted string with examples, or empty string if no examples
    """
    examples = get_examples_for_type(type_name)
    if not examples:
        return ""

    lines = ["=== EXAMPLE QUERIES ===", f"(Examples for '{type_name}' type queries)", ""]
    for i, example in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(example)
        lines.append("")

    lines.append("Generate similar queries adapted to the schema above.")
    lines.append("=" * 30)

    return "\n".join(lines)
