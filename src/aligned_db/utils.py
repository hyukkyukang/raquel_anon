import re
from typing import Dict, List, Optional

import sqlparse
from sqlparse.sql import Identifier, Parenthesis
from sqlparse.tokens import Keyword


# -------------------------------------------------
# helper: split a string on top-level commas only
# -------------------------------------------------
def _split_top_level(text: str) -> List[str]:
    parts, cur, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


# -------------------------------------------------
# main routine
# -------------------------------------------------
def get_schema_stats(schema: str) -> Dict[str, float]:
    """
    Return basic statistics for a collection of CREATE TABLE statements.

    * num_tables               – number of CREATE TABLE statements
    * num_columns              – total number of *column* definitions
    * avg_columns_per_table    – arithmetic mean (float, 2 dp)
    * max_columns_per_table    – largest column count in a single table
    * min_columns_per_table    – smallest column count in a single table
    """
    # split at the SQL-statement level (sqlparse is parenthesis-aware)
    statements = [s.strip() for s in sqlparse.split(schema) if s.strip()]
    create_tbls = [s for s in statements if s.upper().startswith("CREATE TABLE")]
    num_tables = len(create_tbls)

    # keywords that mark table-level constraints, *not* columns
    constraint_kw = ("PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT")

    total_cols, cols_per_table = 0, []

    for stmt in create_tbls:
        # -------------------------------------------------
        # 1. extract the part inside the outermost (…) pair
        # -------------------------------------------------
        start = stmt.find("(")
        if start == -1:
            cols_per_table.append(0)
            continue

        depth = 0
        for i in range(start, len(stmt)):
            if stmt[i] == "(":
                depth += 1
            elif stmt[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        body = stmt[start + 1 : end]  # everything between the parentheses

        # -------------------------------------------------
        # 2. split on *top-level* commas
        # -------------------------------------------------
        items = _split_top_level(body)

        # -------------------------------------------------
        # 3. keep only genuine column definitions
        # -------------------------------------------------
        cols = [
            itm
            for itm in items
            if itm and not itm.lstrip().upper().startswith(constraint_kw)  # not empty
        ]

        cols_per_table.append(len(cols))
        total_cols += len(cols)

    stats = {
        "num_tables": num_tables,
        "num_columns": total_cols,
        "avg_columns_per_table": round(total_cols / num_tables, 2) if num_tables else 0,
        "max_columns_per_table": max(cols_per_table) if cols_per_table else 0,
        "min_columns_per_table": min(cols_per_table) if cols_per_table else 0,
    }
    return stats


# ---------- helper ----------
def split_top_level(text: str):
    """Split on commas that are *not* inside parentheses."""
    parts, current, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


# ---------- /helper ----------


def extract_create_table_statements(sql: str):
    """
    Yield (table_name, [column | constraint, …]) tuples for every
    CREATE TABLE statement in *sql*.
    """
    for stmt in sqlparse.parse(sql):
        if stmt.get_type() != "CREATE":
            continue

        # 1) Table name ------------------------------------------------------
        tbl_name = None
        for tok in stmt.tokens:
            if tok.ttype is Keyword and tok.value.upper() == "TABLE":
                _, next_tok = stmt.token_next(stmt.token_index(tok), skip_ws=True)
                tbl_name = (
                    next_tok.get_real_name()  # Identifier
                    if isinstance(next_tok, Identifier)
                    else next_tok.value  # plain Token
                )
                break
        if not tbl_name:
            continue  # malformed

        # 2) Column/constraint list ------------------------------------------
        cols = []
        for tok in stmt.tokens:
            if isinstance(tok, Parenthesis):
                inside = tok.value[1:-1]  # strip the ( … )
                cols = [c for c in split_top_level(inside) if c]
                break

        yield tbl_name, cols


CONSTRAINT_KW = {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT", "KEY", "INDEX"}
COL_NAME_RE = re.compile(
    r"""\s*
        (                       # capture column identifier
          "(?:[^"]+)"           # "double-quoted id"
        | `[^`]+`               # `back-quoted id`
        | $begin:math:display$[^$end:math:display$]+\]            # [bracket id]
        | [A-Za-z_][A-Za-z0-9_]*  # unquoted id
        )
    """,
    re.VERBOSE,
)


def _get_identifier(defn: str) -> Optional[str]:
    """Return the *lower-cased* column identifier, or None if `defn` is a constraint."""
    first_tok = defn.lstrip().split(None, 1)[0].upper()
    if first_tok in CONSTRAINT_KW:
        return None  # table-level constraint
    m = COL_NAME_RE.match(defn)
    return m.group(1).strip('`"[]').lower() if m else None


def merge_create_tables(sql1: str, sql2: str) -> str:
    """
    Union-merge two SQL strings’ CREATE TABLE definitions, deduplicating columns
    that share the same identifier (case-insensitive).  If two definitions for
    the same column differ, the longer one (i.e., more constraints) is kept.
    """
    catalog: dict[str, dict[str, str] | list[str]] = {}

    for sql in (sql1, sql2):
        for tbl, defs in extract_create_table_statements(sql):
            entry = catalog.setdefault(
                tbl, {"cols": {}, "order": [], "constraints": []}
            )

            for d in defs:
                col_id = _get_identifier(d)
                if col_id is None:  # table-level constraint
                    if d not in entry["constraints"]:
                        entry["constraints"].append(d)
                    continue

                prev = entry["cols"].get(col_id)
                if prev is None:
                    entry["cols"][col_id] = d
                    entry["order"].append(col_id)  # preserve appearance order
                else:
                    # Choose the 'richer' definition
                    if len(d) > len(prev):
                        entry["cols"][col_id] = d

    stmts = []
    for tbl, parts in catalog.items():
        col_defs = [parts["cols"][cid] for cid in parts["order"]]
        body_lines = col_defs + parts["constraints"]
        body = ",\n    ".join(body_lines)
        stmts.append(f"CREATE TABLE {tbl} (\n    {body}\n);")

    return "\n\n".join(stmts)


# ---------------------------------------------------------------------------
# 1) Build a {table: [column_name, …]} map from CREATE TABLE statements
# ---------------------------------------------------------------------------


def _extract_table_columns(schema_sql: str) -> Dict[str, List[str]]:
    """Return {table_name: [column_name, …]} from CREATE TABLE statements."""
    catalog: Dict[str, List[str]] = {}

    for stmt in sqlparse.parse(schema_sql):
        if stmt.get_type() != "CREATE":
            continue

        # -------- table name --------
        tbl = None
        for tok in stmt.tokens:
            if tok.ttype is Keyword and tok.value.upper() == "TABLE":
                _, nxt = stmt.token_next(stmt.token_index(tok), skip_ws=True)
                tbl = nxt.get_real_name() if isinstance(nxt, Identifier) else nxt.value
                break
        if not tbl:
            continue

        # -------- column definitions inside (...) --------
        col_names: List[str] = []
        for tok in stmt.tokens:
            if isinstance(tok, Parenthesis):
                body = tok.value[1:-1]  # strip outer parens
                for defn in split_top_level(body):
                    head = defn.split(None, 1)[0].strip('"`')  # column or keyword
                    # skip table-level constraints
                    if head.upper() in {
                        "PRIMARY",
                        "FOREIGN",
                        "UNIQUE",
                        "CHECK",
                        "CONSTRAINT",
                    }:
                        continue
                    col_names.append(head)
                break

        catalog[tbl] = col_names
    return catalog


# ---------------------------------------------------------------------------
# 2) UPSERT patching
# ---------------------------------------------------------------------------

# Regex that isolates the critical pieces of ONE upsert statement
_UPSERT_RE = re.compile(
    r"""(?ix)                                    # verbose, case-insensitive
    \bINSERT \s+ INTO \s+ (?P<table>[^\s(]+)      # table name
    \s*$begin:math:text$ (?P<cols>.*?) $end:math:text$\s*                     # (col, col, …)
    VALUES\s* (?P<vals>                           # every row: (...) , (...)
        $begin:math:text$ .*? $end:math:text$ (?: \s*,\s* $begin:math:text$ .*? $end:math:text$ )*
    )
    \s* ON \s+ CONFLICT \s* (?P<conflict> .+?)    # conflict clause
    \s* ;?                                        # optional semicolon
    """,
    re.DOTALL,
)


def _patch_single_upsert(stmt: str, table_cols: List[str]) -> str:
    """Return modified *stmt* (one UPSERT) so that it covers *table_cols*."""
    m = _UPSERT_RE.search(stmt)
    if not m:
        return stmt  # not an UPSERT → untouched

    tbl = m.group("table")
    cols_raw = m.group("cols")
    vals_raw = m.group("vals")
    confl = m.group("conflict").strip()

    given_cols = [c.strip() for c in split_top_level(cols_raw)]
    missing = [c for c in table_cols if c not in given_cols]
    if not missing:
        return stmt  # already complete

    # --- 1) columns list ------------------------------------------------
    new_cols_raw = ",\n    ".join(given_cols + missing)

    # --- 2) VALUES (…) rows ---------------------------------------------
    def _patch_row(row: str) -> str:
        inner = row.strip()
        if inner.startswith("(") and inner.endswith(")"):
            inner = inner[1:-1]
        patched_vals = split_top_level(inner) + ["DEFAULT"] * len(missing)
        return "(\n        " + ",\n        ".join(patched_vals) + "\n    )"

    rows = split_top_level(vals_raw[1:-1])  # strip outermost paren of all rows
    new_vals_raw = ", ".join(_patch_row(r) for r in rows)

    # --- 3) ON CONFLICT clause ------------------------------------------
    if confl.upper().startswith("DO NOTHING"):
        new_confl = confl
    else:
        set_m = re.search(r"\bSET\b\s*(.*)$", confl, flags=re.I | re.S)
        if set_m:
            assign_raw = set_m.group(1).strip()
            assigned = [a.split("=")[0].strip() for a in split_top_level(assign_raw)]
            extra_assign = [f"{c} = EXCLUDED.{c}" for c in missing if c not in assigned]
            if extra_assign:
                joined = (
                    assign_raw.rstrip().rstrip(",")
                    + (", " if assign_raw else "")
                    + ", ".join(extra_assign)
                )
                new_confl = re.sub(
                    r"\bSET\b\s*.*$", "SET " + joined, confl, flags=re.I | re.S
                )
            else:
                new_confl = confl
        else:
            new_confl = confl  # unexpected format

    # --- 4) rebuild statement -------------------------------------------
    return (
        f"INSERT INTO {tbl} (\n    {new_cols_raw}\n)"
        f"\nVALUES {new_vals_raw}\nON CONFLICT {new_confl};"
    )


def patch_upserts(upsert_sql: str, schema_sql: str) -> str:
    """
    Return *upsert_sql* with every UPSERT extended so all columns defined
    in *schema_sql* are handled (DEFAULT in VALUES, assignment in SET).
    """
    schema_map = _extract_table_columns(schema_sql)

    patched_statements = []
    for stmt in sqlparse.split(upsert_sql):
        # detect 'INSERT INTO mytable' to see if we need patching
        tbl_m = re.search(r"\bINSERT\s+INTO\s+([^\s(]+)", stmt, flags=re.I)
        if tbl_m:
            tbl = tbl_m.group(1)
            if tbl in schema_map:
                stmt = _patch_single_upsert(stmt, schema_map[tbl])
        patched_statements.append(stmt.strip())

    return "\n\n".join(patched_statements)
