"""Neon PostgreSQL tools for answering questions about the mock store database.

Credentials are read from .env (prefer DATABASE_READONLY_URL, then DATABASE_URL /
PG*). Only read-only SELECT / WITH queries are allowed. Defense in depth:

1. Semantic READ/WRITE intent guard (before SQL generation)
2. Hardened SELECT-only SQL generation prompts
3. Mandatory sqlglot AST validator (independent of the LLM)
4. Structured tool results as the sole source of truth for answers
5. Prefer a SELECT-only DB role so the server rejects writes even if all else fails

Schema for SQL generation is loaded from:
  src/agent/data/solar_store_schema.sql
Refresh that file with: python scripts/export_store_schema.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langchain.tools import tool

from agent.async_utils import run_in_thread
from agent.custom_tools.db_audit_log import log_db_security_event
from agent.custom_tools.sql_readonly_validator import validate_readonly_sql_ast

load_dotenv()

MAX_ROWS = 50

_SCHEMA_FILE_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "data" / "solar_store_schema.sql",
    Path(__file__).resolve().parents[3] / "scripts" / "sql" / "solar_store_schema.sql",
    Path.cwd() / "src" / "agent" / "data" / "solar_store_schema.sql",
    Path.cwd() / "scripts" / "sql" / "solar_store_schema.sql",
)

# Legacy keyword belt — kept behind AST validation as defense-in-depth.
_FORBIDDEN_SQL = re.compile(
    r"(?:"
    r"\b(?:"
    r"insert|update|delete|drop|alter|create|truncate|grant|revoke|"
    r"copy|call|execute|merge|attach|detach|vacuum|"
    r"reindex|cluster|listen|notify|"
    r"prepare|deallocate|discard|refresh|security|owner|"
    r"begin|commit|rollback"
    r")\b"
    r"|replace\s+into\b"
    r"|\bdo\s+\$\$"
    r"|\bset\s+(?:session|local|role|search_path)\b"
    r"|\bcreate\s+policy\b"
    r")",
    re.IGNORECASE,
)

SQL_GENERATOR_SYSTEM = (
    "You are the Solar Store SQL QUERY WRITER.\n"
    "Your ONLY job is to write READ SQL — never write SQL.\n"
    "\n"
    "ALLOWED (read-only):\n"
    "- Exactly one statement: SELECT … or WITH … SELECT …\n"
    "- Aggregations (SUM/COUNT/AVG), JOINs, WHERE, GROUP BY, ORDER BY, LIMIT\n"
    "\n"
    "FORBIDDEN (must never appear):\n"
    "- UPDATE, INSERT, DELETE, ALTER, DROP, CREATE, TRUNCATE, MERGE\n"
    "- CALL, EXECUTE, COPY, DO, GRANT, REVOKE, BEGIN, COMMIT, ROLLBACK\n"
    "- Multiple statements, SELECT INTO, writable CTEs, FOR UPDATE\n"
    "\n"
    "Rules:\n"
    "- The database is PERMANENTLY READ-ONLY.\n"
    "- Output ONLY the SELECT/WITH query text (no markdown, no commentary).\n"
    "- If the user asks to change/correct/sync/import/delete data, reply exactly: "
    "REFUSE_WRITE\n"
    "- Never claim data was modified. Never invent query results — the SQL tool "
    "is the only source of truth.\n"
)

_FALLBACK_SCHEMA_HINT = """
Solar Store mock schema (PostgreSQL / Neon) — READ SQL ONLY
============================================================
POLICY: Query writers may emit ONLY SELECT or WITH ... SELECT.
Never UPDATE / INSERT / DELETE / DDL. Writes are rejected by the agent.

stores(id, name, city, address, phone)
categories(id, name, description)
products(id, name, category_id, sku, price, cost, stock_qty, unit, is_active, description)
customers(id, full_name, email, phone, city, joined_at)
employees(id, full_name, role, store_id, email, hired_at, salary)
orders(id, customer_id, store_id, employee_id, order_date, status, payment_method, total_amount)
order_items(id, order_id, product_id, quantity, unit_price, line_total)

Useful joins: products.category_id → categories.id,
orders.customer_id → customers.id, order_items.order_id → orders.id,
order_items.product_id → products.id, employees.store_id → stores.id.
""".strip()


def store_schema_path() -> Path | None:
    """Return the first existing Solar Store schema file path."""
    for path in _SCHEMA_FILE_CANDIDATES:
        if path.is_file():
            return path
    return None


def load_store_schema(*, reload: bool = False) -> str:
    """Load schema text from solar_store_schema.sql for the LLM to write SQL."""
    del reload  # always read from disk so file edits are picked up
    path = store_schema_path()
    if path is None:
        return _FALLBACK_SCHEMA_HINT
    text = path.read_text(encoding="utf-8").strip()
    return text or _FALLBACK_SCHEMA_HINT


def _schema_output_paths() -> list[Path]:
    """Paths that should receive a refreshed schema dump."""
    primary = Path(__file__).resolve().parents[1] / "data" / "solar_store_schema.sql"
    scripts_copy = (
        Path(__file__).resolve().parents[3] / "scripts" / "sql" / "solar_store_schema.sql"
    )
    paths = [primary, scripts_copy]
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def refresh_store_schema() -> str:
    """Pull the live Neon schema into solar_store_schema.sql and return it.

    Call this before the LLM writes SQL so table/column definitions stay current.
    Falls back to the on-disk file (or built-in hint) if Neon is unreachable.
    """
    try:
        import psycopg
    except ImportError:
        return load_store_schema()

    try:
        url = _build_database_url(prefer_unpooled=True)
        lines: list[str] = [
            "-- Solar Store schema exported from Neon (neondb / public)",
            "-- Auto-refreshed before each store SQL generation.",
            "-- Manual refresh: python scripts/export_store_schema.py",
            "",
        ]
        with psycopg.connect(url, connect_timeout=20) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """
                )
                tables = [row[0] for row in cur.fetchall()]
                if not tables:
                    return load_store_schema()

                hint_lines: list[str] = []
                for table in tables:
                    cur.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = %s
                        ORDER BY ordinal_position
                        """,
                        (table,),
                    )
                    cols = [row[0] for row in cur.fetchall()]
                    hint_lines.append(f"--   {table}({', '.join(cols)})")

                cur.execute(
                    """
                    SELECT
                      tc.table_name,
                      kcu.column_name,
                      ccu.table_name AS foreign_table,
                      ccu.column_name AS foreign_column
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                     AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                      AND tc.table_schema = 'public'
                    ORDER BY tc.table_name, kcu.ordinal_position
                    """
                )
                fks = cur.fetchall()

                lines.extend(
                    [
                        "-- =============================================================================",
                        "-- AGENT QUERY GUIDE (read this first when writing SQL)",
                        "-- =============================================================================",
                        "-- Tables:",
                        *hint_lines,
                        "--",
                        "-- Joins:",
                    ]
                )
                for table, col, ftable, fcol in fks:
                    lines.append(f"--   {table}.{col} → {ftable}.{fcol}")
                lines.extend(
                    [
                        "--",
                        "-- QUERY WRITER POLICY (mandatory):",
                        "--   You may write ONLY read SQL: a single SELECT or WITH ... SELECT.",
                        "--   FORBIDDEN: UPDATE, INSERT, DELETE, ALTER, DROP, CREATE, TRUNCATE,",
                        "--   MERGE, CALL, EXECUTE, COPY, GRANT, REVOKE, BEGIN, COMMIT, ROLLBACK.",
                        "--   If the user asks to change data → do not emit SQL (REFUSE_WRITE).",
                        "-- Tips:",
                        "--   Low stock  → ORDER BY products.stock_qty ASC",
                        "--   Revenue    → SUM(orders.total_amount) / SUM(order_items.line_total)",
                        "-- =============================================================================",
                        "",
                    ]
                )

                for table in tables:
                    cur.execute(
                        """
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = %s
                        ORDER BY ordinal_position
                        """,
                        (table,),
                    )
                    cols = cur.fetchall()
                    cur.execute(
                        f'SELECT COUNT(*) FROM "{table}"'  # noqa: S608 — identifier from info_schema
                    )
                    count = cur.fetchone()[0]
                    lines.append(f"-- TABLE: {table} ({count} rows)")
                    lines.append(f"CREATE TABLE {table} (")
                    col_defs = []
                    for name, dtype, nullable, default in cols:
                        null_sql = "NULL" if nullable == "YES" else "NOT NULL"
                        default_sql = f" DEFAULT {default}" if default else ""
                        col_defs.append(f"    {name} {dtype} {null_sql}{default_sql}")
                    lines.append(",\n".join(col_defs))
                    lines.append(");")
                    lines.append("")

                if fks:
                    lines.append("-- FOREIGN KEYS")
                    for table, col, ftable, fcol in fks:
                        lines.append(f"-- {table}.{col} → {ftable}.{fcol}")
                    lines.append("")

        text = "\n".join(lines) + "\n"
        for path in _schema_output_paths():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text.strip()
    except Exception:  # noqa: BLE001 — keep answering with last-known schema
        return load_store_schema()


# Back-compat name used across the agent (loaded from file when present).
STORE_SCHEMA_HINT = load_store_schema()
_STORE_SCHEMA_HINT = STORE_SCHEMA_HINT

STORE_QUERY_KEYWORDS = (
    "stock",
    "inventory",
    "product",
    "products",
    "customer",
    "customers",
    "order",
    "orders",
    "sales",
    "revenue",
    "profit",
    "profits",
    "margin",
    "margins",
    "statistics",
    "statistic",
    "stats",
    "analytics",
    "analysis",
    "analyze",
    "analyse",
    "kpi",
    "kpis",
    "business performance",
    "decision making",
    "this month",
    "monthly",
    "quarter",
    "employee",
    "employees",
    "sku",
    "category",
    "categories",
    "solar store",
    "store database",
    "neondb",
    "how many products",
    "top selling",
    "best selling",
    "low in stock",
    "out of stock",
    "in stock",
    "price of",
    "prices",
)


def needs_store_analytics(text: str) -> bool:
    """True when the user wants profit / KPI / business decision analytics from Neon."""
    lowered = (text or "").lower()
    if not lowered.strip():
        return False
    analytics_signals = (
        "profit",
        "margin",
        "revenue",
        "sales",
        "statistics",
        "statistic",
        "stats",
        "analytics",
        "analysis",
        "analyze",
        "analyse",
        "kpi",
        "business performance",
        "decision making",
        "this month",
        "monthly sales",
        "monthly revenue",
        "monthly profit",
    )
    return any(signal in lowered for signal in analytics_signals) and needs_store_database(
        text
    )


def needs_store_database(text: str) -> bool:
    """Detect questions that must be answered from the Solar Store Neon DB."""
    lowered = (text or "").lower()
    if not lowered.strip():
        return False
    # Customer how-to / policy questions belong to business RAG, not SQL.
    if any(
        phrase in lowered
        for phrase in (
            "near me",
            "nearby",
            "closest store",
            "nearest store",
            "around me",
            "what should i do",
            "how do i",
            "how can i",
            "want to replace",
            "want to return",
            "want to exchange",
            "replace a product",
            "return a product",
            "exchange a product",
            "replace the product",
            "return the product",
        )
    ):
        return False
    return any(keyword in lowered for keyword in STORE_QUERY_KEYWORDS)


def _build_database_url(*, prefer_unpooled: bool = False) -> str:
    """Build a Neon connection URL.

    Prefer DATABASE_READONLY_URL (SELECT-only role) for tool queries so the
    database itself rejects writes even if application checks fail.
    """
    readonly = (os.getenv("DATABASE_READONLY_URL") or "").strip()
    if readonly and ":@" not in readonly and not prefer_unpooled:
        return readonly

    url = (os.getenv("DATABASE_URL") or "").strip()
    # Prefer full DATABASE_URL when it includes a real password.
    if url and ":@" not in url and not prefer_unpooled:
        return url

    password = (os.getenv("PGPASSWORD") or "").strip()
    # Optional dedicated read-only role credentials
    user = (
        (os.getenv("PGUSER_READONLY") or "").strip()
        or (os.getenv("PGUSER") or "neondb_owner").strip()
    )
    host = (
        (os.getenv("PGHOST_UNPOOLED") if prefer_unpooled else None)
        or os.getenv("PGHOST")
        or os.getenv("PGHOST_UNPOOLED")
        or ""
    ).strip()
    database = (os.getenv("PGDATABASE") or "neondb").strip()
    sslmode = (os.getenv("PGSSLMODE") or "require").strip()
    channel_binding = (os.getenv("PGCHANNELBINDING") or "require").strip()
    ro_password = (os.getenv("PGPASSWORD_READONLY") or password).strip()

    if host and ro_password:
        return (
            f"postgresql://{quote_plus(user)}:{quote_plus(ro_password)}"
            f"@{host}/{database}?sslmode={sslmode}&channel_binding={channel_binding}"
        )

    if url and ":@" not in url:
        # Swap pooler host for unpooled when seeding/DDL.
        if prefer_unpooled:
            unpooled = (os.getenv("PGHOST_UNPOOLED") or "").strip()
            pooled = (os.getenv("PGHOST") or "").strip()
            if unpooled and pooled and pooled in url:
                return url.replace(pooled, unpooled)
        return url

    if not password:
        raise ValueError(
            "PGPASSWORD is empty. Paste your Neon database password into .env "
            "(PGPASSWORD), then restart the agent."
        )
    raise ValueError(
        "Database not configured. Set DATABASE_READONLY_URL or DATABASE_URL / "
        "PGHOST/PGDATABASE/PGUSER/PGPASSWORD in .env."
    )


def _strip_sql_literals(sql: str) -> str:
    """Remove quoted string literals before keyword safety checks."""
    without_single = re.sub(r"'(?:''|[^'])*'", "''", sql)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', without_single)


def _validate_readonly_sql(sql: str) -> str:
    """Mandatory read-only gate: AST validator first, keyword belt second."""
    cleaned = validate_readonly_sql_ast(sql)

    if _FORBIDDEN_SQL.search(_strip_sql_literals(cleaned)):
        # Allow SELECT/WITH that mention forbidden tokens only inside identifiers
        # is rare; AST already enforced SELECT root. Extra belt for DDL verbs.
        lowered = cleaned.lstrip().lower()
        if not (lowered.startswith("select") or lowered.startswith("with")):
            raise ValueError(
                "Query rejected: mutating or administrative SQL is not allowed. "
                "Use SELECT / WITH only."
            )
        # If AST passed but keyword belt hits UPDATE etc. in a CTE name edge case,
        # still reject when a mutating keyword appears as a statement verb-like form.
        if re.search(
            r"(?i)\b(insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+table|"
            r"alter\s+table|truncate\s+table|create\s+table|merge\s+into|"
            r"begin\b|commit\b|rollback\b|grant\b|revoke\b|copy\s+)",
            _strip_sql_literals(cleaned),
        ):
            raise ValueError(
                "Query rejected: mutating or administrative SQL is not allowed. "
                "Use SELECT / WITH only."
            )
    return cleaned


def _format_rows(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    if not rows:
        return "Query returned 0 rows."

    header = " | ".join(columns)
    separator = "-|-".join("-" * max(len(col), 3) for col in columns)
    lines = [header, separator]
    for row in rows:
        lines.append(" | ".join("" if value is None else str(value) for value in row))
    return "\n".join(lines)


def _tool_success_payload(
    *,
    sql: str,
    columns: list[str],
    rows: list[tuple[Any, ...]],
) -> str:
    """Structured success payload — sole source of truth for the assistant."""
    serialized_rows = [
        {
            columns[i]: (None if value is None else value)
            for i, value in enumerate(row)
        }
        for row in rows
    ]
    # JSON-safe: stringify non-JSON types
    safe_rows: list[dict[str, Any]] = []
    for row in serialized_rows:
        safe_rows.append(
            {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in row.items()}
        )
    payload = {
        "success": True,
        "sql": sql,
        "columns": columns,
        "rows": safe_rows,
        "row_count": len(rows),
        "table_text": _format_rows(columns, rows),
    }
    return json.dumps(payload, default=str)


def _tool_failure_payload(error: str, *, sql: str = "") -> str:
    payload = {
        "success": False,
        "error": error,
        "sql": sql,
        "rows": [],
        "row_count": 0,
    }
    return json.dumps(payload, default=str)


def _run_query(sql: str) -> str:
    """Validate + execute a read-only query; return structured JSON."""
    try:
        import psycopg
    except ImportError as exc:
        return _tool_failure_payload(
            "psycopg is not installed. Run: pip install 'psycopg[binary]>=3.2.0'"
        )

    try:
        safe_sql = _validate_readonly_sql(sql)
    except Exception as exc:  # noqa: BLE001
        log_db_security_event(
            event="sql_validator_reject",
            generated_sql=sql,
            validator_ok=False,
            validator_error=str(exc),
            tool_success=False,
            tool_error=str(exc),
        )
        return _tool_failure_payload(str(exc), sql=sql)

    log_db_security_event(
        event="sql_validator_pass",
        generated_sql=safe_sql,
        validator_ok=True,
    )

    limited_sql = (
        safe_sql
        if re.search(r"\blimit\b", safe_sql, re.IGNORECASE)
        else f"{safe_sql} LIMIT {MAX_ROWS}"
    )

    try:
        conn_url = _build_database_url()
        with psycopg.connect(conn_url, connect_timeout=15) as conn:
            # Prefer read-only session even on elevated roles.
            with conn.cursor() as cur:
                try:
                    cur.execute("SET default_transaction_read_only = on")
                except Exception:  # noqa: BLE001
                    pass
                cur.execute(limited_sql)
                if cur.description is None:
                    result = _tool_failure_payload(
                        "Query executed but returned no result set "
                        "(writes are not allowed).",
                        sql=limited_sql,
                    )
                    log_db_security_event(
                        event="tool_execution",
                        generated_sql=limited_sql,
                        validator_ok=True,
                        tool_success=False,
                        tool_error="no result set",
                    )
                    return result
                columns = [desc.name for desc in cur.description]
                rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        log_db_security_event(
            event="tool_execution",
            generated_sql=limited_sql,
            validator_ok=True,
            tool_success=False,
            tool_error=str(exc),
        )
        return _tool_failure_payload(str(exc), sql=limited_sql)

    note = ""
    if len(rows) >= MAX_ROWS and not re.search(r"\blimit\b", safe_sql, re.IGNORECASE):
        note = f"\n\n(Showing first {MAX_ROWS} rows. Add LIMIT for more control.)"

    payload = json.loads(_tool_success_payload(sql=limited_sql, columns=columns, rows=rows))
    if note:
        payload["table_text"] = payload["table_text"] + note
    log_db_security_event(
        event="tool_execution",
        generated_sql=limited_sql,
        validator_ok=True,
        tool_success=True,
        extra={"row_count": len(rows)},
    )
    return json.dumps(payload, default=str)


def tables_referenced_in_sql(sql: str) -> list[str]:
    """Extract base table names from FROM / JOIN clauses (best-effort)."""
    found = re.findall(
        r"\b(?:from|join)\s+([a-zA-Z_][\w]*)",
        sql or "",
        flags=re.IGNORECASE,
    )
    # Preserve order, drop duplicates.
    ordered: list[str] = []
    for name in found:
        lowered = name.lower()
        if lowered not in ordered:
            ordered.append(lowered)
    return ordered


def parse_store_query_tool_result(tool_content: str) -> dict[str, Any]:
    """Parse SQL / row count from a query_store_database tool message.

    Supports structured JSON payloads and the legacy text table format.
    """
    text = (tool_content or "").strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
            sql = str(payload.get("sql") or "").strip()
            return {
                "success": bool(payload.get("success", False)),
                "sql": sql,
                "row_count": payload.get("row_count"),
                "tables": tables_referenced_in_sql(sql),
                "error": str(payload.get("error") or ""),
                "table_text": str(payload.get("table_text") or ""),
                "rows": payload.get("rows") or [],
            }
        except json.JSONDecodeError:
            pass

    sql_match = re.search(
        r"SQL:\s*(.*?)(?:\n\nRows:|\nRows:)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    rows_match = re.search(r"Rows:\s*(\d+)", text, flags=re.IGNORECASE)
    sql = (sql_match.group(1).strip() if sql_match else "").strip()
    err = ""
    success = True
    if text.lower().startswith("database query error") or '"success": false' in text.lower():
        success = False
        err = text
    return {
        "success": success,
        "sql": sql,
        "row_count": int(rows_match.group(1)) if rows_match else None,
        "tables": tables_referenced_in_sql(sql),
        "error": err,
        "table_text": text,
        "rows": [],
    }


def format_store_sources(
    *,
    sql: str = "",
    tables: list[str] | None = None,
    row_count: int | None = None,
    tool_content: str = "",
) -> str:
    """Build a mandatory Sources footer for store SQL answers."""
    parsed = parse_store_query_tool_result(tool_content) if tool_content else {}
    used_sql = (sql or parsed.get("sql") or "").strip()
    used_tables = tables if tables is not None else list(parsed.get("tables") or [])
    if not used_tables and used_sql:
        used_tables = tables_referenced_in_sql(used_sql)
    used_rows = row_count if row_count is not None else parsed.get("row_count")

    lines = [
        "Sources:",
        "- Neon database `neondb` via `query_store_database`",
    ]
    if used_tables:
        lines.append(f"- Tables: {', '.join(used_tables)}")
    if used_rows is not None:
        lines.append(f"- Rows returned: {used_rows}")
    if used_sql:
        lines.append(f"- SQL:\n```sql\n{used_sql}\n```")
    else:
        lines.append("- SQL: (not available)")
    return "\n".join(lines)


def ensure_store_sources_footer(answer: str, tool_content: str) -> str:
    """Append Sources if missing; replace a weak existing footer if present."""
    body = (answer or "").strip()
    body = re.sub(
        r"\n*-{0,3}\s*\n?(?:Sources:|_Queried Neon with:_).*\Z",
        "",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    return f"{body}\n\n{format_store_sources(tool_content=tool_content)}"


def test_connection() -> str:
    """Verify Neon connectivity; used by seed scripts and health checks."""
    try:
        import psycopg
    except ImportError as exc:
        return f"Connection failed: {exc}"

    try:
        conn_url = _build_database_url()
        with psycopg.connect(conn_url, connect_timeout=15) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database(), current_user, version()")
                database, user, version = cur.fetchone()
        return (
            f"Connected to Neon.\n"
            f"database={database}\nuser={user}\nversion={version}"
        )
    except Exception as exc:  # noqa: BLE001 — surface raw DB errors to callers
        return f"Connection failed: {exc}"


def extract_sql_from_text(text: str) -> str:
    """Pull a single SELECT/WITH statement out of an LLM reply."""
    cleaned = (text or "").strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, re.IGNORECASE | re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    cleaned = re.sub(r"^(?:sql|query)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    match = re.search(r"((?:with|select)\b[\s\S]+)", cleaned, re.IGNORECASE)
    if not match:
        raise ValueError(f"No SELECT query found in model output: {text[:200]}")
    return _validate_readonly_sql(match.group(1))


def answer_store_question_sync(user_text: str, llm_invoke) -> str:
    """Generate SQL → run Neon query → ground a natural-language answer.

    Refreshes the live schema first, then uses plain LLM invokes (no tool_choice)
    so Groq does not fail with 'Failed to call a function'.
    """
    schema = refresh_store_schema()
    sql_prompt = (
        f"{SQL_GENERATOR_SYSTEM}\n"
        f"{schema}\n\n"
        "Return ONLY one read-only SELECT or WITH ... SELECT statement. "
        "No markdown, no explanation. Prefer LIMIT 10 for ranked/list questions. "
        "For 'low stock' / 'low in stock', order by stock_qty ASC. "
        "If the user asks to change data, reply with REFUSE_WRITE (no SQL)."
    )
    sql_raw = str(
        llm_invoke(
            [
                {"role": "system", "content": sql_prompt},
                {"role": "user", "content": user_text},
            ]
        )
    ).strip()
    if "REFUSE_WRITE" in sql_raw.upper() and "SELECT" not in sql_raw.upper():
        return (
            "This database is permanently read-only. I cannot change stored data.\n\n"
            "Ask a read question instead (for example: which products are low in stock?)."
        )

    try:
        sql = extract_sql_from_text(sql_raw)
        tool_json = _run_query(sql)
    except Exception as first_error:  # noqa: BLE001
        repair_prompt = (
            f"{SQL_GENERATOR_SYSTEM}\n"
            f"The previous SQL failed with: {first_error}\n"
            "Write a corrected single SELECT/WITH query only. Schema:\n"
            f"{schema}"
        )
        sql_raw = str(
            llm_invoke(
                [
                    {"role": "system", "content": repair_prompt},
                    {"role": "user", "content": user_text},
                ]
            )
        )
        sql = extract_sql_from_text(sql_raw)
        tool_json = _run_query(sql)

    parsed = parse_store_query_tool_result(tool_json)
    if not parsed.get("success"):
        err = parsed.get("error") or "Query failed."
        log_db_security_event(
            event="final_response",
            user_prompt=user_text,
            generated_sql=str(parsed.get("sql") or ""),
            tool_success=False,
            tool_error=str(err),
            final_response=str(err)[:500],
        )
        return (
            f"I could not read from the database.\n\n"
            f"Tool error: {err}\n\n"
            f"{format_store_sources(tool_content=tool_json)}"
        )

    result_view = parsed.get("table_text") or tool_json
    answer_prompt = (
        "Answer the user's store question using ONLY the tool result below. "
        "The tool JSON/table is the ONLY source of truth — never invent rows or "
        "claim a write succeeded. "
        "If success is false, explain the error; do not pretend data changed. "
        "List only names and numbers that appear in the result. Be concise. "
        "Do NOT write a Sources section yourself — it is appended automatically.\n\n"
        f"User question: {user_text}\n\n"
        f"Tool result:\n{result_view}"
    )
    answer = llm_invoke(
        [
            {"role": "system", "content": answer_prompt},
            {"role": "user", "content": "Write the final answer now."},
        ]
    )
    final = ensure_store_sources_footer(str(answer), tool_json)
    log_db_security_event(
        event="final_response",
        user_prompt=user_text,
        intent="read",
        generated_sql=str(parsed.get("sql") or ""),
        validator_ok=True,
        tool_success=True,
        final_response=final,
    )
    return final


@tool
async def get_store_schema() -> str:
    """Refresh Neon schema to disk, then return solar_store_schema.sql contents."""
    return await run_in_thread(refresh_store_schema)


def _get_store_schema_sync() -> str:
    """Refresh from Neon when possible, then return schema (+ live counts)."""
    file_schema = refresh_store_schema()
    try:
        import psycopg
        from psycopg import sql
    except ImportError:
        return file_schema

    try:
        conn_url = _build_database_url()
        with psycopg.connect(conn_url, connect_timeout=15) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """
                )
                tables = [row[0] for row in cur.fetchall()]
                counts: list[str] = []
                for table in tables:
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
                    )
                    counts.append(f"- {table}: {cur.fetchone()[0]} rows")
        live = "\n".join(counts) if counts else "- (no tables yet — run seed script)"
        return f"{file_schema}\n\nLive tables:\n{live}"
    except Exception as exc:  # noqa: BLE001
        return f"{file_schema}\n\n(Live counts unavailable: {exc})"


@tool
async def query_store_database(sql: str) -> str:
    """Execute ONE read-only SELECT against the Solar Store Neon database.

    Explicit contract for the query writer:
    - Pass ONLY a single SELECT or WITH ... SELECT statement.
    - Never pass UPDATE/INSERT/DELETE/DDL or multiple statements.
    - Returns structured JSON: success/rows/row_count or success=false + error.
    - This tool output is the sole source of truth — never invent results.

    Args:
        sql: One SELECT (or WITH ... SELECT) query only.

    Returns:
        JSON string with success, rows, row_count (or error).
    """
    try:
        await run_in_thread(refresh_store_schema)
        return await run_in_thread(_run_query, sql)
    except Exception as exc:  # noqa: BLE001
        log_db_security_event(
            event="tool_execution",
            generated_sql=sql,
            tool_success=False,
            tool_error=str(exc),
        )
        return _tool_failure_payload(f"Database query error: {exc}", sql=sql)


@tool
async def check_database_connection() -> str:
    """Test connectivity to the Neon store database configured in .env."""
    return await run_in_thread(test_connection)


__all__ = [
    "STORE_SCHEMA_HINT",
    "SQL_GENERATOR_SYSTEM",
    "answer_store_question_sync",
    "check_database_connection",
    "ensure_store_sources_footer",
    "extract_sql_from_text",
    "format_store_sources",
    "get_store_schema",
    "load_store_schema",
    "needs_store_analytics",
    "needs_store_database",
    "parse_store_query_tool_result",
    "query_store_database",
    "refresh_store_schema",
    "store_schema_path",
    "tables_referenced_in_sql",
    "test_connection",
    "_build_database_url",
    "_run_query",
    "_validate_readonly_sql",
]
