"""AST-based read-only SQL validator (mandatory, independent of the LLM).

Uses sqlglot to parse Postgres SQL and reject anything that is not a pure
SELECT (including WITH … SELECT). Keyword filters alone are not enough —
this gate must run before any SQL reaches the database.
"""

from __future__ import annotations

from typing import Any

# Statement types / constructs that can change database state.
_FORBIDDEN_TYPES: tuple[type, ...] = ()


def _forbidden_expression_types() -> tuple[type, ...]:
    """Lazy import so unit tests can fail clearly if sqlglot is missing."""
    from sqlglot import exp

    return (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Merge,
        exp.Create,
        exp.Drop,
        exp.Alter,
        exp.TruncateTable,
        exp.Command,  # COPY, VACUUM, etc. often land here
        exp.Grant,
        exp.Set,
        exp.Transaction,  # BEGIN / COMMIT / ROLLBACK
        exp.Commit,
        exp.Rollback,
        exp.Kill,
        exp.Refresh,
        exp.Analyze,
        exp.Copy,
        exp.Pragma,
        exp.Describe,
        # Some dialects
        getattr(exp, "Replace", exp.Insert),
    )


def _is_select_root(statement: Any) -> bool:
    """Return True if the parsed root is a SELECT or WITH … SELECT."""
    from sqlglot import exp

    if isinstance(statement, exp.Select):
        return True
    if isinstance(statement, exp.With):
        # WITH must wrap a SELECT (not UPDATE/INSERT/DELETE).
        this = statement.this
        return isinstance(this, exp.Select)
    if isinstance(statement, exp.Subquery) and isinstance(statement.this, exp.Select):
        return True
    return False


def _reject_writable_constructs(statement: Any) -> None:
    """Reject SELECT variants and CTEs that can mutate state."""
    from sqlglot import exp

    # SELECT … INTO / CREATE TABLE AS style
    if list(statement.find_all(exp.Table)):
        # INTO clause on Select (Postgres SELECT INTO)
        for select in statement.find_all(exp.Select):
            into = select.args.get("into")
            if into is not None:
                raise ValueError(
                    "Only SELECT statements are allowed. "
                    "SELECT INTO / writable targets are forbidden."
                )

    # FOR UPDATE / FOR SHARE locks (can escalate; not needed for analytics)
    sql_text = statement.sql(dialect="postgres").upper()
    if " FOR UPDATE" in f" {sql_text}" or " FOR SHARE" in f" {sql_text}":
        raise ValueError(
            "Only SELECT statements are allowed. "
            "FOR UPDATE / FOR SHARE locking is forbidden."
        )

    # Writable CTE: WITH t AS (...) UPDATE/INSERT/DELETE
    if isinstance(statement, exp.With) and not isinstance(statement.this, exp.Select):
        raise ValueError(
            "Only SELECT statements are allowed. Writable CTEs are forbidden."
        )

    # Nested DML anywhere in the tree
    for forbidden in _forbidden_expression_types():
        if forbidden is None:
            continue
        hits = list(statement.find_all(forbidden))
        # Select roots are fine; ignore if the forbidden type somehow matches Select
        if isinstance(statement, forbidden) and not _is_select_root(statement):
            raise ValueError(
                f"Only SELECT statements are allowed. Found {forbidden.__name__}."
            )
        for hit in hits:
            if isinstance(hit, exp.Select):
                continue
            # SET inside UPDATE is different from session SET — already covered by Update
            if isinstance(hit, exp.Set) and _is_select_root(statement):
                # rare false positive; still reject session SET commands
                raise ValueError(
                    "Only SELECT statements are allowed. SET / session commands are forbidden."
                )
            if not isinstance(hit, (exp.Select, exp.With, exp.Subquery)):
                # Allow common expression nodes that aren't DML
                if forbidden in (exp.Set,) or hit.__class__.__name__ in {
                    "Insert",
                    "Update",
                    "Delete",
                    "Merge",
                    "Create",
                    "Drop",
                    "Alter",
                    "TruncateTable",
                    "Grant",
                    "Command",
                    "Transaction",
                    "Commit",
                    "Rollback",
                    "Copy",
                }:
                    raise ValueError(
                        "Only SELECT statements are allowed. "
                        f"Found mutating/admin node: {hit.__class__.__name__}."
                    )


def validate_readonly_sql_ast(sql: str) -> str:
    """Parse and enforce a single read-only SELECT / WITH … SELECT.

    Raises:
        ValueError: if parsing fails or the statement is not a pure read.
    """
    cleaned = (sql or "").strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("SQL query is empty.")

    if ";" in cleaned:
        raise ValueError("Only a single SQL statement is allowed.")

    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as exc:
        raise ImportError(
            "sqlglot is required for AST SQL validation. "
            "Install with: uv add sqlglot"
        ) from exc

    try:
        statements = sqlglot.parse(cleaned, dialect="postgres")
    except Exception as exc:  # noqa: BLE001 — surface parse errors as rejections
        raise ValueError(f"SQL parse failed: {exc}") from exc

    # Drop trailing Nones some versions emit
    statements = [s for s in statements if s is not None]
    if not statements:
        raise ValueError("SQL parse failed: empty statement list.")
    if len(statements) != 1:
        raise ValueError("Only a single SQL statement is allowed.")

    statement = statements[0]
    if not _is_select_root(statement):
        kind = statement.__class__.__name__
        raise ValueError(
            f"Only SELECT statements are allowed. Rejected root type: {kind}."
        )

    # Reject any nested DML / DDL / admin nodes
    forbidden = _forbidden_expression_types()
    for node_type in forbidden:
        for hit in statement.find_all(node_type):
            # Skip if the "hit" is somehow the select wrapper itself
            if hit is statement and _is_select_root(statement):
                continue
            name = hit.__class__.__name__
            if name in {"Select", "With", "Subquery", "Alias", "Table", "Column"}:
                continue
            if isinstance(hit, (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create,
                                exp.Drop, exp.Alter, exp.TruncateTable, exp.Grant,
                                exp.Transaction, exp.Commit, exp.Rollback, exp.Copy,
                                exp.Command)):
                raise ValueError(
                    f"Only SELECT statements are allowed. Found {name}."
                )

    _reject_writable_constructs(statement)

    # Final text-level belt: classic mutating keywords outside identifiers are
    # already handled by AST; keep INTO / CALL / EXECUTE explicit.
    upper = cleaned.upper()
    for token in (
        " INSERT ",
        " UPDATE ",
        " DELETE ",
        " DROP ",
        " ALTER ",
        " CREATE ",
        " TRUNCATE ",
        " MERGE ",
        " CALL ",
        " EXECUTE ",
        " COPY ",
        " GRANT ",
        " REVOKE ",
        " BEGIN ",
        " COMMIT ",
        " ROLLBACK ",
    ):
        # Avoid flagging column names by requiring statement-level spacing; AST is primary.
        padded = f" {upper} "
        if token in padded and not upper.lstrip().startswith("SELECT") and not upper.lstrip().startswith("WITH"):
            raise ValueError(f"Only SELECT statements are allowed. Found{token.strip()}.")

    return cleaned


__all__ = ["validate_readonly_sql_ast"]
