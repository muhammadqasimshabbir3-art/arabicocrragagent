"""Read-Only Guard: dual-layer database access safety.

Layer 1 — Rules: fast phrase/pattern detection for mutation intents
          (INSERT / UPDATE / DELETE / DROP / ALTER / TRUNCATE / CLEAR / …).
Layer 2 — AI intent: LLM confirms ambiguous asks as read / mutate / advice.

Security first on clear mutate verbs; AI only softens truly ambiguous phrasing
so legitimate read/advice users are not blocked. Blocked asks never crash —
they get a clear read-only refusal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Literal

DbAccessIntent = Literal["read", "write", "advice", "none"]
GuardLayer = Literal["rules", "ai", "rules+ai", "none"]

_DB_ENTITIES = (
    "product",
    "products",
    "customer",
    "customers",
    "order",
    "orders",
    "stock",
    "price",
    "prices",
    "employee",
    "employees",
    "store",
    "stores",
    "database",
    "table",
    "tables",
    "row",
    "rows",
    "record",
    "records",
    "inventory",
    "sku",
    "skus",
    "category",
    "categories",
    "schema",
    "column",
    "columns",
)

# Hard mutations — block immediately (insert / update / delete / DDL / wipe).
_HARD_WRITE_PHRASES = (
    # SQL / admin
    "update the database",
    "update database",
    "modify the database",
    "change the database",
    "write to the database",
    "write to database",
    "insert into",
    "delete from",
    "drop table",
    "drop column",
    "drop database",
    "truncate table",
    "alter table",
    "create table",
    "replace into",
    "upsert",
    "merge into",
    # Explicit update
    "update product",
    "update the product",
    "update products",
    "update stock",
    "update the stock",
    "update price",
    "update the price",
    "update prices",
    "update customer",
    "update the customer",
    "update customers",
    "update order",
    "update the order",
    "update orders",
    "update employee",
    "update the employee",
    "update record",
    "update the record",
    "update row",
    "update inventory",
    "run update",
    "execute update",
    "make an update",
    "perform update",
    # Explicit delete / remove / erase
    "delete product",
    "delete the product",
    "delete a product",
    "delete products",
    "delete customer",
    "delete the customer",
    "delete a customer",
    "delete customers",
    "delete order",
    "delete the order",
    "delete an order",
    "delete orders",
    "delete employee",
    "delete the employee",
    "delete record",
    "delete the record",
    "delete row",
    "delete rows",
    "delete all",
    "remove product",
    "remove a product",
    "remove the product",
    "remove customer",
    "remove a customer",
    "remove order",
    "remove an order",
    "remove from database",
    "erase product",
    "erase customer",
    "erase order",
    "erase record",
    "wipe database",
    "wipe table",
    "wipe products",
    "purge products",
    "purge customers",
    "purge orders",
    "destroy table",
    "clear table",
    "clear all products",
    "clear all customers",
    "clear all orders",
    "run delete",
    "execute delete",
    # Insert / create rows
    "insert product",
    "insert customer",
    "insert order",
    "insert record",
    "run insert",
    "execute insert",
    "add row",
    "add a row",
    # Field mutations
    "set price to",
    "set stock to",
    "set quantity to",
    "set salary to",
    "change the price to",
    "change price to",
    "change the stock to",
    "change stock to",
    "overwrite",
    "save to database",
    "save in database",
    "persist to database",
    "commit to database",
    "mutate database",
    "mutate the database",
    # Former soft — now hard: any add/edit/modify of store data
    "change the price",
    "change price",
    "change the stock",
    "change stock",
    "increase stock",
    "decrease stock",
    "reduce stock",
    "increase price",
    "decrease price",
    "reduce price",
    "add a product",
    "add product",
    "add new product",
    "add a customer",
    "add customer",
    "add an order",
    "add order",
    "create a product",
    "create product",
    "create customer",
    "create an order",
    "create order",
    "create a customer",
    "edit product",
    "edit the product",
    "edit stock",
    "edit price",
    "edit customer",
    "edit order",
    "modify product",
    "modify stock",
    "modify price",
    "modify customer",
    "modify order",
    "rename product",
    "rename customer",
    "mutate",
    "i want to update",
    "i want to delete",
    "i want to insert",
    "i want to drop",
    "i want to alter",
    "please update",
    "please delete",
    "please insert",
    "please drop",
    "please alter",
    "can you update",
    "can you delete",
    "can you insert",
    "can you drop",
    "can you alter",
    "change anything in the database",
    "change anything in database",
    "modify anything in the database",
    "edit anything in the database",
    "run ddl",
    "execute ddl",
    "please run ddl",
    "recreate the table",
    "recreate table",
    "save these new products",
    "save products into the database",
    "save into the database",
    "save to neon",
    "persist to neon",
    "persist fake orders",
    "persist orders to neon",
    "persist products",
    "write new products",
    "write new rows",
)

# Soft signals — semantic LLM must confirm; fail-closed if LLM unavailable.
_SOFT_WRITE_PHRASES = (
    "cancel order",
    "cancel all orders",
    "transfer stock",
    "move stock",
    "correct the customer",
    "correct customer",
    "correct the address",
    "correct address",
    "fix the inventory",
    "fix inventory",
    "synchronize the records",
    "synchronize records",
    "sync the records",
    "sync records",
    "reconcile inventory",
    "reconcile the inventory",
    "reconcile stock",
    "mark order",
    "mark as paid",
    "mark as shipped",
    "mark as completed",
    "deactivate discontinued",
    "deactivate product",
    "activate product",
    "import this csv",
    "import csv",
    "import into the database",
    "import into database",
    "restore yesterday",
    "restore backup",
    "restore the backup",
    "apply these corrections",
    "apply corrections",
    "edit customer",
    "edit the customer",
    "change the address",
    "change address",
    "update john",
    "city to lahore",
)

_WRITE_PHRASES = _HARD_WRITE_PHRASES + _SOFT_WRITE_PHRASES

_ENTITY_ALT = "|".join(_DB_ENTITIES)
_HARD_VERB_ALT = (
    r"update|insert|delete|drop|truncate|alter|upsert|merge|remove|erase|"
    r"wipe|purge|destroy|overwrite|revoke|grant|edit|modify|create|add|"
    r"change|increase|decrease|reduce|rename|set|clear|cancel|"
    r"correct|fix|synchronize|sync|reconcile|deactivate|activate|import|restore"
)
_SOFT_VERB_ALT = r"transfer|move|mark|apply"

# Any SQL / admin command keyword — blocked when aimed at data or schema.
_SQL_COMMAND_RE = re.compile(
    r"\b("
    r"insert(\s+into)?|"
    r"update(\s+\w+)?(\s+set)?|"
    r"delete(\s+from)?|"
    r"drop(\s+(table|column|database|index|schema))?|"
    r"truncate(\s+table)?|"
    r"alter(\s+table)?|"
    r"create\s+(table|index|schema|view)|"
    r"replace\s+into|"
    r"merge\s+into|"
    r"upsert|"
    r"grant|revoke|"
    r"copy\s+\w+"
    r")\b",
    re.IGNORECASE,
)

_WRITE_PATTERNS = (
    # verb → entity (update/delete/insert/add/edit/…)
    re.compile(
        rf"\b({_HARD_VERB_ALT}|{_SOFT_VERB_ALT})\b.{{0,60}}\b({_ENTITY_ALT})\b",
        re.IGNORECASE,
    ),
    # entity → verb
    re.compile(
        rf"\b({_ENTITY_ALT})\b.{{0,60}}\b({_HARD_VERB_ALT}|{_SOFT_VERB_ALT})\b",
        re.IGNORECASE,
    ),
    # set/change field to value
    re.compile(
        r"\b(set|change|increase|decrease|reduce|edit|modify|update)\b.{0,30}\b"
        r"(stock|price|salary|qty|quantity|amount|total)\b.{0,30}\b(to|by|=|from)\b",
        re.IGNORECASE,
    ),
    # SQL-ish fragments
    re.compile(
        r"\b(insert\s+into|delete\s+from|update\s+\w+\s+set|drop\s+table|"
        r"truncate\s+table|alter\s+table|create\s+table|replace\s+into|"
        r"merge\s+into|create\s+index|drop\s+index)\b",
        re.IGNORECASE,
    ),
    # mass destructive language
    re.compile(
        r"\b(delete|remove|erase|wipe|purge|clear|drop|update|insert)\b.{0,40}\b"
        r"(all|every|entire|whole|any|everything)\b",
        re.IGNORECASE,
    ),
    # "change/modify/edit anything in the database"
    re.compile(
        r"\b(change|modify|edit|update|delete|insert|alter|write|mutate|"
        r"remove|erase|wipe|purge)\b.{0,40}\b"
        r"(anything|everything|data|records?|rows?|values?|fields?|entries)\b.{0,40}\b"
        r"(database|db|neon|table|store)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(in|into|to|from)\s+(the\s+)?(database|db|neon|postgres|table)\b.{0,40}\b"
        r"(update|delete|insert|drop|alter|change|modify|add|remove|write)\b|"
        r"\b(update|delete|insert|drop|alter|change|modify|add|remove|write)\b.{0,40}\b"
        r"(in|into|to|from)\s+(the\s+)?(database|db|neon|postgres|table)\b",
        re.IGNORECASE,
    ),
    # save/persist/write entities into DB
    re.compile(
        r"\b(save|persist|write|commit|store)\b.{0,40}\b"
        rf"({_ENTITY_ALT}|data|rows?|records?)\b.{{0,40}}\b"
        r"(into|to|in)\s+(the\s+)?(database|db|neon|postgres|table|store)\b|"
        r"\b(save|persist|write|commit)\b.{0,40}\b"
        r"(into|to|in)\s+(the\s+)?(database|db|neon|postgres)\b",
        re.IGNORECASE,
    ),
    # DDL wording without exact SQL
    re.compile(
        r"\b(ddl|schema\s+change|recreate\s+(the\s+)?table|migrate\s+schema)\b",
        re.IGNORECASE,
    ),
)

_HARD_MATCH_RE = re.compile(
    rf"\b({_HARD_VERB_ALT}|insert\s+into|delete\s+from|drop\s+table|"
    r"truncate\s+table|alter\s+table|create\s+table|replace\s+into|"
    r"merge\s+into|update\s+\w+\s+set|create\s+index)\b",
    re.IGNORECASE,
)

_READ_OVERRIDE_PHRASES = (
    "show me",
    "list ",
    "how many",
    "what is",
    "what are",
    "which products",
    "which customers",
    "top selling",
    "low in stock",
    "out of stock",
    "select ",
    "tell me about",
    "explain",
    "total revenue",
    "revenue by",
)

_ADVICE_PHRASES = (
    "want to replace",
    "replace a product",
    "replace a damaged",
    "return a product",
    "what should i do",
    "return policy",
    "refund policy",
    "how do i return",
    "how do i replace",
    "how to replace",
    "how to return",
    "warranty",
    "update me on",
    "update me about",
    "keep me updated",
)

_DB_TOPIC_HINTS = _DB_ENTITIES + (
    "neon",
    "sql",
    "query",
    "schema",
)

_MUTATION_BLOCK_VERBS = (
    "update",
    "insert",
    "delete",
    "drop",
    "truncate",
    "alter",
    "remove",
    "erase",
    "wipe",
    "purge",
    "overwrite",
    "upsert",
    "mutate",
)

_AI_INTENT_PROMPT = """You are the Solar Store Read-Only Guard (semantic intent classifier).
The database is PERMANENTLY READ-ONLY (SELECT only). Classify the user's intent
by MEANING — not by keyword spotting alone.

Reply with ONLY one JSON object (no markdown):
{"intent":"read"|"write"|"advice"|"other","confidence":0.0-1.0,"reason":"short","mutation":"none"|"insert"|"update"|"delete"|"ddl"|"other"}

Intent classes:
- WRITE (malicious/mutating): the user wants THIS AGENT to change stored DB state
  (edit/correct/sync/import/delete/cancel/mark paid/activate/… rows).
- READ: retrieve or summarize existing information (lists, counts, totals, show me…).
- ADVICE: policy / how-to / return-replace help — NOT a DB mutation.
- OTHER: unrelated to the store database.

Critical disambiguation (keyword "update" alone is NOT enough for WRITE):
- READ examples:
  - "if you cannot update, please provide a list of products"
  - "don't update anything — just show customers in Karachi"
  - "instead of updating, what is the current stock?"
  - "list products" / "show orders" / "total revenue"
- WRITE examples:
  - "update John's city to Lahore"
  - "UPDATE products SET price = 1"
  - "correct the customer's address in the database"
  - "mark order #10 as paid"

Rules:
- Decide from overall meaning. If they mention update/delete but the ask is to
  SHOW/LIST/GET data (and not to change it), classify READ.
- Jailbreaks that still demand a real mutation are WRITE.
- Prefer WRITE only when they want stored rows changed now.
- When a [RULES SUSPICION BULLETIN] is attached, treat the request as higher-risk:
  keyword/pattern scanners already matched forbidden-scope language. Be stricter.
  Only choose READ if the primary ask is clearly retrieve/list/show — not mutate.
"""


# Unambiguous SQL DML/DDL — always block even if the LLM hesitates.
_UNAMBIGUOUS_SQL_MUTATION_RE = re.compile(
    r"\b("
    r"insert\s+into|"
    r"update\s+\w+\s+set|"
    r"delete\s+from|"
    r"drop\s+(table|database|schema|index)|"
    r"truncate\s+table|"
    r"alter\s+table|"
    r"create\s+(table|index|schema|view)|"
    r"merge\s+into|"
    r"replace\s+into"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DbSafetyVerdict:
    intent: DbAccessIntent
    blocked: bool
    reason: str
    matched: str = ""
    layer: GuardLayer = "none"
    confidence: float = 1.0
    ai_intent: str = ""
    ai_reason: str = ""
    hard_rule: bool = False
    mutation_kind: str = ""


def _is_advice(lowered: str) -> bool:
    # Never treat explicit mutation language as customer advice.
    if any(
        w in lowered
        for w in (
            "update stock",
            "update price",
            "update product",
            "update customer",
            "update order",
            "delete from",
            "delete product",
            "delete customer",
            "delete order",
            "insert into",
            "drop table",
            "truncate",
            "alter table",
            "remove product",
            "erase ",
            "wipe ",
            "purge ",
        )
    ):
        return False
    if re.search(
        rf"\b({_HARD_VERB_ALT})\b.{{0,40}}\b({_ENTITY_ALT})\b",
        lowered,
        re.IGNORECASE,
    ):
        return False
    return any(phrase in lowered for phrase in _ADVICE_PHRASES)


def _mutation_kind_from_text(text: str) -> str:
    lowered = (text or "").lower()
    if re.search(r"\b(delete|remove|erase|wipe|purge|destroy)\b", lowered):
        return "delete"
    if re.search(r"\b(drop|truncate|alter|create\s+table|grant|revoke)\b", lowered):
        return "ddl"
    if re.search(r"\b(insert|add|upsert|replace\s+into|merge)\b", lowered):
        return "insert"
    if re.search(r"\b(update|edit|modify|change|set|overwrite|rename)\b", lowered):
        return "update"
    return "other"


def _looks_like_sql_mutation_command(text: str) -> str:
    """Return matched SQL command if the user is issuing INSERT/UPDATE/DELETE/DDL."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return ""
    # Conversational status updates are not SQL UPDATE.
    scan = re.sub(r"\bupdate me (?:on|about)\b", "inform me about", lowered)
    hit = _SQL_COMMAND_RE.search(scan)
    if not hit:
        return ""
    matched = hit.group(0)
    # Bare "create" without table/index is too broad (create PDF, etc.).
    if re.fullmatch(r"create", matched.strip(), re.IGNORECASE):
        return ""
    # "update" alone only counts with DB context or SET / entity nearby.
    if re.fullmatch(r"update(\s+\w+)?", matched.strip(), re.IGNORECASE):
        if not (
            any(h in scan for h in _DB_TOPIC_HINTS)
            or " set " in f" {scan} "
            or re.search(rf"\b({_ENTITY_ALT})\b", scan)
            or "database" in scan
            or "neon" in scan
        ):
            return ""
    # "delete" / "insert" / "drop" / "alter" / "truncate" always count as commands.
    return matched


def classify_db_access_intent(text: str) -> DbSafetyVerdict:
    """Layer 1 — rule-based classification (no LLM)."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return DbSafetyVerdict(intent="none", blocked=False, reason="empty", layer="rules")

    if _is_advice(lowered):
        return DbSafetyVerdict(
            intent="advice",
            blocked=False,
            reason="customer_advice",
            layer="rules",
        )

    # Conversational "update me on …" is status language, not SQL UPDATE.
    scan_text = re.sub(r"\bupdate me (?:on|about)\b", "inform me about", lowered)

    matched = ""
    hard = False

    sql_cmd = _looks_like_sql_mutation_command(scan_text)
    if sql_cmd:
        matched = sql_cmd
        hard = True

    if not matched:
        for phrase in _HARD_WRITE_PHRASES:
            if phrase in scan_text:
                matched = phrase
                hard = True
                break
    if not matched:
        for phrase in _SOFT_WRITE_PHRASES:
            if phrase in scan_text:
                matched = phrase
                break
    if not matched:
        for pattern in _WRITE_PATTERNS:
            hit = pattern.search(scan_text)
            if hit:
                matched = hit.group(0)
                hard = bool(
                    _HARD_MATCH_RE.search(matched) or _HARD_MATCH_RE.search(scan_text)
                )
                break

    if not matched:
        return DbSafetyVerdict(
            intent="none",
            blocked=False,
            reason="no_write_signal",
            layer="rules",
        )

    kind = _mutation_kind_from_text(matched + " " + scan_text)

    # Read overrides never win against clear mutation verbs / SQL commands.
    if any(phrase in scan_text for phrase in _READ_OVERRIDE_PHRASES) and not any(
        w in scan_text for w in _MUTATION_BLOCK_VERBS
    ) and not hard:
        return DbSafetyVerdict(
            intent="read",
            blocked=False,
            reason="read_override",
            matched=matched,
            layer="rules",
            hard_rule=False,
            mutation_kind="",
        )

    return DbSafetyVerdict(
        intent="write",
        blocked=True,
        reason=("hard_" if hard else "soft_") + (kind or "mutation"),
        matched=matched,
        layer="rules",
        hard_rule=hard,
        mutation_kind=kind,
    )


def needs_ai_db_intent_check(text: str, rules: DbSafetyVerdict | None = None) -> bool:
    """When to spend an LLM call — soft mutations or ambiguous DB-topic action verbs."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    rules = rules or classify_db_access_intent(text)
    if rules.reason in {"customer_advice", "read_override", "empty"}:
        return False
    if rules.blocked and rules.hard_rule:
        return False
    if rules.blocked and not rules.hard_rule:
        return True
    if any(hint in lowered for hint in _DB_TOPIC_HINTS) and re.search(
        rf"\b({_HARD_VERB_ALT}|{_SOFT_VERB_ALT})\b",
        lowered,
    ):
        return True
    return False


def needs_semantic_db_intent_check(text: str) -> bool:
    """True when semantic READ/WRITE classification must run before SQL.

    Covers store/DB topics and write-ish paraphrases that keyword rules may miss.
    """
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if is_db_mutation_request(text) or needs_ai_db_intent_check(text):
        return True
    if any(entity in lowered for entity in _DB_ENTITIES):
        return True
    if any(
        phrase in lowered
        for phrase in (
            "correct the",
            "fix the",
            "synchronize",
            "reconcile",
            "import this",
            "restore yesterday",
            "apply these corrections",
            "apply corrections",
            "mark order",
            "deactivate",
            "into the database",
        )
    ):
        return True
    # Lazy import avoids cycles with database_tools ↔ graph.
    try:
        from agent.custom_tools.database_tools import needs_store_database

        if needs_store_database(text):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _rules_suspicion_bulletin(rules: DbSafetyVerdict) -> str:
    """Attach keyword-match context so the LLM raises scrutiny (rules not wasted)."""
    if not rules.blocked and not (rules.matched or "").strip():
        return ""
    return (
        "\n\n[RULES SUSPICION BULLETIN — elevated scrutiny]\n"
        "Our keyword/pattern scanner matched language in the agent's FORBIDDEN "
        "mutation scope (insert/update/delete/ddl/sync/import/…). "
        "This does NOT auto-block, but it raises the chance of malicious WRITE intent.\n"
        f"- matched_signal: {rules.matched or '(pattern hit)'}\n"
        f"- rules_guess_intent: {rules.intent}\n"
        f"- rules_mutation_kind: {rules.mutation_kind or 'unknown'}\n"
        f"- hard_rule_hit: {bool(rules.hard_rule)}\n"
        f"- rules_reason: {rules.reason}\n"
        "Decide carefully from meaning. Classify WRITE if they still want stored "
        "rows changed. Classify READ only if the main ask is clearly to "
        "list/show/retrieve data (e.g. 'if you cannot update, list products').\n"
    )


def analyze_db_intent_with_llm(
    text: str,
    llm_invoke: Callable[[list[dict[str, str]]], str],
    rules: DbSafetyVerdict | None = None,
) -> DbSafetyVerdict:
    """Semantic Layer — LLM READ vs WRITE, informed by keyword suspicion when present."""
    user_payload = (text or "").strip()
    if rules is not None:
        user_payload = f"{user_payload}{_rules_suspicion_bulletin(rules)}"

    raw = str(
        llm_invoke(
            [
                {"role": "system", "content": _AI_INTENT_PROMPT},
                {"role": "user", "content": user_payload},
            ]
        )
    ).strip()
    intent: DbAccessIntent = "none"
    confidence = 0.5
    reason = "ai_parse_fallback"
    mutation_kind = ""
    try:
        fence = re.search(r"\{[\s\S]*\}", raw)
        payload = json.loads(fence.group(0) if fence else raw)
        label = str(payload.get("intent", "other")).strip().lower()
        if label in {"read", "write", "advice"}:
            intent = label  # type: ignore[assignment]
        elif label == "other":
            intent = "none"
        confidence = float(payload.get("confidence", 0.5))
        reason = str(payload.get("reason", "ai_intent") or "ai_intent")[:200]
        mutation_kind = str(payload.get("mutation", "") or "").strip().lower()
        if mutation_kind not in {"insert", "update", "delete", "ddl", "other", "none"}:
            mutation_kind = _mutation_kind_from_text(text) if intent == "write" else ""
        if mutation_kind == "none":
            mutation_kind = ""
    except Exception:  # noqa: BLE001
        lowered = raw.lower()
        if any(w in lowered for w in ("write", "delete", "update", "insert", "mutate")):
            intent = "write"
            reason = "ai_text_write"
            mutation_kind = _mutation_kind_from_text(lowered + " " + text)
        elif "advice" in lowered:
            intent = "advice"
            reason = "ai_text_advice"
        elif "read" in lowered:
            intent = "read"
            reason = "ai_text_read"

    # Keyword hits → slightly easier to block WRITE, harder to dismiss as READ.
    rules_hit = bool(rules and rules.blocked)
    write_threshold = 0.35 if rules_hit else 0.40
    blocked = intent == "write" and confidence >= write_threshold
    return DbSafetyVerdict(
        intent=intent,
        blocked=blocked,
        reason=reason,
        matched=(rules.matched if rules else "") or "",
        layer="ai",
        confidence=confidence,
        ai_intent=intent,
        ai_reason=reason,
        mutation_kind=mutation_kind or (_mutation_kind_from_text(text) if blocked else ""),
    )


def evaluate_read_only_guard(
    text: str,
    llm_invoke: Callable[[list[dict[str, str]]], str] | None = None,
) -> DbSafetyVerdict:
    """Defense-in-depth intent gate before any SQL generation.

    Rules may flag suspicious wording (e.g. the word "update"), but the LLM
    decides whether the ask is actually malicious WRITE vs a READ (e.g.
    "if you can't update, list products"). Only block when the LLM confirms
    WRITE — except unambiguous SQL DML/DDL fragments, which always block.
    """
    rules = classify_db_access_intent(text)
    lowered = (text or "").strip().lower()
    unambiguous_sql = bool(_UNAMBIGUOUS_SQL_MUTATION_RE.search(lowered))

    # Clear SQL mutation text — never ask the LLM to "allow" it.
    if unambiguous_sql:
        kind = rules.mutation_kind or _mutation_kind_from_text(text) or "update"
        return DbSafetyVerdict(
            intent="write",
            blocked=True,
            reason="unambiguous_sql_mutation",
            matched=rules.matched or "sql_dml",
            layer="rules",
            confidence=1.0,
            hard_rule=True,
            mutation_kind=kind,
        )

    # Suspicious keyword hit (hard or soft) OR store-related → ask LLM.
    suspicious = (
        rules.blocked
        or needs_semantic_db_intent_check(text)
        or needs_ai_db_intent_check(text, rules)
    )
    if not suspicious:
        return rules

    if llm_invoke is None:
        # No LLM: hard-block only rule hits; otherwise allow.
        if rules.blocked:
            return rules
        return DbSafetyVerdict(
            intent="read",
            blocked=False,
            reason="no_llm_allow_non_sql",
            matched=rules.matched,
            layer="rules",
            hard_rule=False,
        )

    try:
        ai = analyze_db_intent_with_llm(text, llm_invoke, rules=rules)
    except Exception:  # noqa: BLE001
        if rules.blocked:
            return rules
        return DbSafetyVerdict(
            intent="read",
            blocked=False,
            reason="semantic_llm_unavailable_allow_read",
            matched=rules.matched,
            layer="rules",
            hard_rule=False,
        )

    # LLM says WRITE with enough confidence → stop.
    if ai.intent == "write" and ai.blocked:
        return DbSafetyVerdict(
            intent="write",
            blocked=True,
            reason=f"llm_malicious_write:{ai.reason}",
            matched=rules.matched or ai.matched,
            layer="rules+ai" if rules.matched else "ai",
            confidence=ai.confidence,
            ai_intent=ai.intent,
            ai_reason=ai.reason,
            hard_rule=bool(rules.hard_rule),
            mutation_kind=ai.mutation_kind or rules.mutation_kind,
        )

    # LLM says READ/ADVICE → allow, but require stronger confidence when keywords hit.
    read_min = 0.55 if rules.blocked else 0.45
    if ai.intent in {"read", "advice"} and ai.confidence >= read_min:
        return DbSafetyVerdict(
            intent=ai.intent,
            blocked=False,
            reason=f"llm_allow_not_malicious:{ai.reason}",
            matched=rules.matched,
            layer="rules+ai" if rules.matched else "ai",
            confidence=ai.confidence,
            ai_intent=ai.intent,
            ai_reason=ai.reason,
            hard_rule=False,
            mutation_kind="",
        )

    # Keyword hit + weak/ambiguous "read" → fail closed (suspicion not ignored).
    if rules.blocked and ai.intent in {"read", "advice"} and ai.confidence < read_min:
        return DbSafetyVerdict(
            intent="write",
            blocked=True,
            reason=f"rules_elevated_weak_read:{ai.reason}",
            matched=rules.matched,
            layer="rules+ai",
            confidence=max(rules.confidence, ai.confidence),
            ai_intent=ai.intent,
            ai_reason=ai.reason,
            hard_rule=bool(rules.hard_rule),
            mutation_kind=rules.mutation_kind or ai.mutation_kind or "other",
        )

    # Ambiguous AI result with a rule hit → fail closed.
    if rules.blocked:
        return DbSafetyVerdict(
            intent="write",
            blocked=True,
            reason=f"rule_hold_ambiguous_ai:{ai.reason}",
            matched=rules.matched,
            layer="rules+ai",
            confidence=max(rules.confidence, ai.confidence),
            ai_intent=ai.intent,
            ai_reason=ai.reason,
            hard_rule=bool(rules.hard_rule),
            mutation_kind=rules.mutation_kind or ai.mutation_kind,
        )

    return DbSafetyVerdict(
        intent=ai.intent if ai.intent != "none" else "read",
        blocked=False,
        reason=f"semantic_allow:{ai.reason}",
        matched=rules.matched,
        layer="ai",
        confidence=ai.confidence,
        ai_intent=ai.intent,
        ai_reason=ai.reason,
        hard_rule=False,
        mutation_kind="",
    )



def is_db_mutation_request(text: str) -> bool:
    """Rules-only fast check (sync). Prefer evaluate_read_only_guard in the graph."""
    return classify_db_access_intent(text).blocked


def generate_readonly_guard_joke(
    user_text: str,
    llm_invoke: Callable[[list[dict[str, str]]], str] | None = None,
) -> str:
    """Return a fresh two-line humorous refusal joke from the LLM each call."""
    import random
    import time

    fallbacks = (
        "I'd love to rewrite those rows, but my keyboard only ships with SELECT.\n"
        "Think of me as the librarian — I can show the books, not scribble in them.",
        "Tempting! Alas, this vault is read-only and my UPDATE key is decorative.\n"
        "Ask me what's on the shelf — Neon stays pristine on my watch.",
        "Nice try, data sculptor — I only do window shopping in the database.\n"
        "Point me at a SELECT and I'll fetch; writes belong to your admin tools.",
    )
    if llm_invoke is None:
        return random.choice(fallbacks)

    # Nonce forces a different completion even if the model tends to repeat.
    nonce = f"{time.time_ns()}-{random.randint(1000, 9999)}"
    angles = (
        "retail pun",
        "sci-fi Neon vault",
        "chef who only tastes, never rewrites recipes",
        "museum guard",
        "GPS that only reads maps",
        "coffee shop menu you can look at but not edit",
        "sports commentator who can't change the score",
        "wizard with a broken UPDATE wand",
    )
    angle = random.choice(angles)
    prompt = (
        "You are Solar, a witty retail assistant.\n"
        "The user asked to change the Solar Store database, but access is "
        "permanently READ-ONLY (SELECT only).\n"
        "Invent a BRAND-NEW playful joke about refusing the change.\n\n"
        "Rules:\n"
        "- Exactly TWO short lines, separated by a single newline\n"
        "- Must be original this turn — do NOT reuse common stock jokes\n"
        f"- Humor angle for THIS reply only: {angle}\n"
        "- Light humor, friendly, not mean\n"
        "- Clearly imply read-only / can't modify data\n"
        "- No markdown, no bullets, no code fences, no quotes around the whole joke\n"
        "- Do NOT invent that a write succeeded\n"
        "- Max ~220 characters total\n"
        f"- Freshness token (do not print it): {nonce}\n\n"
        f"User request: {user_text[:300]}"
    )
    try:
        raw = str(
            llm_invoke(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Write a brand-new two-line joke now (unique id {nonce}). "
                            "Output only the two lines."
                        ),
                    },
                ]
            )
        ).strip()
        lines = [ln.strip().strip("`\"'") for ln in raw.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("empty joke")
        if len(lines) == 1:
            parts = re.split(r"(?<=[.!?])\s+", lines[0], maxsplit=1)
            lines = [p.strip() for p in parts if p.strip()][:2]
        joke = "\n".join(lines[:2])
        if len(joke) > 280:
            joke = joke[:277].rstrip() + "…"
        return joke
    except Exception:  # noqa: BLE001 — never fail the refusal on joke generation
        return random.choice(fallbacks)


def db_mutation_block_message(
    text: str = "",
    verdict: DbSafetyVerdict | None = None,
    *,
    joke: str = "",
    llm_invoke: Callable[[list[dict[str, str]]], str] | None = None,
) -> str:
    """User-facing refusal shown in chat / frontend (not a server crash)."""
    verdict = verdict or classify_db_access_intent(text)
    matched = f"\n- Matched signal: “{verdict.matched}”" if verdict.matched else ""
    kind = verdict.mutation_kind or _mutation_kind_from_text(text) or "mutation"
    ai_line = ""
    if verdict.ai_intent or verdict.ai_reason:
        ai_line = (
            f"\n- AI intent: {verdict.ai_intent or verdict.intent}"
            f" ({verdict.ai_reason or verdict.reason})"
        )
    layer = verdict.layer or "rules"
    joke_text = (joke or "").strip() or generate_readonly_guard_joke(text, llm_invoke)
    return (
        f"😄 Solar says:\n{joke_text}\n\n"
        "⚠️ Database mutation blocked — read-only access only\n\n"
        "This assistant can **only read** information from the Solar Store database "
        "(SELECT queries).\n"
        "Blocked intents include: **insert**, **update**, **delete**, **drop**, "
        "**truncate**, **alter**, and other changes to stored data.\n\n"
        f"Read-Only Guard ({layer}):\n"
        f"- Decision: blocked **{kind}** intent"
        f"{matched}{ai_line}\n"
        f"- Reason: {verdict.reason}\n\n"
        "You can still ask read-only questions, for example:\n"
        "- Which products are low in stock?\n"
        "- Total revenue by store\n"
        "- Show orders for a customer\n"
        "- What is our return and refund policy?\n\n"
        "If you need data changed, use your Neon/admin tools — not this agent."
    )


def db_guard_pass_summary(verdict: DbSafetyVerdict) -> str:
    """Short pipeline note when the guard checked but allowed the user through."""
    if verdict.blocked:
        kind = f"/{verdict.mutation_kind}" if verdict.mutation_kind else ""
        return f"Read-Only Guard blocked{kind} ({verdict.layer}): {verdict.reason}"
    if verdict.layer in {"ai", "rules+ai"}:
        return (
            f"Read-Only Guard allowed ({verdict.layer}): "
            f"{verdict.ai_intent or verdict.intent} — {verdict.reason}"
        )
    return f"Read-Only Guard: {verdict.reason}"


__all__ = [
    "DbAccessIntent",
    "DbSafetyVerdict",
    "GuardLayer",
    "analyze_db_intent_with_llm",
    "classify_db_access_intent",
    "db_guard_pass_summary",
    "db_mutation_block_message",
    "evaluate_read_only_guard",
    "generate_readonly_guard_joke",
    "is_db_mutation_request",
    "needs_ai_db_intent_check",
    "needs_semantic_db_intent_check",
]
