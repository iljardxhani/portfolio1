from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from statistics import fmean
from typing import Any
from typing import Iterable
from typing import Protocol
from typing import Sequence
import json
import re
import time


class Route(str, Enum):
    SQL = "sql"
    DOCS = "docs"
    HYBRID = "hybrid"
    CLARIFY = "clarify"


class GuardReason(str, Enum):
    NONE = "none"
    ROUTE_CONFIDENCE_LOW = "route_confidence_low"
    SQL_PLAN_UNSAFE = "sql_plan_unsafe"
    SQL_EXECUTION_FAILED = "sql_execution_failed"
    DOCS_EVIDENCE_WEAK = "docs_evidence_weak"
    CONFLICT_UNRESOLVED = "conflict_unresolved"
    OUTPUT_UNSUPPORTED = "output_unsupported"


class ConflictType(str, Enum):
    NONE = "none"
    NUMERIC_MISMATCH = "numeric_mismatch"
    POLICY_MISMATCH = "policy_mismatch"
    MISSING_DOC_SUPPORT = "missing_doc_support"


@dataclass(slots=True)
class QueryEnvelope:
    request_id: str
    session_id: str
    actor_id: str
    query: str
    locale: str
    created_at_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouteDecision:
    route: Route
    confidence: float
    reason: str
    requires_sql: bool
    requires_docs: bool
    requires_clarification: bool
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SQLPlan:
    sql: str
    params: dict[str, Any]
    tables: list[str]
    columns: list[str]
    estimated_cost: float
    read_only: bool
    policy_tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SQLExecution:
    query_id: str
    rows: list[dict[str, Any]]
    row_count: int
    latency_ms: int
    truncated: bool
    summary: str


@dataclass(slots=True)
class RetrievalChunk:
    chunk_id: str
    source_id: str
    title: str
    section: str
    score: float
    text: str


@dataclass(slots=True)
class DocsRetrieval:
    query: str
    chunks: list[RetrievalChunk]
    citations: list[str]
    avg_score: float
    max_score: float
    latency_ms: int


@dataclass(slots=True)
class EvidenceItem:
    evidence_id: str
    origin: str
    label: str
    payload: dict[str, Any]
    confidence: float


@dataclass(slots=True)
class SynthesisResult:
    answer: str
    confidence: float
    conflict_type: ConflictType
    conflict_notes: list[str]
    sections: list[str]


@dataclass(slots=True)
class GuardrailReport:
    accepted: bool
    reasons: list[GuardReason]
    confidence: float
    route_confidence: float
    evidence_coverage: float
    conflict_type: ConflictType


@dataclass(slots=True)
class RuntimeMetrics:
    route_ms: int
    sql_ms: int
    docs_ms: int
    synth_ms: int
    guard_ms: int
    total_ms: int


@dataclass(slots=True)
class AgentResponse:
    request_id: str
    route: Route
    answer: str
    confidence: float
    reason: str
    citations: list[str]
    sql_rows: list[dict[str, Any]] | None
    evidence: list[EvidenceItem]
    guardrails: GuardrailReport
    metrics: RuntimeMetrics


@dataclass(slots=True)
class PolicySnapshot:
    min_route_confidence: float
    min_docs_score: float
    min_evidence_coverage: float
    min_output_confidence: float
    max_sql_cost: float
    max_sql_rows: int
    max_docs_chunks: int
    blocked_sql_patterns: list[str]
    required_doc_keywords: list[str]


class RouterModel(Protocol):
    def classify(self, query: str) -> dict[str, Any]: ...


class SQLPlanner(Protocol):
    def plan(self, query: str, *, schema: dict[str, Any]) -> SQLPlan: ...


class SQLExecutor(Protocol):
    def execute(self, plan: SQLPlan) -> SQLExecution: ...


class DocsRetriever(Protocol):
    def retrieve(self, query: str, *, top_k: int, min_score: float) -> DocsRetrieval: ...


class Synthesizer(Protocol):
    def synthesize(
        self,
        *,
        query: str,
        route: Route,
        sql_result: SQLExecution | None,
        docs_result: DocsRetrieval | None,
        evidence: Sequence[EvidenceItem],
    ) -> SynthesisResult: ...


class TraceWriter(Protocol):
    def write(self, event_type: str, payload: dict[str, Any]) -> None: ...


class MetricsSink(Protocol):
    def observe(self, metric: str, value: float, tags: dict[str, str] | None = None) -> None: ...


class PolicyStore(Protocol):
    def get(self) -> PolicySnapshot: ...


class SchemaRegistry(Protocol):
    def get_schema(self) -> dict[str, Any]: ...


def now_ms() -> int:
    return int(time.time() * 1000)


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def percentile(values: Sequence[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(clamp(round((len(ordered) - 1) * q), 0, len(ordered) - 1))
    return ordered[idx]


def normalize_query(query: str) -> str:
    compact = re.sub(r"\s+", " ", query.strip())
    return compact


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_\-]{2,}", text.lower())


def contains_any(text: str, terms: Iterable[str]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def parse_route(raw: dict[str, Any]) -> RouteDecision:
    route_raw = str(raw.get("route", Route.CLARIFY.value)).lower()
    confidence = float(raw.get("confidence", 0.0))
    reason = str(raw.get("reason", ""))

    if route_raw == Route.SQL.value:
        route = Route.SQL
    elif route_raw == Route.DOCS.value:
        route = Route.DOCS
    elif route_raw == Route.HYBRID.value:
        route = Route.HYBRID
    else:
        route = Route.CLARIFY

    requires_sql = route in {Route.SQL, Route.HYBRID}
    requires_docs = route in {Route.DOCS, Route.HYBRID}

    return RouteDecision(
        route=route,
        confidence=clamp(confidence, 0.0, 1.0),
        reason=reason,
        requires_sql=requires_sql,
        requires_docs=requires_docs,
        requires_clarification=(route == Route.CLARIFY),
        notes=[],
    )


def infer_route_hint(query: str) -> Route:
    lower = query.lower()

    sql_terms = [
        "count",
        "how many",
        "total",
        "average",
        "sum",
        "revenue",
        "orders",
        "last month",
        "trend",
        "top",
    ]
    docs_terms = [
        "policy",
        "procedure",
        "guideline",
        "how to",
        "what is the rule",
        "onboarding",
        "handbook",
        "documentation",
    ]

    has_sql = contains_any(lower, sql_terms)
    has_docs = contains_any(lower, docs_terms)

    if has_sql and has_docs:
        return Route.HYBRID
    if has_sql:
        return Route.SQL
    if has_docs:
        return Route.DOCS
    return Route.CLARIFY


def refine_route(decision: RouteDecision, query: str, policy: PolicySnapshot) -> RouteDecision:
    hint = infer_route_hint(query)

    if decision.confidence < policy.min_route_confidence:
        decision.notes.append("route_confidence_below_policy")
        if hint != Route.CLARIFY:
            decision.route = hint
            decision.requires_sql = decision.route in {Route.SQL, Route.HYBRID}
            decision.requires_docs = decision.route in {Route.DOCS, Route.HYBRID}
            decision.requires_clarification = decision.route == Route.CLARIFY
            decision.reason = f"low_confidence_router_fallback_to_{hint.value}"
        else:
            decision.route = Route.CLARIFY
            decision.requires_sql = False
            decision.requires_docs = False
            decision.requires_clarification = True
            decision.reason = "low_confidence_with_no_clear_hint"

    if len(tokenize(query)) < 4:
        decision.route = Route.CLARIFY
        decision.requires_sql = False
        decision.requires_docs = False
        decision.requires_clarification = True
        decision.reason = "query_too_short_for_safe_dispatch"
        decision.notes.append("query_too_short")

    return decision


def unsafe_sql_patterns(policy: PolicySnapshot) -> list[re.Pattern[str]]:
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in policy.blocked_sql_patterns]


def check_sql_plan_safety(plan: SQLPlan, policy: PolicySnapshot) -> tuple[bool, list[str]]:
    violations: list[str] = []

    if not plan.read_only:
        violations.append("sql_plan_not_read_only")

    if plan.estimated_cost > policy.max_sql_cost:
        violations.append("sql_plan_cost_above_policy")

    normalized = " ".join(plan.sql.strip().split())

    for pattern in unsafe_sql_patterns(policy):
        if pattern.search(normalized):
            violations.append(f"blocked_pattern:{pattern.pattern}")

    if "select" not in normalized.lower():
        violations.append("sql_missing_select")

    if " limit " not in f" {normalized.lower()} ":
        violations.append("sql_missing_limit")

    return (len(violations) == 0, violations)


def trim_sql_rows(rows: Sequence[dict[str, Any]], max_rows: int) -> tuple[list[dict[str, Any]], bool]:
    if len(rows) <= max_rows:
        return (list(rows), False)
    return (list(rows[:max_rows]), True)


def summarize_sql(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "No rows returned."

    cols = sorted({key for row in rows for key in row.keys()})
    numeric_values: list[float] = []

    for row in rows:
        for value in row.values():
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))

    summary_parts = [f"rows={len(rows)}", f"columns={','.join(cols)[:160]}"]

    if numeric_values:
        summary_parts.append(f"min={min(numeric_values):.2f}")
        summary_parts.append(f"max={max(numeric_values):.2f}")

    return "; ".join(summary_parts)


def sanitize_docs_chunks(chunks: Sequence[RetrievalChunk], max_chunks: int) -> list[RetrievalChunk]:
    seen: set[str] = set()
    sanitized: list[RetrievalChunk] = []

    for chunk in chunks:
        signature = f"{chunk.source_id}:{chunk.section}:{chunk.text[:120].lower()}"
        if signature in seen:
            continue
        seen.add(signature)
        sanitized.append(chunk)
        if len(sanitized) >= max_chunks:
            break

    return sanitized


def docs_score(chunks: Sequence[RetrievalChunk]) -> tuple[float, float]:
    if not chunks:
        return (0.0, 0.0)
    scores = [chunk.score for chunk in chunks]
    return (float(fmean(scores)), max(scores))


def enforce_docs_policy(docs: DocsRetrieval, policy: PolicySnapshot) -> tuple[DocsRetrieval, list[str]]:
    notes: list[str] = []
    cleaned = sanitize_docs_chunks(docs.chunks, policy.max_docs_chunks)
    avg_score, max_score = docs_score(cleaned)

    if not cleaned:
        notes.append("docs_empty_after_dedup")

    if max_score < policy.min_docs_score:
        notes.append("docs_max_score_below_policy")

    normalized_query = docs.query.lower()
    for kw in policy.required_doc_keywords:
        if kw in normalized_query and not any(kw in chunk.text.lower() for chunk in cleaned):
            notes.append(f"required_keyword_not_supported:{kw}")

    citations = [chunk.source_id for chunk in cleaned]

    return (
        DocsRetrieval(
            query=docs.query,
            chunks=cleaned,
            citations=citations,
            avg_score=round(avg_score, 4),
            max_score=round(max_score, 4),
            latency_ms=docs.latency_ms,
        ),
        notes,
    )


def build_sql_evidence(sql_result: SQLExecution | None) -> list[EvidenceItem]:
    if sql_result is None:
        return []

    payload = {
        "query_id": sql_result.query_id,
        "row_count": sql_result.row_count,
        "summary": sql_result.summary,
        "rows_preview": sql_result.rows[:4],
    }

    return [
        EvidenceItem(
            evidence_id=f"sql:{sql_result.query_id}",
            origin="sql",
            label="Structured SQL Result",
            payload=payload,
            confidence=0.93,
        )
    ]


def build_docs_evidence(docs_result: DocsRetrieval | None) -> list[EvidenceItem]:
    if docs_result is None:
        return []

    items: list[EvidenceItem] = []

    for idx, chunk in enumerate(docs_result.chunks, start=1):
        items.append(
            EvidenceItem(
                evidence_id=f"docs:{chunk.chunk_id}:{idx}",
                origin="docs",
                label=f"{chunk.title} [{chunk.section}]",
                payload={
                    "source_id": chunk.source_id,
                    "score": round(chunk.score, 4),
                    "snippet": chunk.text[:260],
                },
                confidence=round(clamp(chunk.score, 0.0, 1.0), 4),
            )
        )

    return items


def detect_numeric_values(text: str) -> list[float]:
    values: list[float] = []
    for token in re.findall(r"[-+]?\d+(?:\.\d+)?", text):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def detect_conflict(sql_result: SQLExecution | None, docs_result: DocsRetrieval | None, answer: str) -> tuple[ConflictType, list[str]]:
    notes: list[str] = []

    if sql_result is None and docs_result is None:
        return (ConflictType.NONE, notes)

    if sql_result is not None and docs_result is None:
        return (ConflictType.NONE, notes)

    if sql_result is None and docs_result is not None:
        return (ConflictType.NONE, notes)

    sql_numbers: list[float] = []
    for row in sql_result.rows:
        for value in row.values():
            if isinstance(value, (int, float)):
                sql_numbers.append(float(value))

    answer_numbers = detect_numeric_values(answer)

    if sql_numbers and answer_numbers:
        sql_mean = fmean(sql_numbers)
        ans_mean = fmean(answer_numbers)
        if abs(sql_mean - ans_mean) > max(1.0, abs(sql_mean) * 0.18):
            notes.append("answer_numeric_values_diverge_from_sql")
            return (ConflictType.NUMERIC_MISMATCH, notes)

    docs_terms = set(tokenize(" ".join(chunk.text for chunk in docs_result.chunks)))
    answer_terms = set(tokenize(answer))
    overlap = safe_div(len(answer_terms & docs_terms), max(1, len(answer_terms)))

    if overlap < 0.18:
        notes.append("answer_has_low_docs_overlap")
        return (ConflictType.MISSING_DOC_SUPPORT, notes)

    if contains_any(answer.lower(), ["policy allows", "policy forbids"]) and not contains_any(
        " ".join(chunk.text.lower() for chunk in docs_result.chunks),
        ["allow", "forbid", "must", "required"],
    ):
        notes.append("policy_claim_without_matching_docs_language")
        return (ConflictType.POLICY_MISMATCH, notes)

    return (ConflictType.NONE, notes)


def evidence_coverage(query: str, evidence: Sequence[EvidenceItem]) -> float:
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0

    evidence_text = " ".join(
        f"{item.label} {json.dumps(item.payload, ensure_ascii=True)}" for item in evidence
    )
    ev_tokens = set(tokenize(evidence_text))
    return round(clamp(safe_div(len(q_tokens & ev_tokens), len(q_tokens)), 0.0, 1.0), 4)


def score_output_confidence(
    decision: RouteDecision,
    docs_result: DocsRetrieval | None,
    sql_result: SQLExecution | None,
    synthesis: SynthesisResult,
    coverage: float,
) -> float:
    route_component = decision.confidence
    docs_component = docs_result.max_score if docs_result else 0.8
    sql_component = 0.9 if sql_result and sql_result.row_count > 0 else 0.65 if sql_result else 0.75
    conflict_penalty = 0.22 if synthesis.conflict_type != ConflictType.NONE else 0.0

    score = (
        0.28 * route_component
        + 0.24 * docs_component
        + 0.22 * sql_component
        + 0.16 * synthesis.confidence
        + 0.10 * coverage
        - conflict_penalty
    )

    return round(clamp(score, 0.0, 1.0), 4)


def build_guardrail_report(
    *,
    decision: RouteDecision,
    policy: PolicySnapshot,
    docs_result: DocsRetrieval | None,
    synthesis: SynthesisResult,
    coverage: float,
    final_confidence: float,
) -> GuardrailReport:
    reasons: list[GuardReason] = []

    if decision.confidence < policy.min_route_confidence:
        reasons.append(GuardReason.ROUTE_CONFIDENCE_LOW)

    if decision.requires_docs and docs_result is not None and docs_result.max_score < policy.min_docs_score:
        reasons.append(GuardReason.DOCS_EVIDENCE_WEAK)

    if synthesis.conflict_type != ConflictType.NONE:
        reasons.append(GuardReason.CONFLICT_UNRESOLVED)

    if coverage < policy.min_evidence_coverage:
        reasons.append(GuardReason.OUTPUT_UNSUPPORTED)

    if final_confidence < policy.min_output_confidence:
        reasons.append(GuardReason.OUTPUT_UNSUPPORTED)

    accepted = len(reasons) == 0

    return GuardrailReport(
        accepted=accepted,
        reasons=reasons,
        confidence=final_confidence,
        route_confidence=decision.confidence,
        evidence_coverage=coverage,
        conflict_type=synthesis.conflict_type,
    )


def build_clarification(reason: str) -> str:
    return (
        "I need one clarification before answering safely. "
        f"Reason: {reason}. "
        "Do you need numeric metrics from database, documentation policy, or both?"
    )


def fallback_response(
    *,
    env: QueryEnvelope,
    route: Route,
    reason: str,
    guardrails: GuardrailReport,
    metrics: RuntimeMetrics,
) -> AgentResponse:
    return AgentResponse(
        request_id=env.request_id,
        route=route,
        answer=build_clarification(reason),
        confidence=0.0,
        reason=reason,
        citations=[],
        sql_rows=None,
        evidence=[],
        guardrails=guardrails,
        metrics=metrics,
    )


def finalize_response(
    *,
    env: QueryEnvelope,
    decision: RouteDecision,
    sql_result: SQLExecution | None,
    docs_result: DocsRetrieval | None,
    synthesis: SynthesisResult,
    evidence: list[EvidenceItem],
    guardrails: GuardrailReport,
    metrics: RuntimeMetrics,
) -> AgentResponse:
    citations = docs_result.citations if docs_result else []

    return AgentResponse(
        request_id=env.request_id,
        route=decision.route,
        answer=synthesis.answer,
        confidence=guardrails.confidence,
        reason=decision.reason,
        citations=citations,
        sql_rows=sql_result.rows if sql_result else None,
        evidence=evidence,
        guardrails=guardrails,
        metrics=metrics,
    )


def answer_with_hybrid_router(
    *,
    env: QueryEnvelope,
    router: RouterModel,
    sql_planner: SQLPlanner,
    sql_executor: SQLExecutor,
    docs_retriever: DocsRetriever,
    synthesizer: Synthesizer,
    schema_registry: SchemaRegistry,
    policy_store: PolicyStore,
    traces: TraceWriter,
    metrics: MetricsSink,
) -> AgentResponse:
    t_total = time.perf_counter()

    query = normalize_query(env.query)
    policy = policy_store.get()

    t_route = time.perf_counter()
    route_raw = router.classify(query)
    decision = parse_route(route_raw)
    decision = refine_route(decision, query, policy)
    route_ms = int((time.perf_counter() - t_route) * 1000)

    sql_result: SQLExecution | None = None
    docs_result: DocsRetrieval | None = None
    sql_ms = 0
    docs_ms = 0

    if decision.requires_sql:
        t_sql = time.perf_counter()
        schema = schema_registry.get_schema()
        plan = sql_planner.plan(query, schema=schema)
        safe, violations = check_sql_plan_safety(plan, policy)

        if not safe:
            decision.route = Route.CLARIFY
            decision.requires_sql = False
            decision.requires_docs = False
            decision.requires_clarification = True
            decision.reason = "sql_plan_blocked_by_policy"
            decision.notes.extend(violations)
        else:
            executed = sql_executor.execute(plan)
            trimmed_rows, truncated = trim_sql_rows(executed.rows, policy.max_sql_rows)
            sql_result = SQLExecution(
                query_id=executed.query_id,
                rows=trimmed_rows,
                row_count=len(trimmed_rows),
                latency_ms=executed.latency_ms,
                truncated=truncated,
                summary=summarize_sql(trimmed_rows),
            )

        sql_ms = int((time.perf_counter() - t_sql) * 1000)

    if decision.requires_docs:
        t_docs = time.perf_counter()
        raw_docs = docs_retriever.retrieve(query, top_k=12, min_score=policy.min_docs_score)
        docs_result, docs_notes = enforce_docs_policy(raw_docs, policy)
        decision.notes.extend(docs_notes)
        docs_ms = int((time.perf_counter() - t_docs) * 1000)

    t_synth = time.perf_counter()

    if decision.requires_clarification:
        synth = SynthesisResult(
            answer=build_clarification(decision.reason),
            confidence=0.0,
            conflict_type=ConflictType.NONE,
            conflict_notes=[],
            sections=["clarification"],
        )
    else:
        sql_evidence = build_sql_evidence(sql_result)
        docs_evidence = build_docs_evidence(docs_result)
        evidence = sql_evidence + docs_evidence
        synth = synthesizer.synthesize(
            query=query,
            route=decision.route,
            sql_result=sql_result,
            docs_result=docs_result,
            evidence=evidence,
        )

    synth_ms = int((time.perf_counter() - t_synth) * 1000)

    t_guard = time.perf_counter()

    if decision.requires_clarification:
        coverage = 0.0
        final_conf = 0.0
        guard = GuardrailReport(
            accepted=False,
            reasons=[GuardReason.ROUTE_CONFIDENCE_LOW],
            confidence=0.0,
            route_confidence=decision.confidence,
            evidence_coverage=0.0,
            conflict_type=ConflictType.NONE,
        )
        evidence_final: list[EvidenceItem] = []
    else:
        conflict_type, conflict_notes = detect_conflict(sql_result, docs_result, synth.answer)
        if conflict_type != ConflictType.NONE:
            synth.conflict_type = conflict_type
            synth.conflict_notes.extend(conflict_notes)

        evidence_final = build_sql_evidence(sql_result) + build_docs_evidence(docs_result)
        coverage = evidence_coverage(query, evidence_final)

        final_conf = score_output_confidence(
            decision,
            docs_result,
            sql_result,
            synth,
            coverage,
        )

        guard = build_guardrail_report(
            decision=decision,
            policy=policy,
            docs_result=docs_result,
            synthesis=synth,
            coverage=coverage,
            final_confidence=final_conf,
        )

    guard_ms = int((time.perf_counter() - t_guard) * 1000)
    total_ms = int((time.perf_counter() - t_total) * 1000)

    runtime_metrics = RuntimeMetrics(
        route_ms=route_ms,
        sql_ms=sql_ms,
        docs_ms=docs_ms,
        synth_ms=synth_ms,
        guard_ms=guard_ms,
        total_ms=total_ms,
    )

    if not guard.accepted:
        response = fallback_response(
            env=env,
            route=decision.route,
            reason=decision.reason,
            guardrails=guard,
            metrics=runtime_metrics,
        )
    else:
        response = finalize_response(
            env=env,
            decision=decision,
            sql_result=sql_result,
            docs_result=docs_result,
            synthesis=synth,
            evidence=evidence_final,
            guardrails=guard,
            metrics=runtime_metrics,
        )

    traces.write(
        "hybrid_agent.query.completed",
        {
            "request_id": env.request_id,
            "route": decision.route.value,
            "route_reason": decision.reason,
            "route_confidence": decision.confidence,
            "accepted": response.guardrails.accepted,
            "guard_reasons": [reason.value for reason in response.guardrails.reasons],
            "conflict_type": response.guardrails.conflict_type.value,
            "evidence_coverage": response.guardrails.evidence_coverage,
            "confidence": response.confidence,
            "latency_ms": response.metrics.total_ms,
            "notes": decision.notes,
        },
    )

    metrics.observe("hybrid.total_ms", float(response.metrics.total_ms), {"route": decision.route.value})
    metrics.observe("hybrid.confidence", float(response.confidence), {"route": decision.route.value})
    metrics.observe("hybrid.evidence_coverage", float(response.guardrails.evidence_coverage))

    return response


class StaticPolicyStore:
    def __init__(self, policy: PolicySnapshot | None = None) -> None:
        self.policy = policy or PolicySnapshot(
            min_route_confidence=0.62,
            min_docs_score=0.60,
            min_evidence_coverage=0.22,
            min_output_confidence=0.63,
            max_sql_cost=2200.0,
            max_sql_rows=50,
            max_docs_chunks=8,
            blocked_sql_patterns=[
                r"\b(drop|truncate|delete|insert|update|alter|create|grant|revoke)\b",
                r"--",
                r"/\*",
            ],
            required_doc_keywords=["policy", "procedure", "compliance"],
        )

    def get(self) -> PolicySnapshot:
        return self.policy


class InMemoryTrace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"event_type": event_type, "payload": payload, "ts": now_ms()})


class NoopMetrics:
    def observe(self, metric: str, value: float, tags: dict[str, str] | None = None) -> None:
        del metric
        del value
        del tags
