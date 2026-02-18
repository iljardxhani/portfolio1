from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from statistics import fmean
from typing import Any
from typing import Iterable
from typing import Protocol
from typing import Sequence
import random
import re
import time
class CaseSeverity(str, Enum):
    NORMAL = "normal"
    IMPORTANT = "important"
    CRITICAL = "critical"
class FailureKind(str, Enum):
    NONE = "none"
    RETRIEVAL_MISS = "retrieval_miss"
    LOW_RELEVANCE = "low_relevance"
    LOW_FAITHFULNESS = "low_faithfulness"
    CITATION_MISMATCH = "citation_mismatch"
    LATENCY_BUDGET = "latency_budget"
    EMPTY_ANSWER = "empty_answer"
    FORMAT_ISSUE = "format_issue"
class DriftSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
@dataclass(slots=True)
class EvalCase:
    case_id: str
    query: str
    expected_answer: str
    required_source_ids: set[str]
    evidence_passages: list[str]
    tags: set[str] = field(default_factory=set)
    severity: CaseSeverity = CaseSeverity.NORMAL
    language: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)
@dataclass(slots=True)
class QueryResponse:
    answer: str
    citations: list[str]
    source_ids: list[str]
    latency_ms: int
    raw: dict[str, Any] = field(default_factory=dict)
@dataclass(slots=True)
class CaseMetrics:
    relevance: float
    faithfulness: float
    citation_precision: float
    citation_recall: float
    citation_f1: float
    hallucination_rate: float
    latency_ms: int
    answer_length: int
    claim_count: int
@dataclass(slots=True)
class EvalResult:
    case_id: str
    severity: CaseSeverity
    tags: set[str]
    metrics: CaseMetrics
    passed: bool
    failure_kind: FailureKind
    notes: list[str]
    answer_preview: str
@dataclass(slots=True)
class SegmentSummary:
    segment: str
    sample_count: int
    pass_rate: float
    avg_relevance: float
    avg_faithfulness: float
    avg_citation_f1: float
    avg_hallucination_rate: float
    p95_latency_ms: int
@dataclass(slots=True)
class BaselineSnapshot:
    suite_name: str
    dataset_version: str
    quality_index: float
    pass_rate: float
    avg_faithfulness: float
    avg_citation_precision: float
    p95_latency_ms: int
    metric_map: dict[str, float]
    updated_at_ms: int
@dataclass(slots=True)
class EvalSummary:
    run_id: str
    suite_name: str
    dataset_version: str
    model_id: str
    retriever_id: str
    created_at_ms: int
    sample_count: int
    pass_rate: float
    avg_relevance: float
    avg_faithfulness: float
    avg_citation_precision: float
    avg_citation_recall: float
    avg_citation_f1: float
    avg_hallucination_rate: float
    p95_latency_ms: int
    quality_index: float
    regression_delta: float
    critical_failures: int
    segment_summaries: list[SegmentSummary]
    confidence_intervals: dict[str, tuple[float, float]]
    top_failure_kinds: list[tuple[str, int]]
@dataclass(slots=True)
class GatePolicy:
    max_quality_regression: float = -0.02
    min_pass_rate: float = 0.90
    min_faithfulness: float = 0.88
    min_citation_precision: float = 0.90
    max_p95_latency_ms: int = 1900
    max_hallucination_rate: float = 0.08
    max_critical_failures: int = 0
    required_segment_pass_rate: dict[str, float] = field(default_factory=lambda: {"core": 0.92})
    allow_warning_only: bool = False
@dataclass(slots=True)
class GateDecision:
    allow_release: bool
    reasons: list[str]
    warnings: list[str]
    blocked_case_ids: list[str]
    quality_regression: float
    policy_snapshot: dict[str, Any]
@dataclass(slots=True)
class DriftSignal:
    metric: str
    baseline_value: float
    current_value: float
    delta: float
    severity: DriftSeverity
    description: str
@dataclass(slots=True)
class FailureBucket:
    kind: FailureKind
    count: int
    case_ids: list[str]
@dataclass(slots=True)
class RunContext:
    run_id: str
    suite_name: str
    dataset_version: str
    model_id: str
    retriever_id: str
    channel: str
    trigger: str
    created_at_ms: int
    seed: int = 17
    metadata: dict[str, Any] = field(default_factory=dict)
class RAGClient(Protocol):
    def ask(
        self,
        query: str,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
class DatasetStore(Protocol):
    def get_cases(self, suite_name: str, dataset_version: str) -> list[EvalCase]: ...
class BaselineStore(Protocol):
    def get_baseline(self, suite_name: str, dataset_version: str) -> BaselineSnapshot | None: ...
    def save_baseline(self, snapshot: BaselineSnapshot) -> None: ...
class ArtifactStore(Protocol):
    def write_text(self, path: str, text: str) -> None: ...
    def write_json(self, path: str, payload: dict[str, Any]) -> None: ...
class TraceWriter(Protocol):
    def write(self, event_type: str, payload: dict[str, Any]) -> None: ...
def now_ms() -> int:
    return int(time.time() * 1000)
def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den
def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
def percentile(values: Sequence[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(clamp(round((len(ordered) - 1) * p), 0, len(ordered) - 1))
    return ordered[idx]
def normalize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    return compact.lower()
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_\-]{2,}", normalize_text(text))
def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
def to_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return slug.strip("-") or "unknown"
def metric_mean(values: Iterable[float]) -> float:
    seq = list(values)
    return float(fmean(seq)) if seq else 0.0
def metric_ci(values: Sequence[float], *, seed: int, rounds: int = 280) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(rounds):
        sample = [values[rng.randrange(0, len(values))] for _ in range(len(values))]
        means.append(metric_mean(sample))
    means.sort()
    low_idx = int(0.05 * (len(means) - 1))
    high_idx = int(0.95 * (len(means) - 1))
    return (round(means[low_idx], 4), round(means[high_idx], 4))
def extract_keywords(text: str, *, top_k: int = 12) -> list[str]:
    freq: dict[str, int] = {}
    for token in tokenize(text):
        if len(token) < 4:
            continue
        freq[token] = freq.get(token, 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:top_k]]
def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    return safe_div(len(sa & sb), len(sa | sb))
def token_f1(answer: str, expected: str) -> float:
    answer_tokens = tokenize(answer)
    expected_tokens = tokenize(expected)
    if not answer_tokens and not expected_tokens:
        return 1.0
    if not answer_tokens or not expected_tokens:
        return 0.0
    answer_set = set(answer_tokens)
    expected_set = set(expected_tokens)
    precision = safe_div(len(answer_set & expected_set), len(answer_set))
    recall = safe_div(len(answer_set & expected_set), len(expected_set))
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)
def keyword_overlap_score(answer: str, expected: str) -> float:
    expected_kw = extract_keywords(expected)
    if not expected_kw:
        return 0.0
    answer_tokens = set(tokenize(answer))
    hits = sum(1 for kw in expected_kw if kw in answer_tokens)
    return safe_div(hits, len(expected_kw))
def score_relevance(answer: str, expected: str) -> float:
    f1 = token_f1(answer, expected)
    keyword = keyword_overlap_score(answer, expected)
    jacc = jaccard_similarity(tokenize(answer), tokenize(expected))
    return round(clamp(0.5 * f1 + 0.35 * keyword + 0.15 * jacc, 0.0, 1.0), 4)
def normalize_source_id(source_id: str) -> str:
    return to_slug(source_id)
def score_citation_precision(citations: Sequence[str], required_source_ids: set[str]) -> float:
    if not citations:
        return 0.0
    required = {normalize_source_id(item) for item in required_source_ids}
    observed = [normalize_source_id(item) for item in citations]
    hits = sum(1 for item in observed if item in required)
    return round(safe_div(hits, len(observed)), 4)
def score_citation_recall(citations: Sequence[str], required_source_ids: set[str]) -> float:
    if not required_source_ids:
        return 1.0
    required = {normalize_source_id(item) for item in required_source_ids}
    observed = {normalize_source_id(item) for item in citations}
    return round(safe_div(len(required & observed), len(required)), 4)
def score_citation_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)
def lexical_support(sentence: str, evidence_passages: Sequence[str]) -> float:
    sent_tokens = tokenize(sentence)
    if not sent_tokens:
        return 0.0
    best = 0.0
    for passage in evidence_passages:
        overlap = jaccard_similarity(sent_tokens, tokenize(passage))
        best = max(best, overlap)
    return best
def score_faithfulness(answer: str, citations: Sequence[str], evidence_passages: Sequence[str]) -> float:
    claims = sentence_split(answer)
    if not claims:
        return 0.0
    support_scores: list[float] = []
    citation_bonus = clamp(len(citations) / max(1, len(claims)), 0.0, 1.0)
    for claim in claims:
        support_scores.append(lexical_support(claim, evidence_passages))
    mean_support = metric_mean(support_scores)
    final = 0.75 * mean_support + 0.25 * citation_bonus
    return round(clamp(final, 0.0, 1.0), 4)
def estimate_hallucination_rate(answer: str, evidence_passages: Sequence[str]) -> float:
    claims = sentence_split(answer)
    if not claims:
        return 1.0
    unsupported = 0
    for claim in claims:
        if lexical_support(claim, evidence_passages) < 0.12:
            unsupported += 1
    return round(clamp(safe_div(unsupported, len(claims)), 0.0, 1.0), 4)
def has_minimum_format(answer: str) -> bool:
    if not answer.strip():
        return False
    too_short = len(answer.strip()) < 30
    if too_short:
        return False
    return True
def case_thresholds(severity: CaseSeverity) -> dict[str, float | int]:
    if severity == CaseSeverity.CRITICAL:
        return {
            "min_relevance": 0.66,
            "min_faithfulness": 0.82,
            "min_citation_f1": 0.74,
            "max_latency_ms": 2200,
        }
    if severity == CaseSeverity.IMPORTANT:
        return {
            "min_relevance": 0.60,
            "min_faithfulness": 0.76,
            "min_citation_f1": 0.68,
            "max_latency_ms": 2500,
        }
    return {
        "min_relevance": 0.54,
        "min_faithfulness": 0.70,
        "min_citation_f1": 0.60,
        "max_latency_ms": 2800,
    }
def classify_failure(metrics: CaseMetrics, passed: bool, format_ok: bool) -> FailureKind:
    if passed:
        return FailureKind.NONE
    if not format_ok:
        return FailureKind.FORMAT_ISSUE
    if metrics.answer_length == 0:
        return FailureKind.EMPTY_ANSWER
    if metrics.citation_f1 < 0.60:
        return FailureKind.CITATION_MISMATCH
    if metrics.faithfulness < 0.70:
        return FailureKind.LOW_FAITHFULNESS
    if metrics.relevance < 0.55:
        return FailureKind.LOW_RELEVANCE
    if metrics.latency_ms > 2600:
        return FailureKind.LATENCY_BUDGET
    if metrics.hallucination_rate > 0.20:
        return FailureKind.RETRIEVAL_MISS
    return FailureKind.LOW_RELEVANCE
def parse_response(raw: dict[str, Any]) -> QueryResponse:
    answer = str(raw.get("answer", ""))
    citations_raw = raw.get("citations", [])
    if isinstance(citations_raw, list):
        citations = [str(item) for item in citations_raw]
    else:
        citations = []
    sources_raw = raw.get("source_ids", raw.get("sources", []))
    if isinstance(sources_raw, list):
        source_ids = [str(item) for item in sources_raw]
    else:
        source_ids = []
    latency = raw.get("latency_ms", 0)
    try:
        latency_ms = int(latency)
    except (TypeError, ValueError):
        latency_ms = 0
    return QueryResponse(
        answer=answer,
        citations=citations,
        source_ids=source_ids,
        latency_ms=max(0, latency_ms),
        raw=raw,
    )
def score_case(response: QueryResponse, case: EvalCase) -> CaseMetrics:
    relevance = score_relevance(response.answer, case.expected_answer)
    precision = score_citation_precision(response.citations, case.required_source_ids)
    recall = score_citation_recall(response.citations, case.required_source_ids)
    citation_f1 = score_citation_f1(precision, recall)
    faithfulness = score_faithfulness(
        answer=response.answer,
        citations=response.citations,
        evidence_passages=case.evidence_passages,
    )
    hallucination_rate = estimate_hallucination_rate(response.answer, case.evidence_passages)
    claim_count = len(sentence_split(response.answer))
    return CaseMetrics(
        relevance=relevance,
        faithfulness=faithfulness,
        citation_precision=precision,
        citation_recall=recall,
        citation_f1=citation_f1,
        hallucination_rate=hallucination_rate,
        latency_ms=response.latency_ms,
        answer_length=len(response.answer.strip()),
        claim_count=claim_count,
    )
def case_passed(case: EvalCase, metrics: CaseMetrics, format_ok: bool) -> tuple[bool, list[str]]:
    notes: list[str] = []
    thresholds = case_thresholds(case.severity)
    min_rel = float(thresholds["min_relevance"])
    min_faith = float(thresholds["min_faithfulness"])
    min_cite = float(thresholds["min_citation_f1"])
    max_latency = int(thresholds["max_latency_ms"])
    if not format_ok:
        notes.append("format_contract_failed")
    if metrics.relevance < min_rel:
        notes.append("relevance_below_threshold")
    if metrics.faithfulness < min_faith:
        notes.append("faithfulness_below_threshold")
    if metrics.citation_f1 < min_cite:
        notes.append("citation_f1_below_threshold")
    if metrics.latency_ms > max_latency:
        notes.append("latency_above_threshold")
    if metrics.hallucination_rate > 0.18:
        notes.append("hallucination_rate_high")
    return (len(notes) == 0, notes)
def evaluate_case(client: RAGClient, case: EvalCase, run_ctx: RunContext) -> EvalResult:
    session_id = f"eval-{run_ctx.run_id}-{case.case_id}"
    start = time.perf_counter()
    raw = client.ask(
        case.query,
        session_id=session_id,
        metadata={
            "suite_name": run_ctx.suite_name,
            "dataset_version": run_ctx.dataset_version,
            "case_id": case.case_id,
            "tags": sorted(case.tags),
            "severity": case.severity.value,
        },
    )
    fallback_latency = int((time.perf_counter() - start) * 1000)
    response = parse_response(raw)
    if response.latency_ms == 0:
        response.latency_ms = fallback_latency
    metrics = score_case(response=response, case=case)
    format_ok = has_minimum_format(response.answer)
    passed, notes = case_passed(case, metrics, format_ok)
    failure = classify_failure(metrics, passed=passed, format_ok=format_ok)
    return EvalResult(
        case_id=case.case_id,
        severity=case.severity,
        tags=set(case.tags),
        metrics=metrics,
        passed=passed,
        failure_kind=failure,
        notes=notes,
        answer_preview=response.answer[:180],
    )
def grouped_by_segment(results: Sequence[EvalResult]) -> dict[str, list[EvalResult]]:
    grouped: dict[str, list[EvalResult]] = {}
    for result in results:
        if not result.tags:
            grouped.setdefault("untagged", []).append(result)
            continue
        for tag in sorted(result.tags):
            grouped.setdefault(tag, []).append(result)
    return grouped
def summarize_segment(segment: str, results: Sequence[EvalResult]) -> SegmentSummary:
    latencies = [item.metrics.latency_ms for item in results]
    return SegmentSummary(
        segment=segment,
        sample_count=len(results),
        pass_rate=round(metric_mean(1.0 if item.passed else 0.0 for item in results), 4),
        avg_relevance=round(metric_mean(item.metrics.relevance for item in results), 4),
        avg_faithfulness=round(metric_mean(item.metrics.faithfulness for item in results), 4),
        avg_citation_f1=round(metric_mean(item.metrics.citation_f1 for item in results), 4),
        avg_hallucination_rate=round(metric_mean(item.metrics.hallucination_rate for item in results), 4),
        p95_latency_ms=percentile(latencies, 0.95),
    )
def top_failure_buckets(results: Sequence[EvalResult], top_k: int = 5) -> list[FailureBucket]:
    buckets: dict[FailureKind, list[str]] = {}
    for result in results:
        buckets.setdefault(result.failure_kind, []).append(result.case_id)
    merged = [
        FailureBucket(kind=kind, count=len(case_ids), case_ids=sorted(case_ids))
        for kind, case_ids in buckets.items()
        if kind != FailureKind.NONE
    ]
    merged.sort(key=lambda item: item.count, reverse=True)
    return merged[:top_k]
def quality_index(
    *,
    pass_rate: float,
    avg_relevance: float,
    avg_faithfulness: float,
    avg_citation_precision: float,
    avg_citation_recall: float,
    avg_hallucination_rate: float,
    p95_latency_ms: int,
) -> float:
    latency_component = clamp(1.0 - (p95_latency_ms / 3000.0), 0.0, 1.0)
    hallucination_component = clamp(1.0 - avg_hallucination_rate, 0.0, 1.0)
    score = (
        0.22 * pass_rate
        + 0.18 * avg_relevance
        + 0.22 * avg_faithfulness
        + 0.16 * avg_citation_precision
        + 0.10 * avg_citation_recall
        + 0.06 * hallucination_component
        + 0.06 * latency_component
    )
    return round(clamp(score, 0.0, 1.0), 4)
def fallback_baseline(run_ctx: RunContext) -> BaselineSnapshot:
    return BaselineSnapshot(
        suite_name=run_ctx.suite_name,
        dataset_version=run_ctx.dataset_version,
        quality_index=0.84,
        pass_rate=0.90,
        avg_faithfulness=0.88,
        avg_citation_precision=0.89,
        p95_latency_ms=1650,
        metric_map={
            "avg_relevance": 0.85,
            "avg_citation_recall": 0.84,
            "avg_citation_f1": 0.865,
            "avg_hallucination_rate": 0.07,
        },
        updated_at_ms=run_ctx.created_at_ms,
    )
def summarize_results(
    run_ctx: RunContext,
    results: Sequence[EvalResult],
    baseline: BaselineSnapshot,
) -> EvalSummary:
    relevance = [item.metrics.relevance for item in results]
    faith = [item.metrics.faithfulness for item in results]
    cite_p = [item.metrics.citation_precision for item in results]
    cite_r = [item.metrics.citation_recall for item in results]
    cite_f1 = [item.metrics.citation_f1 for item in results]
    hallucination = [item.metrics.hallucination_rate for item in results]
    latencies = [item.metrics.latency_ms for item in results]
    pass_rate = metric_mean(1.0 if item.passed else 0.0 for item in results)
    avg_relevance = metric_mean(relevance)
    avg_faithfulness = metric_mean(faith)
    avg_citation_precision = metric_mean(cite_p)
    avg_citation_recall = metric_mean(cite_r)
    avg_citation_f1 = metric_mean(cite_f1)
    avg_hallucination_rate = metric_mean(hallucination)
    p95_latency_ms = percentile(latencies, 0.95)
    idx = quality_index(
        pass_rate=pass_rate,
        avg_relevance=avg_relevance,
        avg_faithfulness=avg_faithfulness,
        avg_citation_precision=avg_citation_precision,
        avg_citation_recall=avg_citation_recall,
        avg_hallucination_rate=avg_hallucination_rate,
        p95_latency_ms=p95_latency_ms,
    )
    regression_delta = round(idx - baseline.quality_index, 4)
    critical_failures = sum(1 for item in results if item.severity == CaseSeverity.CRITICAL and not item.passed)
    segments = [
        summarize_segment(segment=name, results=segment_results)
        for name, segment_results in sorted(grouped_by_segment(results).items())
    ]
    ci = {
        "avg_relevance": metric_ci(relevance, seed=run_ctx.seed + 11),
        "avg_faithfulness": metric_ci(faith, seed=run_ctx.seed + 17),
        "avg_citation_f1": metric_ci(cite_f1, seed=run_ctx.seed + 23),
        "pass_rate": metric_ci(
            [1.0 if item.passed else 0.0 for item in results],
            seed=run_ctx.seed + 31,
        ),
    }
    top_failures = [
        (bucket.kind.value, bucket.count)
        for bucket in top_failure_buckets(results, top_k=5)
    ]
    return EvalSummary(
        run_id=run_ctx.run_id,
        suite_name=run_ctx.suite_name,
        dataset_version=run_ctx.dataset_version,
        model_id=run_ctx.model_id,
        retriever_id=run_ctx.retriever_id,
        created_at_ms=run_ctx.created_at_ms,
        sample_count=len(results),
        pass_rate=round(pass_rate, 4),
        avg_relevance=round(avg_relevance, 4),
        avg_faithfulness=round(avg_faithfulness, 4),
        avg_citation_precision=round(avg_citation_precision, 4),
        avg_citation_recall=round(avg_citation_recall, 4),
        avg_citation_f1=round(avg_citation_f1, 4),
        avg_hallucination_rate=round(avg_hallucination_rate, 4),
        p95_latency_ms=p95_latency_ms,
        quality_index=idx,
        regression_delta=regression_delta,
        critical_failures=critical_failures,
        segment_summaries=segments,
        confidence_intervals=ci,
        top_failure_kinds=top_failures,
    )
def apply_release_gate(
    summary: EvalSummary,
    policy: GatePolicy,
    baseline: BaselineSnapshot,
    *,
    results: Sequence[EvalResult] | None = None,
) -> GateDecision:
    reasons: list[str] = []
    warnings: list[str] = []
    if summary.regression_delta < policy.max_quality_regression:
        reasons.append("quality_regression_exceeds_policy")
    if summary.pass_rate < policy.min_pass_rate:
        reasons.append("pass_rate_below_policy")
    if summary.avg_faithfulness < policy.min_faithfulness:
        reasons.append("faithfulness_below_policy")
    if summary.avg_citation_precision < policy.min_citation_precision:
        reasons.append("citation_precision_below_policy")
    if summary.p95_latency_ms > policy.max_p95_latency_ms:
        reasons.append("latency_budget_exceeded")
    if summary.avg_hallucination_rate > policy.max_hallucination_rate:
        reasons.append("hallucination_rate_exceeded")
    if summary.critical_failures > policy.max_critical_failures:
        reasons.append("critical_case_failures_exceeded")
    segment_map = {seg.segment: seg for seg in summary.segment_summaries}
    for seg_name, required_pass_rate in policy.required_segment_pass_rate.items():
        seg = segment_map.get(seg_name)
        if seg is None:
            warnings.append(f"segment_missing:{seg_name}")
            continue
        if seg.pass_rate < required_pass_rate:
            reasons.append(f"segment_pass_rate_below_policy:{seg_name}")
    blocked_case_ids: list[str] = []
    if results:
        blocked_case_ids = sorted([item.case_id for item in results if not item.passed])
    if policy.allow_warning_only and reasons:
        warnings.extend([f"would_block:{reason}" for reason in reasons])
        reasons = []
    return GateDecision(
        allow_release=(len(reasons) == 0),
        reasons=reasons,
        warnings=warnings,
        blocked_case_ids=blocked_case_ids,
        quality_regression=round(summary.quality_index - baseline.quality_index, 4),
        policy_snapshot={
            "max_quality_regression": policy.max_quality_regression,
            "min_pass_rate": policy.min_pass_rate,
            "min_faithfulness": policy.min_faithfulness,
            "min_citation_precision": policy.min_citation_precision,
            "max_p95_latency_ms": policy.max_p95_latency_ms,
            "max_hallucination_rate": policy.max_hallucination_rate,
            "max_critical_failures": policy.max_critical_failures,
            "required_segment_pass_rate": dict(policy.required_segment_pass_rate),
        },
    )
def detect_drift(summary: EvalSummary, baseline: BaselineSnapshot) -> list[DriftSignal]:
    probes: list[tuple[str, float, float]] = [
        ("quality_index", baseline.quality_index, summary.quality_index),
        ("pass_rate", baseline.pass_rate, summary.pass_rate),
        ("avg_faithfulness", baseline.avg_faithfulness, summary.avg_faithfulness),
        ("avg_citation_precision", baseline.avg_citation_precision, summary.avg_citation_precision),
        ("p95_latency_ms", float(baseline.p95_latency_ms), float(summary.p95_latency_ms)),
        (
            "avg_hallucination_rate",
            float(baseline.metric_map.get("avg_hallucination_rate", 0.0)),
            summary.avg_hallucination_rate,
        ),
    ]
    output: list[DriftSignal] = []
    for metric_name, baseline_value, current_value in probes:
        if metric_name == "p95_latency_ms":
            delta = current_value - baseline_value
            if delta > 280:
                sev = DriftSeverity.CRITICAL
            elif delta > 120:
                sev = DriftSeverity.WARNING
            else:
                sev = DriftSeverity.INFO
            description = f"Latency drift {delta:+.0f}ms against baseline."
        else:
            delta = current_value - baseline_value
            if delta < -0.05:
                sev = DriftSeverity.CRITICAL
            elif delta < -0.025:
                sev = DriftSeverity.WARNING
            else:
                sev = DriftSeverity.INFO
            description = f"Metric drift {delta:+.4f} against baseline."
        output.append(
            DriftSignal(
                metric=metric_name,
                baseline_value=round(float(baseline_value), 4),
                current_value=round(float(current_value), 4),
                delta=round(float(delta), 4),
                severity=sev,
                description=description,
            )
        )
    return output
def serialize_summary(summary: EvalSummary) -> dict[str, Any]:
    return {
        "run_id": summary.run_id,
        "suite_name": summary.suite_name,
        "dataset_version": summary.dataset_version,
        "model_id": summary.model_id,
        "retriever_id": summary.retriever_id,
        "created_at_ms": summary.created_at_ms,
        "sample_count": summary.sample_count,
        "pass_rate": summary.pass_rate,
        "avg_relevance": summary.avg_relevance,
        "avg_faithfulness": summary.avg_faithfulness,
        "avg_citation_precision": summary.avg_citation_precision,
        "avg_citation_recall": summary.avg_citation_recall,
        "avg_citation_f1": summary.avg_citation_f1,
        "avg_hallucination_rate": summary.avg_hallucination_rate,
        "p95_latency_ms": summary.p95_latency_ms,
        "quality_index": summary.quality_index,
        "regression_delta": summary.regression_delta,
        "critical_failures": summary.critical_failures,
        "segment_summaries": [
            {
                "segment": seg.segment,
                "sample_count": seg.sample_count,
                "pass_rate": seg.pass_rate,
                "avg_relevance": seg.avg_relevance,
                "avg_faithfulness": seg.avg_faithfulness,
                "avg_citation_f1": seg.avg_citation_f1,
                "avg_hallucination_rate": seg.avg_hallucination_rate,
                "p95_latency_ms": seg.p95_latency_ms,
            }
            for seg in summary.segment_summaries
        ],
        "confidence_intervals": {
            key: [low, high] for key, (low, high) in summary.confidence_intervals.items()
        },
        "top_failure_kinds": [
            {
                "kind": kind,
                "count": count,
            }
            for kind, count in summary.top_failure_kinds
        ],
    }
def serialize_decision(decision: GateDecision) -> dict[str, Any]:
    return {
        "allow_release": decision.allow_release,
        "reasons": decision.reasons,
        "warnings": decision.warnings,
        "blocked_case_ids": decision.blocked_case_ids,
        "quality_regression": decision.quality_regression,
        "policy_snapshot": decision.policy_snapshot,
    }
def serialize_drift(drift_signals: Sequence[DriftSignal]) -> dict[str, Any]:
    return {
        "signals": [
            {
                "metric": signal.metric,
                "baseline_value": signal.baseline_value,
                "current_value": signal.current_value,
                "delta": signal.delta,
                "severity": signal.severity.value,
                "description": signal.description,
            }
            for signal in drift_signals
        ]
    }
def render_markdown_report(
    *,
    run_ctx: RunContext,
    summary: EvalSummary,
    decision: GateDecision,
    baseline: BaselineSnapshot,
    drift_signals: Sequence[DriftSignal],
    failures: Sequence[FailureBucket],
) -> str:
    lines: list[str] = []
    lines.append(f"# RAG Evaluation Report - {run_ctx.suite_name}")
    lines.append("")
    lines.append(f"- Run ID: `{summary.run_id}`")
    lines.append(f"- Dataset Version: `{summary.dataset_version}`")
    lines.append(f"- Model: `{summary.model_id}`")
    lines.append(f"- Retriever: `{summary.retriever_id}`")
    lines.append(f"- Samples: `{summary.sample_count}`")
    lines.append("")
    lines.append("## Gate Decision")
    lines.append("")
    lines.append(f"- Allow Release: `{decision.allow_release}`")
    lines.append(f"- Quality Regression: `{decision.quality_regression:+.4f}`")
    if decision.reasons:
        lines.append(f"- Reasons: `{', '.join(decision.reasons)}`")
    else:
        lines.append("- Reasons: `none`")
    if decision.warnings:
        lines.append(f"- Warnings: `{', '.join(decision.warnings)}`")
    else:
        lines.append("- Warnings: `none`")
    lines.append("")
    lines.append("## Core Metrics")
    lines.append("")
    lines.append(f"- Pass Rate: `{summary.pass_rate:.4f}` (baseline `{baseline.pass_rate:.4f}`)")
    lines.append(
        f"- Faithfulness: `{summary.avg_faithfulness:.4f}` "
        f"(baseline `{baseline.avg_faithfulness:.4f}`)"
    )
    lines.append(
        f"- Citation Precision: `{summary.avg_citation_precision:.4f}` "
        f"(baseline `{baseline.avg_citation_precision:.4f}`)"
    )
    lines.append(f"- p95 Latency: `{summary.p95_latency_ms}ms` (baseline `{baseline.p95_latency_ms}ms`)")
    lines.append(f"- Hallucination Rate: `{summary.avg_hallucination_rate:.4f}`")
    lines.append("")
    lines.append("## Segment Breakdown")
    lines.append("")
    lines.append("| Segment | Samples | Pass Rate | Faithfulness | Citation F1 | p95 Latency |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for seg in summary.segment_summaries:
        lines.append(
            f"| {seg.segment} | {seg.sample_count} | {seg.pass_rate:.4f} | "
            f"{seg.avg_faithfulness:.4f} | {seg.avg_citation_f1:.4f} | {seg.p95_latency_ms}ms |"
        )
    lines.append("")
    lines.append("## Drift Signals")
    lines.append("")
    if not drift_signals:
        lines.append("No drift signals detected.")
    else:
        for signal in drift_signals:
            lines.append(
                f"- `{signal.metric}`: `{signal.delta:+.4f}` ({signal.severity.value}) - {signal.description}"
            )
    lines.append("")
    lines.append("## Top Failure Buckets")
    lines.append("")
    if not failures:
        lines.append("No case-level failures.")
    else:
        for bucket in failures:
            lines.append(f"- `{bucket.kind.value}`: {bucket.count} cases")
    lines.append("")
    return "\n".join(lines).strip() + "\n"
def report_path_prefix(run_ctx: RunContext) -> str:
    run_token = to_slug(f"{run_ctx.suite_name}-{run_ctx.run_id}")
    return f"reports/rag-eval/{run_token}"
def write_artifacts(
    *,
    run_ctx: RunContext,
    summary: EvalSummary,
    decision: GateDecision,
    baseline: BaselineSnapshot,
    drift_signals: Sequence[DriftSignal],
    failures: Sequence[FailureBucket],
    artifacts: ArtifactStore,
) -> None:
    base = report_path_prefix(run_ctx)
    report_md = render_markdown_report(
        run_ctx=run_ctx,
        summary=summary,
        decision=decision,
        baseline=baseline,
        drift_signals=drift_signals,
        failures=failures,
    )
    artifacts.write_text(f"{base}/report.md", report_md)
    artifacts.write_json(
        f"{base}/summary.json",
        {
            "summary": serialize_summary(summary),
            "decision": serialize_decision(decision),
            "drift": serialize_drift(drift_signals),
            "generated_at_ms": now_ms(),
        },
    )
    artifacts.write_json(
        f"{base}/gate.json",
        {
            "allow_release": decision.allow_release,
            "reasons": decision.reasons,
            "warnings": decision.warnings,
            "blocked_case_ids": decision.blocked_case_ids,
            "quality_regression": decision.quality_regression,
        },
    )
def evaluate_and_gate(
    run_ctx: RunContext,
    cases: Sequence[EvalCase],
    client: RAGClient,
    baselines: BaselineStore,
    policy: GatePolicy,
) -> tuple[EvalSummary, GateDecision]:
    baseline = baselines.get_baseline(run_ctx.suite_name, run_ctx.dataset_version)
    if baseline is None:
        baseline = fallback_baseline(run_ctx)
    results = [evaluate_case(client, case, run_ctx=run_ctx) for case in cases]
    summary = summarize_results(run_ctx=run_ctx, results=results, baseline=baseline)
    decision = apply_release_gate(summary=summary, policy=policy, baseline=baseline, results=results)
    if not decision.allow_release:
        # Upstream CI can catch this and stop promotion.
        raise RuntimeError("Evaluation gate failed: release blocked by policy.")
    return summary, decision
def run_suite(
    *,
    run_ctx: RunContext,
    dataset: DatasetStore,
    client: RAGClient,
    baselines: BaselineStore,
    artifacts: ArtifactStore,
    traces: TraceWriter,
    policy: GatePolicy | None = None,
    smoke_only: bool = False,
    smoke_sample_size: int = 40,
    promote_baseline_on_pass: bool = False,
) -> tuple[EvalSummary, GateDecision]:
    gate_policy = policy or GatePolicy()
    cases = dataset.get_cases(run_ctx.suite_name, run_ctx.dataset_version)
    if smoke_only:
        rng = random.Random(run_ctx.seed)
        sampled = list(cases)
        rng.shuffle(sampled)
        cases = sampled[: min(smoke_sample_size, len(sampled))]
    baseline = baselines.get_baseline(run_ctx.suite_name, run_ctx.dataset_version)
    if baseline is None:
        baseline = fallback_baseline(run_ctx)
    traces.write(
        "eval.run.started",
        {
            "run_id": run_ctx.run_id,
            "suite_name": run_ctx.suite_name,
            "dataset_version": run_ctx.dataset_version,
            "model_id": run_ctx.model_id,
            "retriever_id": run_ctx.retriever_id,
            "sample_count": len(cases),
            "smoke_only": smoke_only,
            "trigger": run_ctx.trigger,
        },
    )
    started = time.perf_counter()
    results: list[EvalResult] = []
    for idx, case in enumerate(cases, start=1):
        result = evaluate_case(client, case, run_ctx=run_ctx)
        results.append(result)
        if idx % 25 == 0:
            traces.write(
                "eval.run.progress",
                {
                    "run_id": run_ctx.run_id,
                    "processed": idx,
                    "total": len(cases),
                    "current_pass_rate": round(
                        metric_mean(1.0 if item.passed else 0.0 for item in results),
                        4,
                    ),
                },
            )
    summary = summarize_results(run_ctx=run_ctx, results=results, baseline=baseline)
    decision = apply_release_gate(summary=summary, policy=gate_policy, baseline=baseline, results=results)
    drift_signals = detect_drift(summary, baseline)
    failures = top_failure_buckets(results, top_k=8)
    write_artifacts(
        run_ctx=run_ctx,
        summary=summary,
        decision=decision,
        baseline=baseline,
        drift_signals=drift_signals,
        failures=failures,
        artifacts=artifacts,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    traces.write(
        "eval.run.completed",
        {
            "run_id": run_ctx.run_id,
            "allow_release": decision.allow_release,
            "elapsed_ms": elapsed_ms,
            "sample_count": summary.sample_count,
            "pass_rate": summary.pass_rate,
            "quality_index": summary.quality_index,
            "regression_delta": summary.regression_delta,
            "reasons": decision.reasons,
            "warnings": decision.warnings,
        },
    )
    if decision.allow_release and promote_baseline_on_pass:
        baselines.save_baseline(
            BaselineSnapshot(
                suite_name=summary.suite_name,
                dataset_version=summary.dataset_version,
                quality_index=summary.quality_index,
                pass_rate=summary.pass_rate,
                avg_faithfulness=summary.avg_faithfulness,
                avg_citation_precision=summary.avg_citation_precision,
                p95_latency_ms=summary.p95_latency_ms,
                metric_map={
                    "avg_relevance": summary.avg_relevance,
                    "avg_citation_recall": summary.avg_citation_recall,
                    "avg_citation_f1": summary.avg_citation_f1,
                    "avg_hallucination_rate": summary.avg_hallucination_rate,
                },
                updated_at_ms=now_ms(),
            )
        )
    if not decision.allow_release:
        raise RuntimeError(
            "Regression gate failed: "
            + ", ".join(decision.reasons)
            + f" | run_id={summary.run_id}"
        )
    return summary, decision
