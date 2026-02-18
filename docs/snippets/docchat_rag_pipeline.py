from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from statistics import mean
from typing import Any
from typing import Iterable
from typing import Protocol
from typing import Sequence
import hashlib
import math
import re
import time
import uuid
class Channel(str, Enum):
    API = "api"
    UI = "ui"
    INTERNAL = "internal"
@dataclass(slots=True)
class SourceDocument:
    source_id: str
    title: str
    uri: str
    revision: str
    text: str
    metadata: dict[str, Any]
@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    source_title: str
    revision: str
    page: int
    section: str
    text: str
    token_count: int
    checksum: str
@dataclass(slots=True)
class RetrievalCandidate:
    chunk_id: str
    source_id: str
    source_title: str
    revision: str
    page: int
    section: str
    text: str
    dense_score: float
    sparse_score: float
    rerank_score: float = 0.0
@dataclass(slots=True)
class RetrievalBundle:
    query: str
    dense_hits: list[RetrievalCandidate]
    sparse_hits: list[RetrievalCandidate]
    fused_hits: list[RetrievalCandidate]
    context: str
    retrieval_recall: float
@dataclass(slots=True)
class GenerationDraft:
    request_id: str
    answer: str
    model_name: str
    prompt_version: str
    token_usage: int
@dataclass(slots=True)
class CitationEvidence:
    token: str
    chunk_id: str
    source_id: str
    source_title: str
    revision: str
    page: int
    section: str
    confidence: float
@dataclass(slots=True)
class RuntimeMetrics:
    latency_ms: int
    retrieval_ms: int
    rerank_ms: int
    generation_ms: int
    citation_alignment_ms: int
    p95_latency_ms: int
@dataclass(slots=True)
class GuardrailReport:
    groundedness: float
    citation_precision: float
    citation_coverage: float
    low_confidence_spans: int
    empty_context: bool
    blocked: bool
    reasons: list[str]
@dataclass(slots=True)
class AnswerPayload:
    request_id: str
    answer: str
    citations: list[CitationEvidence]
    sources: list[dict[str, Any]]
    session_id: str
    confidence: float
    metrics: RuntimeMetrics
    guardrails: GuardrailReport
@dataclass(slots=True)
class QueryEnvelope:
    request_id: str
    session_id: str
    actor_id: str
    channel: Channel
    query: str
    locale: str
    created_at_ms: int
@dataclass(slots=True)
class EvalCase:
    case_id: str
    query: str
    reference_answer: str
    required_source_ids: set[str]
@dataclass(slots=True)
class EvalResult:
    case_id: str
    groundedness: float
    citation_precision: float
    answer_relevance: float
    passed: bool
@dataclass(slots=True)
class ReleaseDecision:
    allow_release: bool
    regression_delta: float
    reasons: list[str]
class EmbeddingModel(Protocol):
    def encode_query(self, text: str) -> list[float]: ...
class VectorIndex(Protocol):
    def upsert(self, chunks: Sequence[ChunkRecord]) -> None: ...
    def search(self, embedding: Sequence[float], top_k: int) -> list[RetrievalCandidate]: ...
class LexicalIndex(Protocol):
    def upsert(self, chunks: Sequence[ChunkRecord]) -> None: ...
    def search(self, query: str, top_k: int) -> list[RetrievalCandidate]: ...
class Reranker(Protocol):
    def score(self, query: str, candidates: Sequence[RetrievalCandidate]) -> list[float]: ...
class LLMGateway(Protocol):
    def generate(self, *, prompt: str, context: str, history: list[dict[str, str]]) -> GenerationDraft: ...
class SessionRepository(Protocol):
    def get_recent_turns(self, session_id: str, max_turns: int = 8) -> list[dict[str, str]]: ...
    def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> None: ...
class TraceRepository(Protocol):
    def write(self, event_type: str, payload: dict[str, Any]) -> None: ...
class MetricsSink(Protocol):
    def observe(self, metric: str, value: float, tags: dict[str, str] | None = None) -> None: ...
class PolicyStore(Protocol):
    def current_error_budget_state(self) -> str: ...
class RuntimePolicy:
    def __init__(
        self,
        *,
        min_groundedness: float,
        min_citation_precision: float,
        max_low_confidence_spans: int,
    ) -> None:
        self.min_groundedness = min_groundedness
        self.min_citation_precision = min_citation_precision
        self.max_low_confidence_spans = max_low_confidence_spans
    def should_block(self, report: GuardrailReport, error_budget_state: str) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if report.empty_context:
            reasons.append("empty_context")
        if report.groundedness < self.min_groundedness:
            reasons.append("groundedness_below_policy")
        if report.citation_precision < self.min_citation_precision:
            reasons.append("citation_precision_below_policy")
        if report.low_confidence_spans > self.max_low_confidence_spans:
            reasons.append("too_many_low_confidence_spans")
        if error_budget_state == "exhausted":
            reasons.append("error_budget_exhausted")
        return (len(reasons) > 0, reasons)
def now_ms() -> int:
    return int(time.time() * 1000)
def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
def normalize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    return compact
def normalize_query(text: str) -> str:
    return normalize_text(text).lower()
def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.split()) * 1.35))
def clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)
def percentile(values: Sequence[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = int(clamp(round((len(ordered) - 1) * p), 0, len(ordered) - 1))
    return ordered[index]
def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = list(values)
    if not vals:
        return default
    return float(mean(vals))
def split_by_sentence(text: str) -> list[str]:
    rough = re.split(r"(?<=[.!?])\s+", text.strip())
    return [seg.strip() for seg in rough if seg.strip()]
def iter_chunk_windows(text: str, *, chunk_chars: int, overlap_chars: int) -> Iterable[str]:
    cursor = 0
    n = len(text)
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")
    while cursor < n:
        end = min(cursor + chunk_chars, n)
        yield text[cursor:end]
        if end >= n:
            break
        cursor = max(0, end - overlap_chars)
def build_chunks_for_document(
    doc: SourceDocument,
    *,
    chunk_chars: int = 1400,
    overlap_chars: int = 220,
) -> list[ChunkRecord]:
    normalized = normalize_text(doc.text)
    chunks: list[ChunkRecord] = []
    for idx, window in enumerate(
        iter_chunk_windows(normalized, chunk_chars=chunk_chars, overlap_chars=overlap_chars),
        start=1,
    ):
        section = doc.metadata.get("section", "unknown")
        page = int(doc.metadata.get("page_start", 1)) + (idx - 1)
        checksum = stable_hash(f"{doc.source_id}:{doc.revision}:{idx}:{window[:240]}")
        chunks.append(
            ChunkRecord(
                chunk_id=f"{doc.source_id}-c{idx:04d}",
                source_id=doc.source_id,
                source_title=doc.title,
                revision=doc.revision,
                page=page,
                section=str(section),
                text=window,
                token_count=estimate_tokens(window),
                checksum=checksum,
            )
        )
    return chunks
def dedupe_chunks(chunks: Sequence[ChunkRecord]) -> list[ChunkRecord]:
    seen: set[str] = set()
    output: list[ChunkRecord] = []
    for chunk in chunks:
        key = f"{chunk.source_id}:{chunk.revision}:{chunk.checksum}"
        if key in seen:
            continue
        seen.add(key)
        output.append(chunk)
    return output
def ingest_documents(
    *,
    docs: Sequence[SourceDocument],
    vector_index: VectorIndex,
    lexical_index: LexicalIndex,
    trace_repo: TraceRepository,
    metrics: MetricsSink,
) -> dict[str, Any]:
    start = time.perf_counter()
    staged: list[ChunkRecord] = []
    for doc in docs:
        doc_start = time.perf_counter()
        chunks = build_chunks_for_document(doc)
        deduped = dedupe_chunks(chunks)
        staged.extend(deduped)
        metrics.observe("ingest.doc.chunks", float(len(deduped)), {"source_id": doc.source_id})
        trace_repo.write(
            "ingest.document.completed",
            {
                "source_id": doc.source_id,
                "title": doc.title,
                "revision": doc.revision,
                "chunks": len(deduped),
                "elapsed_ms": int((time.perf_counter() - doc_start) * 1000),
            },
        )
    unique_chunks = dedupe_chunks(staged)
    vector_index.upsert(unique_chunks)
    lexical_index.upsert(unique_chunks)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    metrics.observe("ingest.batch.elapsed_ms", float(elapsed_ms))
    metrics.observe("ingest.batch.chunk_count", float(len(unique_chunks)))
    trace_repo.write(
        "ingest.batch.completed",
        {
            "documents": len(docs),
            "chunks": len(unique_chunks),
            "elapsed_ms": elapsed_ms,
        },
    )
    return {
        "documents": len(docs),
        "chunks": len(unique_chunks),
        "elapsed_ms": elapsed_ms,
    }
def dedupe_retrieval_candidates(candidates: Sequence[RetrievalCandidate]) -> list[RetrievalCandidate]:
    seen: set[str] = set()
    out: list[RetrievalCandidate] = []
    for candidate in candidates:
        signature = stable_hash(
            f"{candidate.source_id}:{candidate.revision}:{candidate.page}:{candidate.text[:180]}"
        )
        if signature in seen:
            continue
        seen.add(signature)
        out.append(candidate)
    return out
def fuse_dense_sparse(
    dense_hits: Sequence[RetrievalCandidate],
    sparse_hits: Sequence[RetrievalCandidate],
    *,
    dense_weight: float = 0.62,
    sparse_weight: float = 0.38,
) -> list[RetrievalCandidate]:
    by_chunk: dict[str, RetrievalCandidate] = {}
    for hit in dense_hits:
        by_chunk[hit.chunk_id] = RetrievalCandidate(
            chunk_id=hit.chunk_id,
            source_id=hit.source_id,
            source_title=hit.source_title,
            revision=hit.revision,
            page=hit.page,
            section=hit.section,
            text=hit.text,
            dense_score=hit.dense_score,
            sparse_score=0.0,
        )
    for hit in sparse_hits:
        existing = by_chunk.get(hit.chunk_id)
        if existing is None:
            by_chunk[hit.chunk_id] = RetrievalCandidate(
                chunk_id=hit.chunk_id,
                source_id=hit.source_id,
                source_title=hit.source_title,
                revision=hit.revision,
                page=hit.page,
                section=hit.section,
                text=hit.text,
                dense_score=0.0,
                sparse_score=hit.sparse_score,
            )
        else:
            existing.sparse_score = max(existing.sparse_score, hit.sparse_score)
    fused: list[RetrievalCandidate] = []
    for item in by_chunk.values():
        item.rerank_score = item.dense_score * dense_weight + item.sparse_score * sparse_weight
        fused.append(item)
    fused.sort(key=lambda item: item.rerank_score, reverse=True)
    return fused
def rerank(
    *,
    query: str,
    candidates: Sequence[RetrievalCandidate],
    reranker: Reranker,
) -> list[RetrievalCandidate]:
    if not candidates:
        return []
    scores = reranker.score(query, candidates)
    scored: list[RetrievalCandidate] = []
    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = score
        scored.append(candidate)
    scored.sort(key=lambda item: item.rerank_score, reverse=True)
    return scored
def context_budget_planner(
    hits: Sequence[RetrievalCandidate],
    *,
    max_tokens: int,
    reserve_tokens_for_prompt: int,
    reserve_tokens_for_answer: int,
) -> str:
    budget = max(256, max_tokens - reserve_tokens_for_prompt - reserve_tokens_for_answer)
    used = 0
    blocks: list[str] = []
    for hit in hits:
        header = f"[chunk:{hit.chunk_id}] {hit.source_title} (rev:{hit.revision}) p.{hit.page} §{hit.section}\n"
        body = hit.text.strip()
        block = f"{header}{body}\n\n"
        estimate = estimate_tokens(block)
        if used + estimate > budget:
            break
        blocks.append(block)
        used += estimate
    return "".join(blocks).strip()
def estimate_retrieval_recall(hits: Sequence[RetrievalCandidate]) -> float:
    if not hits:
        return 0.0
    high_signal = [hit for hit in hits if hit.rerank_score >= 0.72]
    return clamp(len(high_signal) / max(1, len(hits)), 0.0, 1.0)
def retrieve_context(
    *,
    query: str,
    embedder: EmbeddingModel,
    vector_index: VectorIndex,
    lexical_index: LexicalIndex,
    reranker: Reranker,
    top_k_dense: int = 16,
    top_k_sparse: int = 16,
    top_k_final: int = 10,
) -> RetrievalBundle:
    t0 = time.perf_counter()
    q = normalize_query(query)
    embedding = embedder.encode_query(q)
    dense_hits = vector_index.search(embedding, top_k=top_k_dense)
    sparse_hits = lexical_index.search(q, top_k=top_k_sparse)
    dense_hits = dedupe_retrieval_candidates(dense_hits)
    sparse_hits = dedupe_retrieval_candidates(sparse_hits)
    fused = fuse_dense_sparse(dense_hits, sparse_hits)
    reranked = rerank(query=q, candidates=fused[: max(top_k_final * 2, 12)], reranker=reranker)
    final_hits = reranked[:top_k_final]
    context = context_budget_planner(
        final_hits,
        max_tokens=7800,
        reserve_tokens_for_prompt=850,
        reserve_tokens_for_answer=750,
    )
    retrieval_recall = estimate_retrieval_recall(final_hits)
    _ = int((time.perf_counter() - t0) * 1000)
    return RetrievalBundle(
        query=q,
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        fused_hits=final_hits,
        context=context,
        retrieval_recall=retrieval_recall,
    )
def build_prompt_contract(query: str) -> str:
    return (
        "You are a grounded assistant. Use only provided context. "
        "For every factual claim include citation tokens [S#]. "
        "If evidence is missing, state uncertainty explicitly. "
        f"UserQuery: {query}"
    )
def extract_citation_tokens(text: str) -> list[str]:
    return re.findall(r"\[S(\d+)\]", text)
def align_citations(answer_text: str, hits: Sequence[RetrievalCandidate]) -> list[CitationEvidence]:
    tokens = extract_citation_tokens(answer_text)
    if not tokens:
        return []
    evidences: list[CitationEvidence] = []
    for raw in tokens:
        idx = int(raw) - 1
        if idx < 0 or idx >= len(hits):
            continue
        hit = hits[idx]
        confidence = clamp(hit.rerank_score, 0.0, 1.0)
        evidences.append(
            CitationEvidence(
                token=f"[S{raw}]",
                chunk_id=hit.chunk_id,
                source_id=hit.source_id,
                source_title=hit.source_title,
                revision=hit.revision,
                page=hit.page,
                section=hit.section,
                confidence=round(confidence, 4),
            )
        )
    return evidences
def build_sources(citations: Sequence[CitationEvidence]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for item in citations:
        key = f"{item.source_id}:{item.revision}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source_id": item.source_id,
                "title": item.source_title,
                "revision": item.revision,
                "highest_confidence": item.confidence,
            }
        )
    sources.sort(key=lambda src: src["highest_confidence"], reverse=True)
    return sources
def count_low_confidence_spans(citations: Sequence[CitationEvidence], threshold: float = 0.65) -> int:
    return sum(1 for item in citations if item.confidence < threshold)
def estimate_citation_precision(citations: Sequence[CitationEvidence]) -> float:
    if not citations:
        return 0.0
    return safe_mean(item.confidence for item in citations)
def estimate_citation_coverage(answer: str, citations: Sequence[CitationEvidence]) -> float:
    sentences = split_by_sentence(answer)
    if not sentences:
        return 0.0
    citation_count = max(1, len(citations))
    coverage = citation_count / len(sentences)
    return clamp(coverage, 0.0, 1.0)
def estimate_groundedness(
    *,
    answer: str,
    retrieval_hits: Sequence[RetrievalCandidate],
    citations: Sequence[CitationEvidence],
) -> float:
    if not answer.strip() or not retrieval_hits:
        return 0.0
    lexical_overlap_components: list[float] = []
    answer_terms = set(re.findall(r"[a-zA-Z]{4,}", answer.lower()))
    for hit in retrieval_hits:
        hit_terms = set(re.findall(r"[a-zA-Z]{4,}", hit.text.lower()))
        overlap = len(answer_terms & hit_terms)
        denom = max(1, len(answer_terms))
        lexical_overlap_components.append(overlap / denom)
    overlap_score = clamp(max(lexical_overlap_components, default=0.0), 0.0, 1.0)
    citation_score = estimate_citation_precision(citations)
    return round(clamp(0.58 * overlap_score + 0.42 * citation_score, 0.0, 1.0), 4)
def build_guardrail_report(
    *,
    answer: str,
    retrieval: RetrievalBundle,
    citations: Sequence[CitationEvidence],
    policy: RuntimePolicy,
    policy_store: PolicyStore,
) -> GuardrailReport:
    citation_precision = estimate_citation_precision(citations)
    citation_coverage = estimate_citation_coverage(answer, citations)
    groundedness = estimate_groundedness(
        answer=answer,
        retrieval_hits=retrieval.fused_hits,
        citations=citations,
    )
    low_confidence_spans = count_low_confidence_spans(citations)
    empty_context = len(retrieval.context.strip()) == 0
    report = GuardrailReport(
        groundedness=groundedness,
        citation_precision=round(citation_precision, 4),
        citation_coverage=round(citation_coverage, 4),
        low_confidence_spans=low_confidence_spans,
        empty_context=empty_context,
        blocked=False,
        reasons=[],
    )
    blocked, reasons = policy.should_block(report, policy_store.current_error_budget_state())
    report.blocked = blocked
    report.reasons = reasons
    return report
def fallback_payload(*, env: QueryEnvelope, reason: str, metrics: RuntimeMetrics) -> AnswerPayload:
    return AnswerPayload(
        request_id=env.request_id,
        answer=(
            "I cannot provide a fully grounded answer from the indexed documents right now. "
            "Please refine the question or provide a more specific source scope."
        ),
        citations=[],
        sources=[],
        session_id=env.session_id,
        confidence=0.0,
        metrics=metrics,
        guardrails=GuardrailReport(
            groundedness=0.0,
            citation_precision=0.0,
            citation_coverage=0.0,
            low_confidence_spans=0,
            empty_context=True,
            blocked=True,
            reasons=[reason],
        ),
    )
def answer_with_policy(
    *,
    env: QueryEnvelope,
    embedder: EmbeddingModel,
    vector_index: VectorIndex,
    lexical_index: LexicalIndex,
    reranker: Reranker,
    llm: LLMGateway,
    sessions: SessionRepository,
    traces: TraceRepository,
    metrics: MetricsSink,
    policy_store: PolicyStore,
    policy: RuntimePolicy,
    recent_latency_window: Sequence[int] | None = None,
) -> AnswerPayload:
    start_total = time.perf_counter()
    retrieval_start = time.perf_counter()
    retrieval = retrieve_context(
        query=env.query,
        embedder=embedder,
        vector_index=vector_index,
        lexical_index=lexical_index,
        reranker=reranker,
    )
    retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)
    if not retrieval.context:
        metrics_window = list(recent_latency_window or [])
        total_ms = int((time.perf_counter() - start_total) * 1000)
        metrics_window.append(total_ms)
        runtime_metrics = RuntimeMetrics(
            latency_ms=total_ms,
            retrieval_ms=retrieval_ms,
            rerank_ms=0,
            generation_ms=0,
            citation_alignment_ms=0,
            p95_latency_ms=percentile(metrics_window, 0.95),
        )
        traces.write(
            "runtime.query.blocked",
            {
                "request_id": env.request_id,
                "reason": "empty_context",
                "query": env.query,
                "retrieval_recall": retrieval.retrieval_recall,
            },
        )
        return fallback_payload(env=env, reason="empty_context", metrics=runtime_metrics)
    history = sessions.get_recent_turns(session_id=env.session_id, max_turns=8)
    generation_start = time.perf_counter()
    draft = llm.generate(
        prompt=build_prompt_contract(retrieval.query),
        context=retrieval.context,
        history=history,
    )
    generation_ms = int((time.perf_counter() - generation_start) * 1000)
    citation_start = time.perf_counter()
    citations = align_citations(draft.answer, retrieval.fused_hits)
    citation_alignment_ms = int((time.perf_counter() - citation_start) * 1000)
    guardrails = build_guardrail_report(
        answer=draft.answer,
        retrieval=retrieval,
        citations=citations,
        policy=policy,
        policy_store=policy_store,
    )
    total_ms = int((time.perf_counter() - start_total) * 1000)
    latency_window = list(recent_latency_window or [])
    latency_window.append(total_ms)
    runtime_metrics = RuntimeMetrics(
        latency_ms=total_ms,
        retrieval_ms=retrieval_ms,
        rerank_ms=0,
        generation_ms=generation_ms,
        citation_alignment_ms=citation_alignment_ms,
        p95_latency_ms=percentile(latency_window, 0.95),
    )
    if guardrails.blocked:
        traces.write(
            "runtime.query.blocked",
            {
                "request_id": env.request_id,
                "reasons": guardrails.reasons,
                "metrics": {
                    "latency_ms": runtime_metrics.latency_ms,
                    "groundedness": guardrails.groundedness,
                    "citation_precision": guardrails.citation_precision,
                },
            },
        )
        return fallback_payload(env=env, reason="guardrail_block", metrics=runtime_metrics)
    sources = build_sources(citations)
    confidence = round(
        clamp(
            0.55 * guardrails.groundedness
            + 0.35 * guardrails.citation_precision
            + 0.10 * retrieval.retrieval_recall,
            0.0,
            1.0,
        ),
        4,
    )
    payload = AnswerPayload(
        request_id=env.request_id,
        answer=draft.answer,
        citations=citations,
        sources=sources,
        session_id=env.session_id,
        confidence=confidence,
        metrics=runtime_metrics,
        guardrails=guardrails,
    )
    sessions.append_turn(session_id=env.session_id, user_text=retrieval.query, assistant_text=draft.answer)
    traces.write(
        "runtime.query.completed",
        {
            "request_id": env.request_id,
            "session_id": env.session_id,
            "model": draft.model_name,
            "prompt_version": draft.prompt_version,
            "token_usage": draft.token_usage,
            "metrics": {
                "latency_ms": runtime_metrics.latency_ms,
                "retrieval_ms": runtime_metrics.retrieval_ms,
                "generation_ms": runtime_metrics.generation_ms,
                "citation_alignment_ms": runtime_metrics.citation_alignment_ms,
                "p95_latency_ms": runtime_metrics.p95_latency_ms,
            },
            "quality": {
                "confidence": payload.confidence,
                "groundedness": guardrails.groundedness,
                "citation_precision": guardrails.citation_precision,
                "citation_coverage": guardrails.citation_coverage,
                "retrieval_recall": retrieval.retrieval_recall,
            },
        },
    )
    metrics.observe("runtime.latency_ms", float(runtime_metrics.latency_ms))
    metrics.observe("runtime.groundedness", float(guardrails.groundedness))
    metrics.observe("runtime.citation_precision", float(guardrails.citation_precision))
    return payload
def score_eval_case(payload: AnswerPayload, case: EvalCase) -> EvalResult:
    cited_sources = {e.source_id for e in payload.citations}
    citation_hit = len(cited_sources & case.required_source_ids) / max(1, len(case.required_source_ids))
    ref_terms = set(re.findall(r"[a-zA-Z]{4,}", case.reference_answer.lower()))
    ans_terms = set(re.findall(r"[a-zA-Z]{4,}", payload.answer.lower()))
    overlap = len(ref_terms & ans_terms) / max(1, len(ref_terms))
    answer_relevance = round(clamp(overlap, 0.0, 1.0), 4)
    groundedness = payload.guardrails.groundedness
    citation_precision = payload.guardrails.citation_precision
    passed = (
        groundedness >= 0.86
        and citation_precision >= 0.9
        and answer_relevance >= 0.72
        and citation_hit >= 0.75
    )
    return EvalResult(
        case_id=case.case_id,
        groundedness=groundedness,
        citation_precision=citation_precision,
        answer_relevance=answer_relevance,
        passed=passed,
    )
def run_eval_suite(
    *,
    suite_name: str,
    cases: Sequence[EvalCase],
    request_runner: callable,
    traces: TraceRepository,
) -> dict[str, Any]:
    started = now_ms()
    results: list[EvalResult] = []
    latencies: list[int] = []
    for case in cases:
        env = QueryEnvelope(
            request_id=f"eval-{suite_name}-{uuid.uuid4().hex[:10]}",
            session_id=f"eval-session-{case.case_id}",
            actor_id="eval-bot",
            channel=Channel.INTERNAL,
            query=case.query,
            locale="en-US",
            created_at_ms=now_ms(),
        )
        payload: AnswerPayload = request_runner(env)
        results.append(score_eval_case(payload, case))
        latencies.append(payload.metrics.latency_ms)
    pass_rate = safe_mean(1.0 if r.passed else 0.0 for r in results)
    avg_groundedness = safe_mean(r.groundedness for r in results)
    avg_citation_precision = safe_mean(r.citation_precision for r in results)
    avg_answer_relevance = safe_mean(r.answer_relevance for r in results)
    summary = {
        "suite_name": suite_name,
        "case_count": len(results),
        "pass_rate": round(pass_rate, 4),
        "avg_groundedness": round(avg_groundedness, 4),
        "avg_citation_precision": round(avg_citation_precision, 4),
        "avg_answer_relevance": round(avg_answer_relevance, 4),
        "p95_latency_ms": percentile(latencies, 0.95),
        "elapsed_ms": now_ms() - started,
    }
    traces.write("eval.suite.completed", summary)
    return summary
def release_gate_decision(
    *,
    current_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    max_allowed_regression: float = -0.02,
) -> ReleaseDecision:
    reasons: list[str] = []
    current_quality = (
        0.40 * current_summary.get("avg_groundedness", 0.0)
        + 0.35 * current_summary.get("avg_citation_precision", 0.0)
        + 0.25 * current_summary.get("avg_answer_relevance", 0.0)
    )
    baseline_quality = (
        0.40 * baseline_summary.get("avg_groundedness", 0.0)
        + 0.35 * baseline_summary.get("avg_citation_precision", 0.0)
        + 0.25 * baseline_summary.get("avg_answer_relevance", 0.0)
    )
    delta = round(current_quality - baseline_quality, 4)
    if delta < max_allowed_regression:
        reasons.append("quality_regression_exceeds_policy")
    if current_summary.get("pass_rate", 0.0) < 0.9:
        reasons.append("pass_rate_below_policy")
    if current_summary.get("p95_latency_ms", 999999) > 1900:
        reasons.append("latency_budget_exceeded")
    return ReleaseDecision(
        allow_release=len(reasons) == 0,
        regression_delta=delta,
        reasons=reasons,
    )
def build_runtime_env(*, session_id: str, actor_id: str, query: str, channel: Channel = Channel.API) -> QueryEnvelope:
    return QueryEnvelope(
        request_id=f"req-{uuid.uuid4().hex[:12]}",
        session_id=session_id,
        actor_id=actor_id,
        channel=channel,
        query=query,
        locale="en-US",
        created_at_ms=now_ms(),
    )
# Orchestration entry used by app layer.
def handle_query(
    *,
    query: str,
    session_id: str,
    actor_id: str,
    embedder: EmbeddingModel,
    vector_index: VectorIndex,
    lexical_index: LexicalIndex,
    reranker: Reranker,
    llm: LLMGateway,
    sessions: SessionRepository,
    traces: TraceRepository,
    metrics: MetricsSink,
    policy_store: PolicyStore,
    recent_latency_window: Sequence[int] | None = None,
) -> AnswerPayload:
    env = build_runtime_env(session_id=session_id, actor_id=actor_id, query=query)
    policy = RuntimePolicy(
        min_groundedness=0.86,
        min_citation_precision=0.90,
        max_low_confidence_spans=2,
    )
    return answer_with_policy(
        env=env,
        embedder=embedder,
        vector_index=vector_index,
        lexical_index=lexical_index,
        reranker=reranker,
        llm=llm,
        sessions=sessions,
        traces=traces,
        metrics=metrics,
        policy_store=policy_store,
        policy=policy,
        recent_latency_window=recent_latency_window,
    )
