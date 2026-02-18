from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from statistics import fmean
from typing import Any, Iterable, Protocol, Sequence
import hashlib
import json
import re
import time

class SourceType(str, Enum):
    PDF = "pdf"
    URL = "url"
    MARKDOWN = "markdown"
    TEXT = "text"


class FetchStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"

class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

class RunState(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NO_CHANGES = "no_changes"
    FAILED = "failed"

class FailureKind(str, Enum):
    NONE = "none"
    FETCH_FAILED = "fetch_failed"
    NORMALIZATION_FAILED = "normalization_failed"
    INDEX_FAILED = "index_failed"
    POLICY_BLOCKED = "policy_blocked"

@dataclass(slots=True)
class SourceInput:
    source_id: str
    source_type: SourceType
    uri: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FetchedSource:
    source: SourceInput
    status: FetchStatus
    raw_text: str
    fetched_at_ms: int
    error: str | None = None


@dataclass(slots=True)
class NormalizedSource:
    source_id: str
    source_type: SourceType
    uri: str
    text: str
    checksum: str
    token_count: int
    language: str
    parser_notes: list[str]
    metadata: dict[str, Any]


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    version_id: str
    position: int
    text: str
    token_count: int
    checksum: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class VersionSnapshot:
    version_id: str
    pipeline_id: str
    parent_version_id: str | None
    source_checksums: dict[str, str]
    chunk_count: int
    created_at_ms: int
    label: str


@dataclass(slots=True)
class DeltaPlan:
    changed_sources: list[NormalizedSource]
    unchanged_source_ids: list[str]
    removed_source_ids: list[str]
    changed_ratio: float
    reason: str


@dataclass(slots=True)
class IndexTask:
    task_id: str
    pipeline_id: str
    version_id: str
    chunks: list[ChunkRecord]
    attempt: int
    max_attempts: int
    state: TaskState
    created_at_ms: int
    last_error: str | None = None


@dataclass(slots=True)
class TaskResult:
    task_id: str
    state: TaskState
    indexed_count: int
    latency_ms: int
    attempt: int
    error: str | None = None


@dataclass(slots=True)
class IngestionPolicy:
    chunk_size_chars: int = 900
    overlap_chars: int = 120
    max_chunks_per_source: int = 220
    max_chunks_per_task: int = 320
    min_text_chars: int = 40
    reindex_cron: str = "0 */6 * * *"
    retry_base_ms: int = 200
    max_retries: int = 3
    strip_html: bool = True
    remove_boilerplate: bool = True
    redact_pii: bool = True
    default_language: str = "en"


@dataclass(slots=True)
class IngestionEnv:
    run_id: str
    pipeline_id: str
    sources: list[SourceInput]
    trigger: str
    actor: str
    policy: IngestionPolicy
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunMetrics:
    fetch_ms: int
    normalize_ms: int
    planning_ms: int
    chunk_ms: int
    queue_ms: int
    index_ms: int
    scheduler_ms: int
    total_ms: int


@dataclass(slots=True)
class IngestionRun:
    run_id: str
    pipeline_id: str
    state: RunState
    version_id: str | None
    source_count: int
    fetched_count: int
    normalized_count: int
    changed_count: int
    chunk_count: int
    indexed_count: int
    queued_tasks: int
    completed_tasks: int
    failed_tasks: int
    failure_kind: FailureKind
    warnings: list[str]
    reason: str
    metrics: RunMetrics


@dataclass(slots=True)
class FreshnessEntry:
    pipeline_id: str
    version_id: str
    changed_sources: int
    source_count: int
    recorded_at_ms: int


class ConnectorHub(Protocol):
    def fetch(self, source: SourceInput) -> FetchedSource: ...


class VersionStore(Protocol):
    def latest(self, pipeline_id: str) -> VersionSnapshot | None: ...

    def put(self, version: VersionSnapshot) -> None: ...


class QueueStore(Protocol):
    def enqueue(self, task: IndexTask) -> None: ...

    def dequeue(self, max_tasks: int) -> list[IndexTask]: ...

    def ack(self, task_id: str) -> None: ...

    def retry(self, task: IndexTask, delay_ms: int) -> None: ...


class Indexer(Protocol):
    def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> int: ...


class Scheduler(Protocol):
    def ensure_job(self, pipeline_id: str, cron_expr: str) -> None: ...


class FreshnessStore(Protocol):
    def record(self, entry: FreshnessEntry) -> None: ...


class MetricsSink(Protocol):
    def observe(self, metric: str, value: float, tags: dict[str, str] | None = None) -> None: ...


class TraceWriter(Protocol):
    def write(self, event_type: str, payload: dict[str, Any]) -> None: ...


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


def normalize_whitespace(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    return compact


def strip_html(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(text)


def remove_common_boilerplate(text: str) -> str:
    patterns = [
        r"cookie policy",
        r"all rights reserved",
        r"privacy policy",
        r"terms of service",
        r"subscribe to newsletter",
        r"click here to accept",
    ]

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    filtered: list[str] = []

    for line in lines:
        lower = line.lower()
        if any(pattern in lower for pattern in patterns):
            continue
        filtered.append(line)

    return "\n".join(filtered)


def redact_pii(text: str) -> tuple[str, int]:
    redactions = 0
    out = text

    for email in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", out):
        out = out.replace(email, "[redacted_email]")
        redactions += 1

    for card in re.findall(r"\b(?:\d[ -]*?){13,16}\b", out):
        out = out.replace(card, "[redacted_card]")
        redactions += 1

    return out, redactions


def detect_language(text: str, fallback: str) -> str:
    if not text:
        return fallback

    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    ratio = safe_div(ascii_chars, len(text))

    if ratio > 0.94:
        return "en"
    return fallback


def estimate_tokens(text: str) -> int:
    words = re.findall(r"[a-zA-Z0-9_\-]{2,}", text)
    return max(1, int(len(words) * 1.35))


def checksum_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_checksum(source: SourceInput, text: str) -> str:
    seed = f"{source.source_type.value}|{source.uri}|{text[:600]}|{len(text)}"
    return checksum_text(seed)


def merge_metadata(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    merged = dict(a)
    merged.update(b)
    return merged


def normalize_fetched_source(fetched: FetchedSource, policy: IngestionPolicy) -> NormalizedSource:
    notes: list[str] = []
    text = fetched.raw_text

    if policy.strip_html and fetched.source.source_type in {SourceType.URL, SourceType.MARKDOWN}:
        text = strip_html(text)
        notes.append("html_stripped")

    if policy.remove_boilerplate:
        text = remove_common_boilerplate(text)
        notes.append("boilerplate_removed")

    text = normalize_whitespace(text)

    if policy.redact_pii:
        text, redaction_count = redact_pii(text)
        if redaction_count > 0:
            notes.append(f"pii_redacted:{redaction_count}")

    language = detect_language(text, fallback=policy.default_language)

    return NormalizedSource(
        source_id=fetched.source.source_id,
        source_type=fetched.source.source_type,
        uri=fetched.source.uri,
        text=text,
        checksum=source_checksum(fetched.source, text),
        token_count=estimate_tokens(text),
        language=language,
        parser_notes=notes,
        metadata=merge_metadata(fetched.source.metadata, {"fetched_at_ms": fetched.fetched_at_ms}),
    )


def build_chunk_windows(text: str, size: int, overlap: int) -> Iterable[str]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap >= size:
        raise ValueError("chunk overlap must be less than chunk size")

    cursor = 0
    n = len(text)

    while cursor < n:
        end = min(cursor + size, n)
        yield text[cursor:end]
        if end >= n:
            break
        cursor = max(0, end - overlap)


def chunk_source(source: NormalizedSource, version_id: str, policy: IngestionPolicy) -> tuple[list[ChunkRecord], list[str]]:
    warnings: list[str] = []

    if len(source.text) < policy.min_text_chars:
        warnings.append(f"source_too_short:{source.source_id}")
        return ([], warnings)

    records: list[ChunkRecord] = []

    for idx, window in enumerate(
        build_chunk_windows(source.text, policy.chunk_size_chars, policy.overlap_chars),
        start=1,
    ):
        if idx > policy.max_chunks_per_source:
            warnings.append(f"source_chunk_limit_reached:{source.source_id}")
            break

        chunk_checksum = checksum_text(f"{source.source_id}|{idx}|{window[:320]}")
        chunk_id = hashlib.sha1(f"{source.source_id}:{version_id}:{idx}".encode("utf-8")).hexdigest()[:16]

        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                source_id=source.source_id,
                version_id=version_id,
                position=idx,
                text=window,
                token_count=estimate_tokens(window),
                checksum=chunk_checksum,
                metadata={
                    "uri": source.uri,
                    "source_type": source.source_type.value,
                    "language": source.language,
                },
            )
        )

    return (records, warnings)


def dedupe_chunks(chunks: Sequence[ChunkRecord]) -> list[ChunkRecord]:
    seen: set[str] = set()
    out: list[ChunkRecord] = []

    for chunk in chunks:
        signature = f"{chunk.source_id}:{chunk.checksum}"
        if signature in seen:
            continue
        seen.add(signature)
        out.append(chunk)

    return out


def chunk_batches(chunks: Sequence[ChunkRecord], max_per_batch: int) -> list[list[ChunkRecord]]:
    if max_per_batch <= 0:
        raise ValueError("max_per_batch must be positive")

    batches: list[list[ChunkRecord]] = []

    for idx in range(0, len(chunks), max_per_batch):
        batches.append(list(chunks[idx : idx + max_per_batch]))

    return batches


def build_version_label(env: IngestionEnv, changed_count: int) -> str:
    return f"{env.pipeline_id}:{env.trigger}:{changed_count}:{env.run_id[:10]}"


def full_checksum_map(sources: Sequence[NormalizedSource]) -> dict[str, str]:
    return {item.source_id: item.checksum for item in sources}


def plan_delta(current_sources: Sequence[NormalizedSource], latest: VersionSnapshot | None) -> DeltaPlan:
    current_map = full_checksum_map(current_sources)

    if latest is None:
        return DeltaPlan(
            changed_sources=list(current_sources),
            unchanged_source_ids=[],
            removed_source_ids=[],
            changed_ratio=1.0 if current_sources else 0.0,
            reason="initial_version",
        )

    changed: list[NormalizedSource] = []
    unchanged: list[str] = []

    for item in current_sources:
        prev = latest.source_checksums.get(item.source_id)
        if prev is None or prev != item.checksum:
            changed.append(item)
        else:
            unchanged.append(item.source_id)

    removed = [sid for sid in latest.source_checksums.keys() if sid not in current_map]

    ratio = safe_div(len(changed), len(current_sources)) if current_sources else 0.0
    reason = "delta_detected" if changed or removed else "no_changes"

    return DeltaPlan(
        changed_sources=changed,
        unchanged_source_ids=unchanged,
        removed_source_ids=removed,
        changed_ratio=round(ratio, 4),
        reason=reason,
    )


def build_next_checksums(plan: DeltaPlan, current_sources: Sequence[NormalizedSource]) -> dict[str, str]:
    current_map = full_checksum_map(current_sources)

    # Removed ids are naturally excluded from current_map.
    for removed_id in plan.removed_source_ids:
        current_map.pop(removed_id, None)

    return current_map


def create_version_snapshot(
    env: IngestionEnv,
    latest: VersionSnapshot | None,
    checksums: dict[str, str],
    chunk_count: int,
) -> VersionSnapshot:
    material = json.dumps(
        {
            "pipeline_id": env.pipeline_id,
            "run_id": env.run_id,
            "checksums": checksums,
            "chunk_count": chunk_count,
        },
        sort_keys=True,
    )
    version_id = checksum_text(material)[:22]

    return VersionSnapshot(
        version_id=version_id,
        pipeline_id=env.pipeline_id,
        parent_version_id=latest.version_id if latest else None,
        source_checksums=checksums,
        chunk_count=chunk_count,
        created_at_ms=now_ms(),
        label=build_version_label(env, changed_count=chunk_count),
    )


def make_index_tasks(env: IngestionEnv, version_id: str, chunks: Sequence[ChunkRecord]) -> list[IndexTask]:
    tasks: list[IndexTask] = []
    batches = chunk_batches(chunks, env.policy.max_chunks_per_task)

    for idx, batch in enumerate(batches, start=1):
        task_id = hashlib.sha1(f"{env.run_id}:{version_id}:{idx}:{len(batch)}".encode()).hexdigest()[:18]
        tasks.append(
            IndexTask(
                task_id=task_id,
                pipeline_id=env.pipeline_id,
                version_id=version_id,
                chunks=batch,
                attempt=0,
                max_attempts=env.policy.max_retries,
                state=TaskState.PENDING,
                created_at_ms=now_ms(),
            )
        )

    return tasks


def backoff_ms(base_ms: int, attempt: int) -> int:
    return int(base_ms * (2 ** max(0, attempt - 1)))


def process_task(task: IndexTask, indexer: Indexer) -> TaskResult:
    started = time.perf_counter()

    try:
        indexed = indexer.upsert_chunks(task.chunks)
        return TaskResult(
            task_id=task.task_id,
            state=TaskState.DONE,
            indexed_count=indexed,
            latency_ms=int((time.perf_counter() - started) * 1000),
            attempt=task.attempt + 1,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return TaskResult(
            task_id=task.task_id,
            state=TaskState.FAILED,
            indexed_count=0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            attempt=task.attempt + 1,
            error=str(exc),
        )


def run_queue_worker(
    *,
    queue: QueueStore,
    indexer: Indexer,
    policy: IngestionPolicy,
    traces: TraceWriter,
    metrics: MetricsSink,
    max_loops: int = 80,
) -> list[TaskResult]:
    results: list[TaskResult] = []

    for _ in range(max_loops):
        tasks = queue.dequeue(max_tasks=12)
        if not tasks:
            break

        for task in tasks:
            result = process_task(task, indexer)
            results.append(result)

            traces.write(
                "ingestion.task.processed",
                {
                    "task_id": task.task_id,
                    "pipeline_id": task.pipeline_id,
                    "version_id": task.version_id,
                    "attempt": result.attempt,
                    "state": result.state.value,
                    "indexed_count": result.indexed_count,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                },
            )

            metrics.observe(
                "ingestion.task.latency_ms",
                float(result.latency_ms),
                {"state": result.state.value},
            )

            if result.state == TaskState.DONE:
                queue.ack(task.task_id)
                continue

            next_attempt = task.attempt + 1
            if next_attempt >= task.max_attempts:
                queue.ack(task.task_id)
                continue

            retry_task = IndexTask(
                task_id=task.task_id,
                pipeline_id=task.pipeline_id,
                version_id=task.version_id,
                chunks=task.chunks,
                attempt=next_attempt,
                max_attempts=task.max_attempts,
                state=TaskState.PENDING,
                created_at_ms=task.created_at_ms,
                last_error=result.error,
            )
            delay = backoff_ms(policy.retry_base_ms, next_attempt)
            queue.retry(retry_task, delay_ms=delay)

    return results


def fetch_sources(env: IngestionEnv, connectors: ConnectorHub, traces: TraceWriter) -> tuple[list[FetchedSource], list[str]]:
    fetched: list[FetchedSource] = []
    warnings: list[str] = []

    for source in env.sources:
        item = connectors.fetch(source)
        fetched.append(item)

        if item.status == FetchStatus.FAILED:
            warnings.append(f"fetch_failed:{source.source_id}")
            traces.write(
                "ingestion.source.fetch_failed",
                {
                    "run_id": env.run_id,
                    "source_id": source.source_id,
                    "uri": source.uri,
                    "error": item.error,
                },
            )

    return fetched, warnings


def normalize_sources(
    fetched: Sequence[FetchedSource],
    policy: IngestionPolicy,
    traces: TraceWriter,
) -> tuple[list[NormalizedSource], list[str]]:
    normalized: list[NormalizedSource] = []
    warnings: list[str] = []

    for item in fetched:
        if item.status != FetchStatus.OK:
            continue

        normalized_item = normalize_fetched_source(item, policy)

        if len(normalized_item.text) < policy.min_text_chars:
            warnings.append(f"source_under_min_chars:{normalized_item.source_id}")
            traces.write(
                "ingestion.source.skipped",
                {
                    "source_id": normalized_item.source_id,
                    "reason": "too_short",
                    "text_len": len(normalized_item.text),
                },
            )
            continue

        normalized.append(normalized_item)

    return normalized, warnings


def build_no_change_run(env: IngestionEnv, warnings: list[str], total_ms: int) -> IngestionRun:
    return IngestionRun(
        run_id=env.run_id,
        pipeline_id=env.pipeline_id,
        state=RunState.NO_CHANGES,
        version_id=None,
        source_count=len(env.sources),
        fetched_count=len(env.sources),
        normalized_count=0,
        changed_count=0,
        chunk_count=0,
        indexed_count=0,
        queued_tasks=0,
        completed_tasks=0,
        failed_tasks=0,
        failure_kind=FailureKind.NONE,
        warnings=warnings,
        reason="no_changed_sources",
        metrics=RunMetrics(
            fetch_ms=0,
            normalize_ms=0,
            planning_ms=0,
            chunk_ms=0,
            queue_ms=0,
            index_ms=0,
            scheduler_ms=0,
            total_ms=total_ms,
        ),
    )


def ingest_run(
    *,
    env: IngestionEnv,
    connectors: ConnectorHub,
    versions: VersionStore,
    queue: QueueStore,
    indexer: Indexer,
    scheduler: Scheduler,
    freshness: FreshnessStore,
    metrics: MetricsSink,
    traces: TraceWriter,
) -> IngestionRun:
    t_total = time.perf_counter()

    traces.write(
        "ingestion.run.started",
        {
            "run_id": env.run_id,
            "pipeline_id": env.pipeline_id,
            "source_count": len(env.sources),
            "trigger": env.trigger,
            "actor": env.actor,
        },
    )

    t_fetch = time.perf_counter()
    fetched, fetch_warnings = fetch_sources(env, connectors, traces)
    fetch_ms = int((time.perf_counter() - t_fetch) * 1000)

    t_norm = time.perf_counter()
    normalized, norm_warnings = normalize_sources(fetched, env.policy, traces)
    normalize_ms = int((time.perf_counter() - t_norm) * 1000)

    warnings: list[str] = [*fetch_warnings, *norm_warnings]

    t_plan = time.perf_counter()
    latest = versions.latest(env.pipeline_id)
    plan = plan_delta(normalized, latest)
    planning_ms = int((time.perf_counter() - t_plan) * 1000)

    if not plan.changed_sources and not plan.removed_source_ids:
        scheduler.ensure_job(env.pipeline_id, env.policy.reindex_cron)
        run = build_no_change_run(
            env,
            warnings,
            total_ms=int((time.perf_counter() - t_total) * 1000),
        )
        traces.write(
            "ingestion.run.completed",
            {
                "run_id": env.run_id,
                "state": run.state.value,
                "reason": run.reason,
                "warnings": warnings,
            },
        )
        return run

    t_chunk = time.perf_counter()
    changed_chunks: list[ChunkRecord] = []

    for source in plan.changed_sources:
        chunks, chunk_warnings = chunk_source(source, version_id="pending", policy=env.policy)
        changed_chunks.extend(chunks)
        warnings.extend(chunk_warnings)

    changed_chunks = dedupe_chunks(changed_chunks)
    chunk_ms = int((time.perf_counter() - t_chunk) * 1000)

    next_checksums = build_next_checksums(plan, normalized)
    version = create_version_snapshot(
        env,
        latest,
        checksums=next_checksums,
        chunk_count=len(changed_chunks),
    )

    rebound_chunks = [
        ChunkRecord(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            version_id=version.version_id,
            position=chunk.position,
            text=chunk.text,
            token_count=chunk.token_count,
            checksum=chunk.checksum,
            metadata=chunk.metadata,
        )
        for chunk in changed_chunks
    ]

    tasks = make_index_tasks(env, version.version_id, rebound_chunks)

    t_queue = time.perf_counter()
    for task in tasks:
        queue.enqueue(task)
    queue_ms = int((time.perf_counter() - t_queue) * 1000)

    t_index = time.perf_counter()
    outcomes = run_queue_worker(
        queue=queue,
        indexer=indexer,
        policy=env.policy,
        traces=traces,
        metrics=metrics,
    )
    index_ms = int((time.perf_counter() - t_index) * 1000)

    indexed_count = sum(item.indexed_count for item in outcomes if item.state == TaskState.DONE)
    completed_tasks = sum(1 for item in outcomes if item.state == TaskState.DONE)
    failed_tasks = sum(1 for item in outcomes if item.state == TaskState.FAILED)

    versions.put(version)

    t_sched = time.perf_counter()
    scheduler.ensure_job(env.pipeline_id, env.policy.reindex_cron)
    freshness.record(
        FreshnessEntry(
            pipeline_id=env.pipeline_id,
            version_id=version.version_id,
            changed_sources=len(plan.changed_sources),
            source_count=len(normalized),
            recorded_at_ms=now_ms(),
        )
    )
    scheduler_ms = int((time.perf_counter() - t_sched) * 1000)

    total_ms = int((time.perf_counter() - t_total) * 1000)

    if failed_tasks == 0:
        state = RunState.SUCCESS
        failure = FailureKind.NONE
        reason = "all_tasks_completed"
    elif completed_tasks > 0:
        state = RunState.PARTIAL
        failure = FailureKind.INDEX_FAILED
        reason = "some_tasks_failed"
    else:
        state = RunState.FAILED
        failure = FailureKind.INDEX_FAILED
        reason = "all_tasks_failed"

    run = IngestionRun(
        run_id=env.run_id,
        pipeline_id=env.pipeline_id,
        state=state,
        version_id=version.version_id,
        source_count=len(env.sources),
        fetched_count=len([item for item in fetched if item.status == FetchStatus.OK]),
        normalized_count=len(normalized),
        changed_count=len(plan.changed_sources),
        chunk_count=len(rebound_chunks),
        indexed_count=indexed_count,
        queued_tasks=len(tasks),
        completed_tasks=completed_tasks,
        failed_tasks=failed_tasks,
        failure_kind=failure,
        warnings=warnings,
        reason=reason,
        metrics=RunMetrics(
            fetch_ms=fetch_ms,
            normalize_ms=normalize_ms,
            planning_ms=planning_ms,
            chunk_ms=chunk_ms,
            queue_ms=queue_ms,
            index_ms=index_ms,
            scheduler_ms=scheduler_ms,
            total_ms=total_ms,
        ),
    )

    traces.write(
        "ingestion.run.completed",
        {
            "run_id": run.run_id,
            "pipeline_id": run.pipeline_id,
            "state": run.state.value,
            "version_id": run.version_id,
            "changed_count": run.changed_count,
            "chunk_count": run.chunk_count,
            "indexed_count": run.indexed_count,
            "queued_tasks": run.queued_tasks,
            "completed_tasks": run.completed_tasks,
            "failed_tasks": run.failed_tasks,
            "warnings": run.warnings,
            "reason": run.reason,
            "metrics": {
                "fetch_ms": run.metrics.fetch_ms,
                "normalize_ms": run.metrics.normalize_ms,
                "planning_ms": run.metrics.planning_ms,
                "chunk_ms": run.metrics.chunk_ms,
                "queue_ms": run.metrics.queue_ms,
                "index_ms": run.metrics.index_ms,
                "scheduler_ms": run.metrics.scheduler_ms,
                "total_ms": run.metrics.total_ms,
            },
        },
    )

    metrics.observe("ingestion.run.total_ms", float(run.metrics.total_ms), {"state": run.state.value})
    metrics.observe("ingestion.run.changed_sources", float(run.changed_count), None)
    metrics.observe("ingestion.run.chunk_count", float(run.chunk_count), None)

    return run


def summarize_runs(runs: Sequence[IngestionRun]) -> dict[str, Any]:
    if not runs:
        return {
            "run_count": 0,
            "success_rate": 0.0,
            "avg_total_ms": 0,
            "p95_total_ms": 0,
            "avg_changed_sources": 0.0,
            "avg_chunks": 0.0,
        }

    latencies = [run.metrics.total_ms for run in runs]
    success_rate = fmean(1.0 if run.state == RunState.SUCCESS else 0.0 for run in runs)
    avg_changed = fmean(run.changed_count for run in runs)
    avg_chunks = fmean(run.chunk_count for run in runs)

    return {
        "run_count": len(runs),
        "success_rate": round(success_rate, 4),
        "avg_total_ms": int(fmean(latencies)),
        "p95_total_ms": percentile(latencies, 0.95),
        "avg_changed_sources": round(avg_changed, 4),
        "avg_chunks": round(avg_chunks, 4),
    }


def run_to_json(run: IngestionRun) -> str:
    payload = {
        "run_id": run.run_id,
        "pipeline_id": run.pipeline_id,
        "state": run.state.value,
        "version_id": run.version_id,
        "source_count": run.source_count,
        "fetched_count": run.fetched_count,
        "normalized_count": run.normalized_count,
        "changed_count": run.changed_count,
        "chunk_count": run.chunk_count,
        "indexed_count": run.indexed_count,
        "queued_tasks": run.queued_tasks,
        "completed_tasks": run.completed_tasks,
        "failed_tasks": run.failed_tasks,
        "failure_kind": run.failure_kind.value,
        "warnings": run.warnings,
        "reason": run.reason,
        "metrics": {
            "fetch_ms": run.metrics.fetch_ms,
            "normalize_ms": run.metrics.normalize_ms,
            "planning_ms": run.metrics.planning_ms,
            "chunk_ms": run.metrics.chunk_ms,
            "queue_ms": run.metrics.queue_ms,
            "index_ms": run.metrics.index_ms,
            "scheduler_ms": run.metrics.scheduler_ms,
            "total_ms": run.metrics.total_ms,
        },
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
