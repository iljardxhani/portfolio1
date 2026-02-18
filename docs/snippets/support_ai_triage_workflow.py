from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Iterable
from typing import Protocol
from typing import Sequence
import re
import time

class Channel(str, Enum):
    EMAIL = "email"
    DM = "dm"
    WEB = "web"
    API = "api"
class Priority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    SECURITY = "security"
    LEGAL = "legal"
    SALES = "sales"
    OTHER = "other"
class TicketStatus(str, Enum):
    OPEN = "open"
    PENDING_HUMAN_APPROVAL = "pending_human_approval"
    APPROVED = "approved"
    AUTO_SENT = "auto_sent"
    RESOLVED = "resolved"
    BLOCKED = "blocked"
class ApprovalState(str, Enum):
    NOT_REQUIRED = "not_required"
    QUEUED = "queued"
    APPROVED = "approved"
    REJECTED = "rejected"
class RiskFlag(str, Enum):
    NONE = "none"
    LEGAL_THREAT = "legal_threat"
    ACCOUNT_TAKEOVER = "account_takeover"
    PAYMENT_DISPUTE = "payment_dispute"
    PII_EXPOSURE = "pii_exposure"
    ABUSE_LANGUAGE = "abuse_language"
    SECURITY_INCIDENT = "security_incident"
class FailureReason(str, Enum):
    NONE = "none"
    DUPLICATE_MESSAGE = "duplicate_message"
    VALIDATION_FAILED = "validation_failed"
    CLASSIFICATION_FAILED = "classification_failed"
    POLICY_BLOCKED = "policy_blocked"
    TICKETING_FAILED = "ticketing_failed"
    APPROVAL_FAILED = "approval_failed"
    DRAFT_FAILED = "draft_failed"
@dataclass(slots=True)
class SupportMessage:
    message_id: str
    channel: Channel
    customer_id: str
    customer_email: str
    subject: str
    body: str
    language_hint: str
    received_at_ms: int
    attachments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
@dataclass(slots=True)
class CustomerProfile:
    customer_id: str
    plan: str
    region: str
    tier: str
    has_open_incident: bool
    recent_chargeback: bool
    lifetime_value_usd: float
    preferred_language: str = "en"
@dataclass(slots=True)
class AttachmentScanResult:
    scanned_files: int
    blocked_files: int
    has_risky_content: bool
    reasons: list[str]
@dataclass(slots=True)
class PreparedMessage:
    message: SupportMessage
    cleaned_subject: str
    cleaned_body: str
    normalized_text: str
    language: str
    pii_redactions: int
    attachment_scan: AttachmentScanResult
@dataclass(slots=True)
class ClassificationOutput:
    category: Category
    priority: Priority
    confidence: float
    intent_labels: list[str]
    sentiment: float
    raw: dict[str, Any]
@dataclass(slots=True)
class EscalationSignal:
    score: float
    risk_flags: list[RiskFlag]
    reasons: list[str]
@dataclass(slots=True)
class RoutingDecision:
    category: Category
    priority: Priority
    target_queue: str
    owner_group: str
    sla_minutes: int
    requires_human_approval: bool
    auto_reply_allowed: bool
    escalation: EscalationSignal
    policy_notes: list[str]
@dataclass(slots=True)
class DraftReply:
    text: str
    tone: str
    template_id: str
    confidence: float
    guardrail_notes: list[str]
@dataclass(slots=True)
class TicketRecord:
    ticket_id: str
    priority: Priority
    category: Category
    queue: str
    status: TicketStatus
    owner: str
    created_at_ms: int
@dataclass(slots=True)
class ApprovalRecord:
    approval_id: str
    ticket_id: str
    state: ApprovalState
    reviewer_group: str
    queued_at_ms: int
@dataclass(slots=True)
class PolicySnapshot:
    min_classifier_confidence: float
    urgent_risk_threshold: float
    high_risk_threshold: float
    blocklist_terms: set[str]
    requires_human_for_categories: set[Category]
    requires_human_for_tiers: set[str]
    sla_minutes: dict[Priority, int]
    queue_map: dict[Category, str]
@dataclass(slots=True)
class ProcessingMetrics:
    preprocess_ms: int
    classify_ms: int
    policy_ms: int
    draft_ms: int
    ticket_ms: int
    approval_ms: int
    total_ms: int
@dataclass(slots=True)
class WorkflowResult:
    message_id: str
    ticket_id: str | None
    status: TicketStatus
    priority: Priority | None
    queue: str | None
    requires_human_approval: bool
    approval_state: ApprovalState
    failure_reason: FailureReason
    risk_flags: list[RiskFlag]
    metrics: ProcessingMetrics
class ClassifierClient(Protocol):
    def classify(self, text: str, *, channel: str, language: str) -> dict[str, Any]: ...
class DraftClient(Protocol):
    def draft_reply(
        self,
        *,
        message: PreparedMessage,
        category: Category,
        priority: Priority,
        tone: str,
        template_hint: str,
    ) -> str: ...
class TicketingClient(Protocol):
    def create_ticket(self, payload: dict[str, Any]) -> TicketRecord: ...
class ApprovalQueue(Protocol):
    def enqueue(self, *, ticket: TicketRecord, draft: DraftReply, reviewer_group: str) -> ApprovalRecord: ...
class AuditLog(Protocol):
    def write(self, event_type: str, payload: dict[str, Any]) -> None: ...
class NotificationClient(Protocol):
    def send_internal(self, channel: str, message: str) -> None: ...
class IdempotencyStore(Protocol):
    def seen(self, message_id: str) -> bool: ...
    def mark(self, message_id: str, payload: dict[str, Any]) -> None: ...
    def get(self, message_id: str) -> dict[str, Any] | None: ...
class PolicyStore(Protocol):
    def get(self) -> PolicySnapshot: ...
class MetricsSink(Protocol):
    def observe(self, metric: str, value: float, tags: dict[str, str] | None = None) -> None: ...
class TemplateLibrary(Protocol):
    def choose_template(self, *, category: Category, priority: Priority, language: str) -> str: ...
class PiiRedactor(Protocol):
    def redact(self, text: str) -> tuple[str, int]: ...
class LanguageDetector(Protocol):
    def detect(self, text: str, hint: str | None = None) -> str: ...
class AttachmentScanner(Protocol):
    def scan(self, attachments: Sequence[dict[str, Any]]) -> AttachmentScanResult: ...
class SimilarCaseSearcher(Protocol):
    def find_similar(self, text: str, top_k: int = 3) -> list[dict[str, Any]]: ...
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
def normalize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    return compact
def lower_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_@.\-]{2,}", text.lower())
def contains_any(text: str, terms: Iterable[str]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)
def detect_email_addresses(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
def detect_card_like_numbers(text: str) -> list[str]:
    return re.findall(r"\b(?:\d[ -]*?){13,16}\b", text)
def sanitize_subject(subject: str) -> str:
    stripped = normalize_text(subject)
    return stripped[:220]
def sanitize_body(body: str) -> str:
    stripped = normalize_text(body)
    return stripped[:12000]
def message_text(message: SupportMessage) -> str:
    return normalize_text(f"{message.subject} {message.body}")
def estimate_tokens(text: str) -> int:
    words = len(lower_words(text))
    return max(1, int(words * 1.35))
def classify_priority_name(priority: str) -> Priority:
    normalized = priority.strip().lower()
    if normalized == "urgent":
        return Priority.URGENT
    if normalized == "high":
        return Priority.HIGH
    if normalized == "low":
        return Priority.LOW
    return Priority.NORMAL
def classify_category_name(category: str) -> Category:
    normalized = category.strip().lower()
    for item in Category:
        if item.value == normalized:
            return item
    return Category.OTHER
def dedupe_flags(flags: Iterable[RiskFlag]) -> list[RiskFlag]:
    seen: set[RiskFlag] = set()
    ordered: list[RiskFlag] = []
    for flag in flags:
        if flag in seen:
            continue
        seen.add(flag)
        ordered.append(flag)
    return ordered
def preprocess_message(
    message: SupportMessage,
    pii_redactor: PiiRedactor,
    language_detector: LanguageDetector,
    attachment_scanner: AttachmentScanner,
) -> PreparedMessage:
    clean_subject = sanitize_subject(message.subject)
    clean_body = sanitize_body(message.body)
    redacted_body, redaction_count = pii_redactor.redact(clean_body)
    combined = normalize_text(f"{clean_subject} {redacted_body}")
    language = language_detector.detect(combined, hint=message.language_hint)
    attachment_result = attachment_scanner.scan(message.attachments)
    return PreparedMessage(
        message=message,
        cleaned_subject=clean_subject,
        cleaned_body=redacted_body,
        normalized_text=combined,
        language=language,
        pii_redactions=redaction_count,
        attachment_scan=attachment_result,
    )
def parse_classification(raw: dict[str, Any]) -> ClassificationOutput:
    category = classify_category_name(str(raw.get("category", "other")))
    priority = classify_priority_name(str(raw.get("priority", "normal")))
    confidence = float(raw.get("confidence", 0.0))
    labels_raw = raw.get("intent_labels", [])
    labels = [str(item) for item in labels_raw] if isinstance(labels_raw, list) else []
    sentiment = float(raw.get("sentiment", 0.0))
    return ClassificationOutput(
        category=category,
        priority=priority,
        confidence=clamp(confidence, 0.0, 1.0),
        intent_labels=labels,
        sentiment=clamp(sentiment, -1.0, 1.0),
        raw=raw,
    )
def classify_with_retry(
    classifier: ClassifierClient,
    prepared: PreparedMessage,
    *,
    retries: int = 2,
) -> ClassificationOutput:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = classifier.classify(
                prepared.normalized_text,
                channel=prepared.message.channel.value,
                language=prepared.language,
            )
            return parse_classification(raw)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries:
                raise
            time.sleep(0.02 * (attempt + 1))
    raise RuntimeError(f"Classification failed: {last_error}")
def compute_risk_score(
    prepared: PreparedMessage,
    classification: ClassificationOutput,
    customer: CustomerProfile,
    policy: PolicySnapshot,
) -> EscalationSignal:
    text = prepared.normalized_text.lower()
    flags: list[RiskFlag] = []
    reasons: list[str] = []
    score = 0.0
    legal_terms = ["lawsuit", "attorney", "legal notice", "court", "cease and desist"]
    security_terms = ["hacked", "breach", "unauthorized", "takeover", "compromised"]
    payment_terms = ["chargeback", "fraud charge", "refund now", "bank dispute"]
    abuse_terms = ["idiot", "scam", "thief", "fraud company", "worst service"]
    if contains_any(text, legal_terms):
        flags.append(RiskFlag.LEGAL_THREAT)
        reasons.append("legal_terms_detected")
        score += 0.42
    if contains_any(text, security_terms) or classification.category == Category.SECURITY:
        flags.append(RiskFlag.SECURITY_INCIDENT)
        reasons.append("security_signal_detected")
        score += 0.48
    if contains_any(text, payment_terms) or customer.recent_chargeback:
        flags.append(RiskFlag.PAYMENT_DISPUTE)
        reasons.append("payment_dispute_signal")
        score += 0.31
    if contains_any(text, abuse_terms):
        flags.append(RiskFlag.ABUSE_LANGUAGE)
        reasons.append("abusive_language")
        score += 0.12
    if detect_email_addresses(prepared.cleaned_body) or detect_card_like_numbers(prepared.cleaned_body):
        flags.append(RiskFlag.PII_EXPOSURE)
        reasons.append("pii_detected")
        score += 0.20
    if prepared.attachment_scan.has_risky_content:
        flags.append(RiskFlag.ACCOUNT_TAKEOVER)
        reasons.append("risky_attachment")
        score += 0.36
    if classification.confidence < policy.min_classifier_confidence:
        reasons.append("low_classifier_confidence")
        score += 0.18
    if customer.has_open_incident:
        reasons.append("customer_has_open_incident")
        score += 0.14
    if customer.tier.lower() in {"enterprise", "vip"}:
        reasons.append("high_business_impact_customer")
        score += 0.08
    if not flags:
        flags = [RiskFlag.NONE]
    return EscalationSignal(
        score=round(clamp(score, 0.0, 1.0), 4),
        risk_flags=dedupe_flags(flags),
        reasons=reasons,
    )
def choose_priority(
    classification: ClassificationOutput,
    escalation: EscalationSignal,
    customer: CustomerProfile,
    policy: PolicySnapshot,
) -> Priority:
    priority = classification.priority
    if escalation.score >= policy.urgent_risk_threshold:
        return Priority.URGENT
    if escalation.score >= policy.high_risk_threshold and priority in {Priority.NORMAL, Priority.LOW}:
        priority = Priority.HIGH
    if customer.tier.lower() in {"enterprise", "vip"} and priority == Priority.NORMAL:
        priority = Priority.HIGH
    if classification.confidence < policy.min_classifier_confidence and priority == Priority.LOW:
        priority = Priority.NORMAL
    return priority
def choose_queue(category: Category, priority: Priority, policy: PolicySnapshot) -> str:
    base = policy.queue_map.get(category, "support-general")
    if priority == Priority.URGENT:
        return f"{base}-urgent"
    if priority == Priority.HIGH:
        return f"{base}-priority"
    return base
def choose_owner_group(category: Category, escalation: EscalationSignal) -> str:
    if RiskFlag.SECURITY_INCIDENT in escalation.risk_flags:
        return "secops"
    if RiskFlag.LEGAL_THREAT in escalation.risk_flags:
        return "legal-ops"
    if category == Category.BILLING:
        return "billing-ops"
    if category == Category.TECHNICAL:
        return "support-l2"
    if category == Category.ACCOUNT:
        return "account-services"
    return "support-general"
def requires_human(
    category: Category,
    priority: Priority,
    escalation: EscalationSignal,
    customer: CustomerProfile,
    classification: ClassificationOutput,
    policy: PolicySnapshot,
) -> bool:
    if category in policy.requires_human_for_categories:
        return True
    if customer.tier.lower() in {item.lower() for item in policy.requires_human_for_tiers}:
        return True
    if priority == Priority.URGENT:
        return True
    if escalation.score >= policy.high_risk_threshold:
        return True
    if classification.confidence < policy.min_classifier_confidence:
        return True
    if RiskFlag.PII_EXPOSURE in escalation.risk_flags:
        return True
    return False
def auto_reply_allowed(
    *,
    category: Category,
    escalation: EscalationSignal,
    requires_human_approval: bool,
) -> bool:
    if requires_human_approval:
        return False
    if category in {Category.SECURITY, Category.LEGAL}:
        return False
    if escalation.score > 0.45:
        return False
    return True
def resolve_routing(
    prepared: PreparedMessage,
    classification: ClassificationOutput,
    customer: CustomerProfile,
    policy: PolicySnapshot,
) -> RoutingDecision:
    risk = compute_risk_score(prepared, classification, customer, policy)
    priority = choose_priority(classification, risk, customer, policy)
    queue = choose_queue(classification.category, priority, policy)
    owner_group = choose_owner_group(classification.category, risk)
    require_approval = requires_human(
        classification.category,
        priority,
        risk,
        customer,
        classification,
        policy,
    )
    allow_auto = auto_reply_allowed(
        category=classification.category,
        escalation=risk,
        requires_human_approval=require_approval,
    )
    notes: list[str] = []
    if classification.confidence < policy.min_classifier_confidence:
        notes.append("confidence_below_policy")
    if risk.score >= policy.urgent_risk_threshold:
        notes.append("risk_forced_urgent")
    if require_approval:
        notes.append("human_approval_required")
    return RoutingDecision(
        category=classification.category,
        priority=priority,
        target_queue=queue,
        owner_group=owner_group,
        sla_minutes=policy.sla_minutes.get(priority, 240),
        requires_human_approval=require_approval,
        auto_reply_allowed=allow_auto,
        escalation=risk,
        policy_notes=notes,
    )
def classify_tone(decision: RoutingDecision, sentiment: float) -> str:
    if decision.priority == Priority.URGENT:
        return "calm-urgent"
    if sentiment < -0.45:
        return "de-escalate"
    if decision.category == Category.SALES:
        return "friendly-informative"
    return "professional"
def select_template(
    templates: TemplateLibrary,
    decision: RoutingDecision,
    prepared: PreparedMessage,
) -> str:
    return templates.choose_template(
        category=decision.category,
        priority=decision.priority,
        language=prepared.language,
    )
def clean_draft(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("<customer>", "there")
    return text[:3500]
def enforce_draft_guardrails(
    text: str,
    decision: RoutingDecision,
    policy: PolicySnapshot,
) -> tuple[str, list[str], float]:
    notes: list[str] = []
    confidence = 0.88
    if contains_any(text.lower(), policy.blocklist_terms):
        notes.append("blocked_term_rewritten")
        for term in policy.blocklist_terms:
            text = re.sub(re.escape(term), "[redacted]", text, flags=re.IGNORECASE)
        confidence -= 0.12
    if len(text) < 60:
        notes.append("draft_too_short")
        confidence -= 0.16
    if decision.priority == Priority.URGENT and "urgent" not in text.lower():
        notes.append("urgency_language_missing")
        confidence -= 0.06
    if RiskFlag.LEGAL_THREAT in decision.escalation.risk_flags:
        notes.append("legal_disclaimer_required")
        text += " We have escalated this case to our legal response team."
        confidence -= 0.05
    if RiskFlag.SECURITY_INCIDENT in decision.escalation.risk_flags:
        notes.append("security_protocol_notice")
        text += " Our security team is now reviewing this incident with priority."
        confidence -= 0.05
    return clean_draft(text), notes, round(clamp(confidence, 0.0, 1.0), 4)
def build_draft(
    prepared: PreparedMessage,
    decision: RoutingDecision,
    drafter: DraftClient,
    templates: TemplateLibrary,
    policy: PolicySnapshot,
) -> DraftReply:
    tone = classify_tone(decision, sentiment=0.0)
    template_id = select_template(templates, decision, prepared)
    raw = drafter.draft_reply(
        message=prepared,
        category=decision.category,
        priority=decision.priority,
        tone=tone,
        template_hint=template_id,
    )
    cleaned, notes, conf = enforce_draft_guardrails(raw, decision, policy)
    return DraftReply(
        text=cleaned,
        tone=tone,
        template_id=template_id,
        confidence=conf,
        guardrail_notes=notes,
    )
def build_ticket_payload(
    prepared: PreparedMessage,
    customer: CustomerProfile,
    decision: RoutingDecision,
    classification: ClassificationOutput,
    draft: DraftReply,
    similar_cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "external_message_id": prepared.message.message_id,
        "customer_id": customer.customer_id,
        "customer_plan": customer.plan,
        "subject": prepared.cleaned_subject,
        "body": prepared.cleaned_body,
        "channel": prepared.message.channel.value,
        "language": prepared.language,
        "category": decision.category.value,
        "priority": decision.priority.value,
        "queue": decision.target_queue,
        "owner_group": decision.owner_group,
        "sla_minutes": decision.sla_minutes,
        "requires_human_approval": decision.requires_human_approval,
        "auto_reply_allowed": decision.auto_reply_allowed,
        "classifier_confidence": classification.confidence,
        "intent_labels": classification.intent_labels,
        "risk_score": decision.escalation.score,
        "risk_flags": [flag.value for flag in decision.escalation.risk_flags],
        "policy_notes": list(decision.policy_notes),
        "draft_preview": draft.text[:320],
        "draft_template": draft.template_id,
        "similar_case_ids": [str(case.get("ticket_id", "")) for case in similar_cases],
        "attachments": prepared.message.attachments,
        "token_estimate": estimate_tokens(prepared.normalized_text),
    }
def emit_audit_start(audit: AuditLog, message: SupportMessage) -> None:
    audit.write(
        event_type="triage.started",
        payload={
            "message_id": message.message_id,
            "channel": message.channel.value,
            "customer_id": message.customer_id,
            "received_at_ms": message.received_at_ms,
        },
    )
def emit_audit_policy(
    audit: AuditLog,
    message: SupportMessage,
    decision: RoutingDecision,
    classification: ClassificationOutput,
) -> None:
    audit.write(
        event_type="triage.policy.decided",
        payload={
            "message_id": message.message_id,
            "category": decision.category.value,
            "priority": decision.priority.value,
            "queue": decision.target_queue,
            "owner_group": decision.owner_group,
            "requires_human_approval": decision.requires_human_approval,
            "auto_reply_allowed": decision.auto_reply_allowed,
            "risk_score": decision.escalation.score,
            "risk_flags": [item.value for item in decision.escalation.risk_flags],
            "classifier_confidence": classification.confidence,
            "policy_notes": decision.policy_notes,
        },
    )
def emit_audit_complete(
    audit: AuditLog,
    result: WorkflowResult,
    draft: DraftReply | None,
    approval: ApprovalRecord | None,
) -> None:
    audit.write(
        event_type="triage.completed",
        payload={
            "message_id": result.message_id,
            "ticket_id": result.ticket_id,
            "status": result.status.value,
            "priority": result.priority.value if result.priority else None,
            "queue": result.queue,
            "requires_human_approval": result.requires_human_approval,
            "approval_state": result.approval_state.value,
            "failure_reason": result.failure_reason.value,
            "risk_flags": [flag.value for flag in result.risk_flags],
            "metrics": {
                "preprocess_ms": result.metrics.preprocess_ms,
                "classify_ms": result.metrics.classify_ms,
                "policy_ms": result.metrics.policy_ms,
                "draft_ms": result.metrics.draft_ms,
                "ticket_ms": result.metrics.ticket_ms,
                "approval_ms": result.metrics.approval_ms,
                "total_ms": result.metrics.total_ms,
            },
            "draft_confidence": draft.confidence if draft else None,
            "draft_template": draft.template_id if draft else None,
            "approval_id": approval.approval_id if approval else None,
        },
    )
def notify_urgent(
    notify: NotificationClient,
    ticket: TicketRecord,
    decision: RoutingDecision,
    customer: CustomerProfile,
) -> None:
    if decision.priority != Priority.URGENT:
        return
    notify.send_internal(
        "support-priority",
        (
            f"Urgent triage ticket {ticket.ticket_id} queue={decision.target_queue} "
            f"customer={customer.customer_id} risk={decision.escalation.score:.2f}"
        ),
    )
def result_from_cache(payload: dict[str, Any], message_id: str) -> WorkflowResult:
    metrics = payload.get("metrics", {})
    return WorkflowResult(
        message_id=message_id,
        ticket_id=payload.get("ticket_id"),
        status=TicketStatus(str(payload.get("status", TicketStatus.OPEN.value))),
        priority=Priority(str(payload.get("priority", Priority.NORMAL.value))),
        queue=str(payload.get("queue", "support-general")),
        requires_human_approval=bool(payload.get("requires_human_approval", False)),
        approval_state=ApprovalState(str(payload.get("approval_state", ApprovalState.NOT_REQUIRED.value))),
        failure_reason=FailureReason.NONE,
        risk_flags=[RiskFlag.NONE],
        metrics=ProcessingMetrics(
            preprocess_ms=int(metrics.get("preprocess_ms", 0)),
            classify_ms=int(metrics.get("classify_ms", 0)),
            policy_ms=int(metrics.get("policy_ms", 0)),
            draft_ms=int(metrics.get("draft_ms", 0)),
            ticket_ms=int(metrics.get("ticket_ms", 0)),
            approval_ms=int(metrics.get("approval_ms", 0)),
            total_ms=int(metrics.get("total_ms", 0)),
        ),
    )
def failed_result(
    message_id: str,
    reason: FailureReason,
    *,
    total_ms: int,
) -> WorkflowResult:
    return WorkflowResult(
        message_id=message_id,
        ticket_id=None,
        status=TicketStatus.BLOCKED,
        priority=None,
        queue=None,
        requires_human_approval=False,
        approval_state=ApprovalState.NOT_REQUIRED,
        failure_reason=reason,
        risk_flags=[RiskFlag.NONE],
        metrics=ProcessingMetrics(
            preprocess_ms=0,
            classify_ms=0,
            policy_ms=0,
            draft_ms=0,
            ticket_ms=0,
            approval_ms=0,
            total_ms=total_ms,
        ),
    )
def process_support_message(
    *,
    message: SupportMessage,
    customer_profile: CustomerProfile,
    classifier: ClassifierClient,
    drafter: DraftClient,
    ticketing: TicketingClient,
    approvals: ApprovalQueue,
    audit: AuditLog,
    notify: NotificationClient,
    idempotency: IdempotencyStore,
    policy_store: PolicyStore,
    metrics: MetricsSink,
    templates: TemplateLibrary,
    pii_redactor: PiiRedactor,
    language_detector: LanguageDetector,
    attachment_scanner: AttachmentScanner,
    similar_cases: SimilarCaseSearcher,
) -> WorkflowResult:
    start_total = time.perf_counter()
    cached = idempotency.get(message.message_id)
    if cached is not None:
        return result_from_cache(cached, message.message_id)
    if idempotency.seen(message.message_id):
        total_ms = int((time.perf_counter() - start_total) * 1000)
        return failed_result(message.message_id, FailureReason.DUPLICATE_MESSAGE, total_ms=total_ms)
    emit_audit_start(audit, message)
    try:
        preprocess_start = time.perf_counter()
        prepared = preprocess_message(message, pii_redactor, language_detector, attachment_scanner)
        preprocess_ms = int((time.perf_counter() - preprocess_start) * 1000)
        if not prepared.cleaned_body and not prepared.cleaned_subject:
            raise ValueError("empty_message_payload")
        classify_start = time.perf_counter()
        classification = classify_with_retry(classifier, prepared, retries=2)
        classify_ms = int((time.perf_counter() - classify_start) * 1000)
        policy_start = time.perf_counter()
        policy = policy_store.get()
        decision = resolve_routing(prepared, classification, customer_profile, policy)
        policy_ms = int((time.perf_counter() - policy_start) * 1000)
        emit_audit_policy(audit, message, decision, classification)
        draft_start = time.perf_counter()
        draft = build_draft(prepared, decision, drafter, templates, policy)
        draft_ms = int((time.perf_counter() - draft_start) * 1000)
        similar = similar_cases.find_similar(prepared.normalized_text, top_k=3)
        ticket_start = time.perf_counter()
        payload = build_ticket_payload(
            prepared,
            customer_profile,
            decision,
            classification,
            draft,
            similar,
        )
        ticket = ticketing.create_ticket(payload)
        ticket_ms = int((time.perf_counter() - ticket_start) * 1000)
        approval_start = time.perf_counter()
        approval_record: ApprovalRecord | None = None
        approval_state = ApprovalState.NOT_REQUIRED
        if decision.requires_human_approval:
            approval_record = approvals.enqueue(
                ticket=ticket,
                draft=draft,
                reviewer_group=decision.owner_group,
            )
            ticket.status = TicketStatus.PENDING_HUMAN_APPROVAL
            approval_state = approval_record.state
        elif decision.auto_reply_allowed:
            ticket.status = TicketStatus.AUTO_SENT
            approval_state = ApprovalState.NOT_REQUIRED
        else:
            ticket.status = TicketStatus.OPEN
            approval_state = ApprovalState.NOT_REQUIRED
        approval_ms = int((time.perf_counter() - approval_start) * 1000)
        notify_urgent(notify, ticket, decision, customer_profile)
        total_ms = int((time.perf_counter() - start_total) * 1000)
        result = WorkflowResult(
            message_id=message.message_id,
            ticket_id=ticket.ticket_id,
            status=ticket.status,
            priority=decision.priority,
            queue=decision.target_queue,
            requires_human_approval=decision.requires_human_approval,
            approval_state=approval_state,
            failure_reason=FailureReason.NONE,
            risk_flags=decision.escalation.risk_flags,
            metrics=ProcessingMetrics(
                preprocess_ms=preprocess_ms,
                classify_ms=classify_ms,
                policy_ms=policy_ms,
                draft_ms=draft_ms,
                ticket_ms=ticket_ms,
                approval_ms=approval_ms,
                total_ms=total_ms,
            ),
        )
        idempotency.mark(
            message.message_id,
            {
                "ticket_id": result.ticket_id,
                "status": result.status.value,
                "priority": result.priority.value if result.priority else None,
                "queue": result.queue,
                "requires_human_approval": result.requires_human_approval,
                "approval_state": result.approval_state.value,
                "metrics": {
                    "preprocess_ms": result.metrics.preprocess_ms,
                    "classify_ms": result.metrics.classify_ms,
                    "policy_ms": result.metrics.policy_ms,
                    "draft_ms": result.metrics.draft_ms,
                    "ticket_ms": result.metrics.ticket_ms,
                    "approval_ms": result.metrics.approval_ms,
                    "total_ms": result.metrics.total_ms,
                },
            },
        )
        emit_audit_complete(audit, result, draft, approval_record)
        metrics.observe("triage.total_ms", float(result.metrics.total_ms), {"queue": result.queue or "none"})
        metrics.observe("triage.risk_score", float(decision.escalation.score), {"priority": decision.priority.value})
        metrics.observe("triage.classifier_confidence", float(classification.confidence))
        return result
    except ValueError:
        total_ms = int((time.perf_counter() - start_total) * 1000)
        result = failed_result(message.message_id, FailureReason.VALIDATION_FAILED, total_ms=total_ms)
        emit_audit_complete(audit, result, None, None)
        return result
    except RuntimeError:
        total_ms = int((time.perf_counter() - start_total) * 1000)
        result = failed_result(message.message_id, FailureReason.CLASSIFICATION_FAILED, total_ms=total_ms)
        emit_audit_complete(audit, result, None, None)
        return result
    except Exception:  # noqa: BLE001
        total_ms = int((time.perf_counter() - start_total) * 1000)
        result = failed_result(message.message_id, FailureReason.TICKETING_FAILED, total_ms=total_ms)
        emit_audit_complete(audit, result, None, None)
        return result
