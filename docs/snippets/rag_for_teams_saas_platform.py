from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from statistics import fmean
from typing import Any, Protocol, Sequence
import hashlib
import hmac
import json
import re
import secrets
import time
class PlanTier(str, Enum):
    STARTER = "starter"
    GROWTH = "growth"
    ENTERPRISE = "enterprise"
class Role(str, Enum):
    MEMBER = "member"
    ADMIN = "admin"
    OWNER = "owner"
    SUPPORT = "support"
class OrgState(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELINQUENT = "delinquent"
class QueryMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"
class RequestState(str, Enum):
    SUCCESS = "success"
    DENIED = "denied"
    FAILED = "failed"
class AuditKind(str, Enum):
    AUTH = "auth"
    QUERY = "query"
    BILLING = "billing"
    ADMIN = "admin"
    SAFETY = "safety"
class EventSeverity(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"
class ChargeType(str, Enum):
    REQUEST = "request"
    TOKEN_INPUT = "token_input"
    TOKEN_OUTPUT = "token_output"
    OVERAGE = "overage"
class InvoiceState(str, Enum):
    OPEN = "open"
    PAID = "paid"
    VOID = "void"
@dataclass(slots=True)
class PlanPolicy:
    tier: PlanTier
    monthly_token_quota: int
    requests_per_minute: int
    burst_capacity: int
    default_top_k: int
    max_top_k: int
    allowed_features: set[str]
    allow_overage: bool
    price_per_1k_input_tokens_cents: int
    price_per_1k_output_tokens_cents: int
    overage_per_1k_tokens_cents: int
@dataclass(slots=True)
class OrgRecord:
    org_id: str
    name: str
    tier: PlanTier
    state: OrgState
    credit_balance_cents: int
    created_at_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)
    safety_switches: set[str] = field(default_factory=set)
@dataclass(slots=True)
class UserRecord:
    user_id: str
    org_id: str
    email: str
    role: Role
    is_active: bool
@dataclass(slots=True)
class TokenClaims:
    sub: str
    org_id: str
    role: Role
    scopes: tuple[str, ...]
    issued_at_ms: int
    exp_ms: int
    nonce: str
@dataclass(slots=True)
class AuthContext:
    request_id: str
    user: UserRecord
    org: OrgRecord
    scopes: tuple[str, ...]
    token_nonce: str
@dataclass(slots=True)
class Citation:
    doc_id: str
    title: str
    uri: str
    score: float
@dataclass(slots=True)
class RAGAnswer:
    answer: str
    citations: list[Citation]
    model: str
    retrieval_ms: int
    generation_ms: int
    input_tokens: int
    output_tokens: int
@dataclass(slots=True)
class RequestEnvelope:
    request_id: str
    token: str
    endpoint: str
    query: str
    mode: QueryMode
    top_k: int | None
    request_ip: str
    user_agent: str
    metadata: dict[str, Any] = field(default_factory=dict)
@dataclass(slots=True)
class RateDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_at_ms: int
    reason: str
@dataclass(slots=True)
class QuotaDecision:
    allowed: bool
    monthly_quota: int
    used_before: int
    used_after: int
    remaining: int
    reason: str
@dataclass(slots=True)
class UsageEvent:
    event_id: str
    org_id: str
    user_id: str
    endpoint: str
    mode: QueryMode
    input_tokens: int
    output_tokens: int
    billable_cents: int
    request_latency_ms: int
    created_at_ms: int
@dataclass(slots=True)
class BillingLine:
    line_id: str
    charge_type: ChargeType
    description: str
    quantity: int
    unit_price_cents: int
    amount_cents: int
    metadata: dict[str, Any] = field(default_factory=dict)
@dataclass(slots=True)
class Invoice:
    invoice_id: str
    org_id: str
    state: InvoiceState
    issued_at_ms: int
    period_start_ms: int
    period_end_ms: int
    subtotal_cents: int
    tax_cents: int
    total_cents: int
    lines: list[BillingLine] = field(default_factory=list)
@dataclass(slots=True)
class AuditRecord:
    audit_id: str
    kind: AuditKind
    request_id: str
    org_id: str
    user_id: str
    endpoint: str
    status_code: int
    state: RequestState
    latency_ms: int
    message: str
    details: dict[str, Any]
    created_at_ms: int
@dataclass(slots=True)
class AdminEvent:
    event_id: str
    org_id: str
    actor_user_id: str
    severity: EventSeverity
    event_type: str
    payload: dict[str, Any]
    created_at_ms: int
@dataclass(slots=True)
class IncidentPolicy:
    incident_id: str
    title: str
    enabled: bool
    blocked_endpoints: set[str] = field(default_factory=set)
    blocked_org_ids: set[str] = field(default_factory=set)
    allow_admin_override: bool = False
@dataclass(slots=True)
class WebhookEvent:
    provider: str
    event_id: str
    event_type: str
    org_id: str
    amount_cents: int
    currency: str
    payload: dict[str, Any]
    published_at_ms: int
@dataclass(slots=True)
class DashboardSnapshot:
    org_id: str
    plan: PlanTier
    state: OrgState
    window_start_ms: int
    window_end_ms: int
    total_requests: int
    success_rate: float
    avg_latency_ms: int
    p95_latency_ms: int
    token_usage: int
    billable_cents: int
    rate_limit_hits: int
    safety_blocks: int
class Clock(Protocol):
    def now_ms(self) -> int: ...
class IdGenerator(Protocol):
    def new_id(self, prefix: str) -> str: ...
class TokenVerifier(Protocol):
    def verify(self, token: str) -> TokenClaims: ...
class OrgRepository(Protocol):
    def get(self, org_id: str) -> OrgRecord | None: ...
    def save(self, org: OrgRecord) -> None: ...
class UserRepository(Protocol):
    def get(self, user_id: str) -> UserRecord | None: ...
class PlanCatalog(Protocol):
    def get(self, tier: PlanTier) -> PlanPolicy: ...
class RateLimitRepository(Protocol):
    def consume(self, org_id: str, endpoint: str, limit: int, burst: int, now_ms: int) -> RateDecision: ...
class UsageRepository(Protocol):
    def monthly_tokens(self, org_id: str, month_key: str) -> int: ...
    def append(self, event: UsageEvent) -> None: ...
    def list(self, org_id: str, start_ms: int, end_ms: int) -> list[UsageEvent]: ...
class BillingRepository(Protocol):
    def open_invoice(self, org_id: str, period_start_ms: int, period_end_ms: int, issued_at_ms: int) -> Invoice: ...
    def add_line(self, invoice_id: str, line: BillingLine) -> None: ...
    def mark_paid(self, invoice_id: str, payment_ref: str) -> None: ...
    def latest_for_org(self, org_id: str) -> Invoice | None: ...
class AuditSink(Protocol):
    def write(self, record: AuditRecord) -> None: ...
    def list(self, org_id: str, start_ms: int, end_ms: int) -> list[AuditRecord]: ...
class EventSink(Protocol):
    def write(self, event: AdminEvent) -> None: ...
    def list(self, org_id: str, start_ms: int, end_ms: int) -> list[AdminEvent]: ...
class RAGRuntime(Protocol):
    def answer(self, org_id: str, query: str, top_k: int, mode: QueryMode) -> RAGAnswer: ...
class SecretStore(Protocol):
    def get(self, key: str) -> str | None: ...
class APIKeyRepository(Protocol):
    def rotate(self, org_id: str, label: str, rotated_by: str, created_at_ms: int) -> str: ...
@dataclass(slots=True)
class PlatformDeps:
    clock: Clock
    ids: IdGenerator
    tokens: TokenVerifier
    orgs: OrgRepository
    users: UserRepository
    plans: PlanCatalog
    rate_limits: RateLimitRepository
    usage: UsageRepository
    billing: BillingRepository
    audits: AuditSink
    events: EventSink
    rag: RAGRuntime
    secrets: SecretStore
    api_keys: APIKeyRepository
class PlatformError(Exception):
    def __init__(self, code: str, message: str, status_code: int, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details or {}
class UnauthorizedError(PlatformError):
    def __init__(self, message: str = "Unauthorized") -> None:
        super().__init__("unauthorized", message, 401)
class ForbiddenError(PlatformError):
    def __init__(self, message: str = "Forbidden") -> None:
        super().__init__("forbidden", message, 403)
class SuspendedOrgError(PlatformError):
    def __init__(self, org_id: str, state: OrgState) -> None:
        super().__init__("org_blocked", f"Organization {org_id} is {state.value}", 403, {"org_state": state.value})
class RateLimitError(PlatformError):
    def __init__(self, decision: RateDecision) -> None:
        super().__init__(
            "rate_limited",
            "Rate limit exceeded",
            429,
            {
                "limit": decision.limit,
                "remaining": decision.remaining,
                "reset_at_ms": decision.reset_at_ms,
                "reason": decision.reason,
            },
        )
class QuotaExceededError(PlatformError):
    def __init__(self, quota: QuotaDecision) -> None:
        super().__init__(
            "quota_exceeded",
            "Monthly token quota exceeded",
            402,
            {
                "monthly_quota": quota.monthly_quota,
                "used_before": quota.used_before,
                "used_after": quota.used_after,
                "remaining": quota.remaining,
            },
        )
class SafetyViolationError(PlatformError):
    def __init__(self, reason: str) -> None:
        super().__init__("safety_blocked", reason, 400)
class BillingWebhookError(PlatformError):
    def __init__(self, reason: str) -> None:
        super().__init__("billing_webhook_error", reason, 400)
def now_ms() -> int:
    return int(time.time() * 1000)
def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den
def percentile(values: Sequence[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(min(max(round((len(ordered) - 1) * q), 0), len(ordered) - 1))
    return ordered[idx]
def month_bucket(ts_ms: int) -> str:
    seconds = max(ts_ms // 1000, 0)
    tm = time.gmtime(seconds)
    return f"{tm.tm_year:04d}-{tm.tm_mon:02d}"
def month_window(ts_ms: int) -> tuple[int, int]:
    sec = max(ts_ms // 1000, 0)
    t = time.gmtime(sec)
    start_tuple = time.struct_time((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, 0))
    start_s = int(time.mktime(start_tuple))
    if t.tm_mon == 12:
        next_tuple = time.struct_time((t.tm_year + 1, 1, 1, 0, 0, 0, 0, 0, 0))
    else:
        next_tuple = time.struct_time((t.tm_year, t.tm_mon + 1, 1, 0, 0, 0, 0, 0, 0))
    end_s = int(time.mktime(next_tuple)) - 1
    return (start_s * 1000, end_s * 1000)
def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
def normalize_query(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    return compact
def estimate_tokens(text: str) -> int:
    words = re.findall(r"[A-Za-z0-9_\-]{2,}", text)
    return max(1, int(len(words) * 1.33))
def detect_prompt_injection(query: str) -> bool:
    lower = query.lower()
    suspicious = [
        "ignore previous instructions",
        "reveal system prompt",
        "bypass policy",
        "show hidden policy",
        "dump secrets",
    ]
    return any(fragment in lower for fragment in suspicious)
def cents_for_tokens(tokens: int, price_per_1k_cents: int) -> int:
    if tokens <= 0 or price_per_1k_cents <= 0:
        return 0
    return int((tokens / 1000.0) * price_per_1k_cents)
def clamp_top_k(requested: int | None, policy: PlanPolicy) -> int:
    if requested is None:
        return policy.default_top_k
    return min(max(requested, 1), policy.max_top_k)
def parse_signature_header(header_value: str) -> dict[str, str]:
    parts = [part.strip() for part in header_value.split(",") if part.strip()]
    parsed: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key] = value
    return parsed
def verify_webhook_signature(secret: str, raw_body: str, signature_header: str) -> bool:
    parsed = parse_signature_header(signature_header)
    timestamp = parsed.get("t")
    candidate = parsed.get("v1")
    if not timestamp or not candidate:
        return False
    signed_payload = f"{timestamp}.{raw_body}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, candidate)
def feature_for_endpoint(endpoint: str) -> str:
    mapping = {
        "/v1/query": "rag_query",
        "/v1/admin/dashboard": "admin_dashboard",
        "/v1/admin/keys/rotate": "api_key_rotation",
    }
    return mapping.get(endpoint, "unknown")
class TenantSaaSPlatform:
    def __init__(self, deps: PlatformDeps) -> None:
        self.deps = deps
    def authorize(self, token: str, request_id: str) -> AuthContext:
        claims = self.deps.tokens.verify(token)
        now = self.deps.clock.now_ms()
        if claims.exp_ms < now:
            raise UnauthorizedError("Token expired")
        user = self.deps.users.get(claims.sub)
        if user is None or not user.is_active:
            raise UnauthorizedError("User not found or inactive")
        org = self.deps.orgs.get(claims.org_id)
        if org is None:
            raise UnauthorizedError("Organization not found")
        if user.org_id != org.org_id:
            raise ForbiddenError("User does not belong to organization")
        return AuthContext(
            request_id=request_id,
            user=user,
            org=org,
            scopes=claims.scopes,
            token_nonce=claims.nonce,
        )
    def assert_org_state(self, org: OrgRecord) -> None:
        if org.state in {OrgState.SUSPENDED, OrgState.DELINQUENT}:
            raise SuspendedOrgError(org.org_id, org.state)
    def assert_incident_policy(
        self,
        *,
        ctx: AuthContext,
        endpoint: str,
        incident: IncidentPolicy | None,
    ) -> None:
        if incident is None or not incident.enabled:
            return
        blocked_org = ctx.org.org_id in incident.blocked_org_ids
        blocked_endpoint = endpoint in incident.blocked_endpoints
        is_admin = ctx.user.role in {Role.ADMIN, Role.OWNER}
        if blocked_org or blocked_endpoint:
            if incident.allow_admin_override and is_admin:
                self.deps.events.write(
                    AdminEvent(
                        event_id=self.deps.ids.new_id("evt"),
                        org_id=ctx.org.org_id,
                        actor_user_id=ctx.user.user_id,
                        severity=EventSeverity.WARN,
                        event_type="incident.override",
                        payload={
                            "incident_id": incident.incident_id,
                            "endpoint": endpoint,
                            "blocked_org": blocked_org,
                            "blocked_endpoint": blocked_endpoint,
                        },
                        created_at_ms=self.deps.clock.now_ms(),
                    )
                )
                return
            raise ForbiddenError("Incident policy blocks this request")
    def assert_feature_gate(self, ctx: AuthContext, endpoint: str, policy: PlanPolicy) -> None:
        feature = feature_for_endpoint(endpoint)
        if feature == "unknown":
            raise ForbiddenError("Endpoint not available")
        if feature not in policy.allowed_features:
            raise ForbiddenError("Feature is not enabled for current plan")
    def enforce_rate_limit(self, ctx: AuthContext, endpoint: str, policy: PlanPolicy) -> RateDecision:
        decision = self.deps.rate_limits.consume(
            org_id=ctx.org.org_id,
            endpoint=endpoint,
            limit=policy.requests_per_minute,
            burst=policy.burst_capacity,
            now_ms=self.deps.clock.now_ms(),
        )
        if not decision.allowed:
            raise RateLimitError(decision)
        return decision
    def assert_safety(self, query: str, org: OrgRecord) -> None:
        if not query.strip():
            raise SafetyViolationError("Query cannot be empty")
        if len(query) > 5000:
            raise SafetyViolationError("Query length exceeds safety limit")
        if "query_block" in org.safety_switches:
            raise SafetyViolationError("Organization query channel is temporarily disabled")
        if detect_prompt_injection(query):
            raise SafetyViolationError("Query blocked by prompt-injection safeguards")
    def compute_quota_decision(self, org: OrgRecord, policy: PlanPolicy, token_delta: int, now_ts_ms: int) -> QuotaDecision:
        bucket = month_bucket(now_ts_ms)
        used_before = self.deps.usage.monthly_tokens(org.org_id, bucket)
        used_after = used_before + token_delta
        remaining = max(policy.monthly_token_quota - used_after, 0)
        allowed = used_after <= policy.monthly_token_quota
        reason = "within_quota" if allowed else "quota_exceeded"
        if not allowed and policy.allow_overage:
            allowed = True
            reason = "overage_allowed"
        return QuotaDecision(
            allowed=allowed,
            monthly_quota=policy.monthly_token_quota,
            used_before=used_before,
            used_after=used_after,
            remaining=remaining,
            reason=reason,
        )
    def compute_billable_cents(
        self,
        *,
        policy: PlanPolicy,
        input_tokens: int,
        output_tokens: int,
        quota: QuotaDecision,
    ) -> tuple[int, list[BillingLine]]:
        input_cents = cents_for_tokens(input_tokens, policy.price_per_1k_input_tokens_cents)
        output_cents = cents_for_tokens(output_tokens, policy.price_per_1k_output_tokens_cents)
        lines: list[BillingLine] = [
            BillingLine(
                line_id=self.deps.ids.new_id("line"),
                charge_type=ChargeType.TOKEN_INPUT,
                description="Input tokens",
                quantity=input_tokens,
                unit_price_cents=policy.price_per_1k_input_tokens_cents,
                amount_cents=input_cents,
                metadata={"unit": "tokens"},
            ),
            BillingLine(
                line_id=self.deps.ids.new_id("line"),
                charge_type=ChargeType.TOKEN_OUTPUT,
                description="Output tokens",
                quantity=output_tokens,
                unit_price_cents=policy.price_per_1k_output_tokens_cents,
                amount_cents=output_cents,
                metadata={"unit": "tokens"},
            ),
        ]
        total = input_cents + output_cents
        if quota.reason == "overage_allowed" and quota.used_after > quota.monthly_quota:
            overage_tokens = quota.used_after - quota.monthly_quota
            overage_cents = cents_for_tokens(overage_tokens, policy.overage_per_1k_tokens_cents)
            total += overage_cents
            lines.append(
                BillingLine(
                    line_id=self.deps.ids.new_id("line"),
                    charge_type=ChargeType.OVERAGE,
                    description="Quota overage",
                    quantity=overage_tokens,
                    unit_price_cents=policy.overage_per_1k_tokens_cents,
                    amount_cents=overage_cents,
                    metadata={"unit": "tokens"},
                )
            )
        return total, lines
    def record_usage_and_billing(
        self,
        *,
        ctx: AuthContext,
        env: RequestEnvelope,
        answer: RAGAnswer,
        latency_ms: int,
        policy: PlanPolicy,
    ) -> tuple[QuotaDecision, int]:
        token_delta = answer.input_tokens + answer.output_tokens
        ts_now = self.deps.clock.now_ms()
        quota = self.compute_quota_decision(ctx.org, policy, token_delta, ts_now)
        if not quota.allowed:
            raise QuotaExceededError(quota)
        total_cents, lines = self.compute_billable_cents(
            policy=policy,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            quota=quota,
        )
        usage_event = UsageEvent(
            event_id=self.deps.ids.new_id("use"),
            org_id=ctx.org.org_id,
            user_id=ctx.user.user_id,
            endpoint=env.endpoint,
            mode=env.mode,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            billable_cents=total_cents,
            request_latency_ms=latency_ms,
            created_at_ms=ts_now,
        )
        self.deps.usage.append(usage_event)
        start_ms, end_ms = month_window(ts_now)
        invoice = self.deps.billing.open_invoice(
            org_id=ctx.org.org_id,
            period_start_ms=start_ms,
            period_end_ms=end_ms,
            issued_at_ms=ts_now,
        )
        for line in lines:
            self.deps.billing.add_line(invoice.invoice_id, line)
        return quota, total_cents
    def write_query_audit(
        self,
        *,
        ctx: AuthContext,
        env: RequestEnvelope,
        status_code: int,
        state: RequestState,
        latency_ms: int,
        message: str,
        details: dict[str, Any],
    ) -> None:
        self.deps.audits.write(
            AuditRecord(
                audit_id=self.deps.ids.new_id("aud"),
                kind=AuditKind.QUERY,
                request_id=env.request_id,
                org_id=ctx.org.org_id,
                user_id=ctx.user.user_id,
                endpoint=env.endpoint,
                status_code=status_code,
                state=state,
                latency_ms=latency_ms,
                message=message,
                details=details,
                created_at_ms=self.deps.clock.now_ms(),
            )
        )
    def write_admin_event(self, ctx: AuthContext, event_type: str, severity: EventSeverity, payload: dict[str, Any]) -> None:
        if ctx.user.role not in {Role.ADMIN, Role.OWNER}:
            return
        self.deps.events.write(
            AdminEvent(
                event_id=self.deps.ids.new_id("evt"),
                org_id=ctx.org.org_id,
                actor_user_id=ctx.user.user_id,
                severity=severity,
                event_type=event_type,
                payload=payload,
                created_at_ms=self.deps.clock.now_ms(),
            )
        )
    def process_team_query(
        self,
        env: RequestEnvelope,
        incident: IncidentPolicy | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        ctx: AuthContext | None = None
        try:
            if env.endpoint != "/v1/query":
                raise ForbiddenError("Unsupported endpoint")
            ctx = self.authorize(env.token, env.request_id)
            self.assert_org_state(ctx.org)
            policy = self.deps.plans.get(ctx.org.tier)
            self.assert_feature_gate(ctx, env.endpoint, policy)
            self.assert_incident_policy(ctx=ctx, endpoint=env.endpoint, incident=incident)
            normalized_query = normalize_query(env.query)
            self.assert_safety(normalized_query, ctx.org)
            rate = self.enforce_rate_limit(ctx, env.endpoint, policy)
            top_k = clamp_top_k(env.top_k, policy)
            answer = self.deps.rag.answer(
                org_id=ctx.org.org_id,
                query=normalized_query,
                top_k=top_k,
                mode=env.mode,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            quota, billed_cents = self.record_usage_and_billing(
                ctx=ctx,
                env=env,
                answer=answer,
                latency_ms=latency_ms,
                policy=policy,
            )
            response = {
                "request_id": env.request_id,
                "org_id": ctx.org.org_id,
                "plan": ctx.org.tier.value,
                "answer": answer.answer,
                "citations": [
                    {
                        "doc_id": c.doc_id,
                        "title": c.title,
                        "uri": c.uri,
                        "score": round(c.score, 4),
                    }
                    for c in answer.citations
                ],
                "model": answer.model,
                "mode": env.mode.value,
                "top_k": top_k,
                "latency_ms": latency_ms,
                "retrieval_ms": answer.retrieval_ms,
                "generation_ms": answer.generation_ms,
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
                "billable_cents": billed_cents,
                "quota": {
                    "monthly_quota": quota.monthly_quota,
                    "used_before": quota.used_before,
                    "used_after": quota.used_after,
                    "remaining": quota.remaining,
                    "reason": quota.reason,
                },
                "rate": {
                    "limit": rate.limit,
                    "remaining": rate.remaining,
                    "reset_at_ms": rate.reset_at_ms,
                },
            }
            self.write_query_audit(
                ctx=ctx,
                env=env,
                status_code=200,
                state=RequestState.SUCCESS,
                latency_ms=latency_ms,
                message="query_completed",
                details={
                    "model": answer.model,
                    "top_k": top_k,
                    "billable_cents": billed_cents,
                },
            )
            self.write_admin_event(
                ctx,
                event_type="admin.query.executed",
                severity=EventSeverity.INFO,
                payload={
                    "request_id": env.request_id,
                    "endpoint": env.endpoint,
                    "latency_ms": latency_ms,
                    "billable_cents": billed_cents,
                },
            )
            return response
        except PlatformError as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            if ctx is not None:
                state = RequestState.DENIED if exc.status_code < 500 else RequestState.FAILED
                self.write_query_audit(
                    ctx=ctx,
                    env=env,
                    status_code=exc.status_code,
                    state=state,
                    latency_ms=latency_ms,
                    message=exc.code,
                    details=exc.details,
                )
                if exc.status_code >= 500:
                    self.write_admin_event(
                        ctx,
                        event_type="query.internal_error",
                        severity=EventSeverity.CRITICAL,
                        payload={
                            "request_id": env.request_id,
                            "code": exc.code,
                        },
                    )
            raise
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - started) * 1000)
            if ctx is not None:
                self.write_query_audit(
                    ctx=ctx,
                    env=env,
                    status_code=500,
                    state=RequestState.FAILED,
                    latency_ms=latency_ms,
                    message="internal_error",
                    details={"error": str(exc)},
                )
                self.write_admin_event(
                    ctx,
                    event_type="query.unhandled_exception",
                    severity=EventSeverity.CRITICAL,
                    payload={"request_id": env.request_id, "error": str(exc)},
                )
            raise PlatformError("internal_error", "Unexpected platform error", 500)
    def rotate_api_key(self, actor_token: str, org_id: str, label: str) -> dict[str, Any]:
        ctx = self.authorize(actor_token, request_id=self.deps.ids.new_id("req"))
        if ctx.user.role not in {Role.ADMIN, Role.OWNER}:
            raise ForbiddenError("Only admin/owner can rotate API keys")
        if ctx.org.org_id != org_id:
            raise ForbiddenError("Cross-org key rotation is not allowed")
        key = self.deps.api_keys.rotate(
            org_id=org_id,
            label=label,
            rotated_by=ctx.user.user_id,
            created_at_ms=self.deps.clock.now_ms(),
        )
        self.deps.audits.write(
            AuditRecord(
                audit_id=self.deps.ids.new_id("aud"),
                kind=AuditKind.ADMIN,
                request_id=self.deps.ids.new_id("req"),
                org_id=ctx.org.org_id,
                user_id=ctx.user.user_id,
                endpoint="/v1/admin/keys/rotate",
                status_code=200,
                state=RequestState.SUCCESS,
                latency_ms=0,
                message="api_key_rotated",
                details={"label": label},
                created_at_ms=self.deps.clock.now_ms(),
            )
        )
        self.write_admin_event(
            ctx,
            event_type="admin.api_key.rotated",
            severity=EventSeverity.WARN,
            payload={"label": label, "org_id": org_id},
        )
        return {"org_id": org_id, "label": label, "api_key": key}
    def suspend_org(self, actor_token: str, target_org_id: str, reason: str) -> dict[str, Any]:
        ctx = self.authorize(actor_token, request_id=self.deps.ids.new_id("req"))
        if ctx.user.role != Role.OWNER:
            raise ForbiddenError("Only owners can suspend organizations")
        org = self.deps.orgs.get(target_org_id)
        if org is None:
            raise ForbiddenError("Target organization not found")
        org.state = OrgState.SUSPENDED
        org.metadata["suspended_reason"] = reason
        self.deps.orgs.save(org)
        self.write_admin_event(
            ctx,
            event_type="admin.org.suspended",
            severity=EventSeverity.CRITICAL,
            payload={"target_org_id": target_org_id, "reason": reason},
        )
        self.deps.audits.write(
            AuditRecord(
                audit_id=self.deps.ids.new_id("aud"),
                kind=AuditKind.ADMIN,
                request_id=self.deps.ids.new_id("req"),
                org_id=ctx.org.org_id,
                user_id=ctx.user.user_id,
                endpoint="/v1/admin/org/suspend",
                status_code=200,
                state=RequestState.SUCCESS,
                latency_ms=0,
                message="org_suspended",
                details={"target_org_id": target_org_id, "reason": reason},
                created_at_ms=self.deps.clock.now_ms(),
            )
        )
        return {"target_org_id": target_org_id, "state": org.state.value, "reason": reason}
    def process_billing_webhook(self, headers: dict[str, str], raw_body: str) -> dict[str, Any]:
        sig = headers.get("stripe-signature", "")
        secret = self.deps.secrets.get("stripe_webhook_secret")
        if not secret:
            raise BillingWebhookError("Webhook secret is not configured")
        if not verify_webhook_signature(secret=secret, raw_body=raw_body, signature_header=sig):
            raise BillingWebhookError("Invalid webhook signature")
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:  # noqa: PERF203
            raise BillingWebhookError("Webhook payload is not valid JSON") from exc
        event = WebhookEvent(
            provider="stripe",
            event_id=str(payload.get("id", self.deps.ids.new_id("evt"))),
            event_type=str(payload.get("type", "unknown")),
            org_id=str(payload.get("data", {}).get("metadata", {}).get("org_id", "")),
            amount_cents=int(payload.get("data", {}).get("amount", 0)),
            currency=str(payload.get("data", {}).get("currency", "usd")),
            payload=payload,
            published_at_ms=self.deps.clock.now_ms(),
        )
        if not event.org_id:
            raise BillingWebhookError("Missing org_id metadata in webhook payload")
        invoice = self.deps.billing.latest_for_org(event.org_id)
        if invoice and event.event_type in {"invoice.paid", "charge.succeeded"}:
            self.deps.billing.mark_paid(invoice.invoice_id, payment_ref=event.event_id)
        org = self.deps.orgs.get(event.org_id)
        if org and event.event_type in {"invoice.paid", "charge.succeeded"}:
            org.state = OrgState.ACTIVE
            self.deps.orgs.save(org)
        self.deps.events.write(
            AdminEvent(
                event_id=self.deps.ids.new_id("evt"),
                org_id=event.org_id,
                actor_user_id="system",
                severity=EventSeverity.INFO,
                event_type="billing.webhook.processed",
                payload={
                    "provider_event_id": event.event_id,
                    "provider_event_type": event.event_type,
                    "amount_cents": event.amount_cents,
                    "currency": event.currency,
                },
                created_at_ms=self.deps.clock.now_ms(),
            )
        )
        return {
            "accepted": True,
            "event_id": event.event_id,
            "event_type": event.event_type,
            "org_id": event.org_id,
        }
    def build_admin_dashboard_snapshot(self, actor_token: str, org_id: str, lookback_hours: int = 24) -> DashboardSnapshot:
        ctx = self.authorize(actor_token, request_id=self.deps.ids.new_id("req"))
        if ctx.user.role not in {Role.ADMIN, Role.OWNER, Role.SUPPORT}:
            raise ForbiddenError("Dashboard access denied")
        if ctx.org.org_id != org_id:
            raise ForbiddenError("Cross-org dashboard access denied")
        end_ms = self.deps.clock.now_ms()
        start_ms = end_ms - lookback_hours * 60 * 60 * 1000
        audits = self.deps.audits.list(org_id, start_ms, end_ms)
        usage = self.deps.usage.list(org_id, start_ms, end_ms)
        total = len(audits)
        success_rate = fmean(1.0 if item.state == RequestState.SUCCESS else 0.0 for item in audits) if audits else 0.0
        latencies = [item.latency_ms for item in audits]
        rate_limit_hits = sum(1 for item in audits if item.message == "rate_limited")
        safety_blocks = sum(1 for item in audits if item.message == "safety_blocked")
        return DashboardSnapshot(
            org_id=ctx.org.org_id,
            plan=ctx.org.tier,
            state=ctx.org.state,
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            total_requests=total,
            success_rate=round(success_rate, 4),
            avg_latency_ms=int(fmean(latencies)) if latencies else 0,
            p95_latency_ms=percentile(latencies, 0.95),
            token_usage=sum(item.input_tokens + item.output_tokens for item in usage),
            billable_cents=sum(item.billable_cents for item in usage),
            rate_limit_hits=rate_limit_hits,
            safety_blocks=safety_blocks,
        )
