# Iljard Xhani
AI Application Engineer (LLM + RAG + Reliability)

Email: xhani.iljard@gmail.com

## Summary
AI Application Engineer focused on shipping production-ready LLM features with measurable quality gates. Builds end-to-end systems across retrieval, orchestration, evaluation, and operational reliability. Portfolio includes six detailed case studies covering RAG, multi-tenant SaaS controls, and workflow automation.

## Core Skills
- LLM application engineering: prompt contracts, tool/function calling, deterministic API outputs
- RAG systems: ingestion pipelines, chunking strategy, hybrid retrieval, reranking, citation grounding
- Evaluation and release safety: faithfulness checks, regression gates, offline and CI evaluation workflows
- Platform reliability: observability, rate limits, fallback paths, audit logging, incident controls
- Product backend: FastAPI, Python, Pydantic, Postgres, pgvector, Docker, n8n

## Selected Projects
### RAG Evaluation Lab (Release Gate System)
- Built policy-driven evaluation orchestration to block low-quality releases before deployment.
- Implemented explicit gate thresholds: max quality regression `-0.02`, min pass rate `0.90`, min faithfulness `0.88`, max p95 latency `1900ms`.
- Added PR smoke checks plus nightly benchmark runs for faster feedback and long-horizon quality tracking.

### DocChat RAG (Production)
- Built ingestion-to-answer pipeline for PDF/docs with hybrid retrieval and citation-constrained responses.
- Added runtime guardrails, fallback lanes, and retrieval confidence checks to reduce unsupported answers.
- Instrumented evaluation and runtime signals including faithfulness, fallback rate, and p95 latency tracking.

### Mini SaaS: RAG for Teams
- Built multi-tenant AI product layer around RAG: auth context, org isolation, plan entitlements, quota and rate controls.
- Implemented usage metering, billing flow hooks, and structured audit events for governance.
- Validated platform readiness with tenant isolation checks and burst-traffic rate-limit tests.

### Customer Support AI Triage (n8n + FastAPI)
- Designed event-driven triage workflow for email/DM: classification, routing, AI draft, ticket creation, approval queue.
- Enforced human-in-the-loop approvals for outbound responses and maintained event-level auditability.
- Added idempotent retries, dead-letter handling, and low-confidence fallback templates for safer operations.

### SQL + Docs Hybrid Agent
- Built intent router to choose SQL for structured facts and RAG for policy/process context.
- Added read-only SQL safety envelope, route precision checks, and evidence-linked synthesis.

### Knowledge Base Builder
- Implemented versioned ingestion control plane for PDF/URL/markdown normalization, change detection, and incremental reindexing.
- Reduced unnecessary reprocessing by fingerprinting sources and indexing only changed content.

## Professional Focus
- Open to AI Application Engineer roles building stable, measurable, and production-oriented AI features.
