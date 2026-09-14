# EndoMind

**A production system for automated, clinically-validated patient education —
built with a real physician, for real patients on a waitlist.**

EndoMind ingests medical textbooks (starting with Greenspan's *Basic & Clinical
Endocrinology*), generates personalized educational email sequences using a
RAG pipeline, and validates every piece of content against the source
material before it reaches a patient — combining automated evaluation
(RAGAS, LLM-as-judge) with sign-off from a licensed endocrinologist.

Unlike typical "chatbot" portfolio projects, EndoMind has a real user, a real
data pipeline with genuine business entities (patients, conditions, content
sequences, deliveries), and a reliability layer designed to catch
hallucinations before they matter.

## Current phase — Type 2 Diabetes (MVP)

- Corpus ingestion from clinical textbooks (chunked by chapter/section, not
  fixed-size windows)
- PostgreSQL + pgvector for hybrid relational/vector storage
- RAG-based generation of day 1 / day 7 / day 30 educational email sequences
- Data quality validation (Great Expectations) + content quality validation
  (LLM-as-judge against a physician-approved golden dataset)
- Airflow-orchestrated delivery via Resend, with retry/alerting
- Patient waitlist intake with explicit consent
- Physician approval panel before any new sequence goes live
- Deployed on AWS (S3, RDS, EC2), observability dashboard for hallucination
  rate, cost, latency, and engagement

## Phase 2 — Expansion

- Additional conditions from the same physician-provided corpus (thyroid,
  hypertension, etc.)
- Patient-facing web portal (registration, content history)
- Formal legal review for Colombia's Ley 1581 as the system scales toward
  personalized clinical data
- Migration to Amazon SES at scale

## Phase 3 — Content distribution

- Automated repurposing of validated educational content into short-form
  social media posts (Instagram/TikTok/LinkedIn) for the physician's own
  patient acquisition channel, with the same clinical-accuracy validation
  layer gating anything that goes out publicly

## Why this exists

Built to demonstrate production AI engineering: retrieval, evaluation
pipelines, resilient orchestration, and honest handling of failure — not just
another RAG demo.
