# Security Boundary and Disclosure Policy

## Purpose

Define what PCSA discloses publicly and what remains private to preserve user-facing security and sovereignty.

## Public-First Disclosure Principle

PCSA publishes philosophy, governance intent, and architectural invariants.
PCSA withholds implementation-sensitive details that would materially increase exploitability or privacy risk.

## What Is Public

- Principles and mandates (Universal Floor, Authentic Partnership)
- High-level architecture and trust zones
- Policy invariants and user guarantees
- Review and governance process
- Responsible disclosure channel and response policy

## What Is Restricted (Not Public)

- Exact policy thresholds and timing values
- Internal failure signatures and retry heuristics
- Internal anti-abuse and guardrail tuning
- Local persistence internals tied to sensitive state behavior
- Operational hardening details that materially reduce attack cost

## Rationale

User sovereignty requires both transparency and protection.
Publishing sensitive internals can lower attack effort against users even when code is local-first.
Disclosure decisions therefore prioritize user risk reduction over maximal technical exposure.

## Safe Documentation Rules

1. Publish invariants, not exploit-enabling mechanics.
2. Use abstract sequence descriptions instead of production-identical logic.
3. Describe data classes, not sensitive field-level internals.
4. Exclude details that could guide denial-of-service timing or consent bypass attempts.
5. Keep examples non-production and non-deployable.

## Review Gate for Public Docs

Every public architecture change must include:

- Security boundary impact statement
- Exploitability review (can this reduce attacker effort?)
- Privacy leakage review (does this reveal sensitive data behavior?)
- Mitigation decision (publish, redact, or move to restricted docs)

## Threat-Aware Redaction Triggers

Redact or move details to restricted docs when a change reveals:

- Precise control thresholds or cooldown windows
- Deterministic signatures for triggering fallback states
- Internal persistence mechanics tied to user state protections
- Concrete bypass strategies for trust/consent workflow

## Responsible Disclosure

Security vulnerabilities and privacy weaknesses must be reported privately.
Do not publish proof-of-concept exploit steps in public issues.

## Boundary Statement

PCSA is open about purpose, constraints, and guarantees.
PCSA is intentionally selective about implementation details to preserve the security and sovereignty of end users.
