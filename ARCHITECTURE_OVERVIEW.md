# PCSA Architecture Overview (Public)

## Scope

This document describes high-level architecture, trust boundaries, and behavior invariants.
It intentionally omits implementation-sensitive details.

## Architectural Objectives

- Maximize local execution within user constraints
- Preserve explicit user agency during delegation decisions
- Sustain utility under degraded conditions
- Minimize setup burden for constrained users

## Trust Zones

1. User-Controlled Local Zone
- Local runtime
- Local indexing and retrieval
- Local persistence and policy state

2. Optional Delegated Compute Zone
- External compute used only through policy-governed escalation
- Never required for baseline operation

3. Trust Gateway
- Policy decision layer between local and optional external execution
- Enforces consent and trust preferences

## Public Decision Flow

1. Attempt local retrieval and local inference.
2. Evaluate local sufficiency against policy criteria.
3. If sufficient, return local output.
4. If insufficient, generate an escalation proposal.
5. Require explicit confirmation unless persistent trust override is enabled.
6. If unavailable or unstable locally, provide degraded deterministic utility.

## Core Design Patterns

- Constraint-first setup optimization
- Policy-gated delegation
- Explicit consent intercept
- Persistent trust preference
- Cooldown-based degraded mode
- Incremental indexing for low startup overhead

## Resilience Model

The architecture treats outages and latency spikes as expected operating conditions.

Public guarantees:

- Interaction remains useful in degraded mode.
- Startup avoids heavy work by default.
- Health checks use bounded, non-blocking behavior.

## Data and Privacy Model (Public)

- Local data remains in user-controlled storage by default.
- Delegation proposals disclose metadata classes at decision time.
- External delegation is not automatic under default trust settings.

For publication limits and protected details, see SECURITY_BOUNDARY.md.

## Documentation Boundaries

This public document does not include:

- Internal threshold values
- Internal anti-abuse tuning
- Storage layout details that could aid targeting
- Environment-specific hardening internals

## Related Documents

- MANIFESTO.md
- SECURITY_BOUNDARY.md
- docs/trust_gateway_policy.md
