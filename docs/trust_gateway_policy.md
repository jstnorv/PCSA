# Trust Gateway Policy Specification

## Purpose

Define a two-phase local-first policy for PCSA:

1. Setup Policy: maximize local capability within user limitations.
2. Runtime Policy: execute local-first and escalate only when local execution is insufficient.

This specification is normative for delegated computation behavior.

## Core Principles

- Local by default: all computation starts on local resources.
- User-constrained optimization: local configuration must respect declared limitations.
- Cloud as exception path: external delegation is allowed only after insufficiency is established.
- Trust-preserving escalation: user confirmation is required before external execution unless a persistent trust override is enabled.

## Policy Entities

- User Limitations: explicit constraints provided by user profile.
- Local Capability Profile: measured local resources and available local models/tools.
- Trust Preference: persisted setting controlling intercept behavior.
- Escalation Proposal: user-facing security intercept payload for external execution.

## User Limitations Schema

The policy assumes a logical profile with these dimensions:

- Privacy Sensitivity: low, medium, high, strict-local
- Latency Tolerance: max acceptable response time
- Cost Ceiling: allowed external spend or no-cloud
- Compute Budget: allowed local CPU, RAM, and optional GPU usage
- Reliability Preference: priority for deterministic local completion over higher-capability delegation

## Phase 1: Setup Policy (Local Maximization)

### Objective

Select and tune local resources such that expected workload is handled locally under user limitations.

### Inputs

- User Limitations profile
- Detected hardware capabilities
- Available local model set and embedding options
- Local storage limits and index parameters

### Required Setup Actions

1. Baseline capability detection
- Detect CPU, RAM, optional GPU, available local models, and local storage paths.

2. Constraint validation
- Enforce hard user constraints before selecting runtime defaults.

3. Local configuration selection
- Choose local chat model and embedding model compatible with constraints.
- Set retrieval and memory defaults to maximize local quality within latency and memory bounds.
- Set confidence and complexity thresholds for runtime escalation.

4. Safety defaults
- Default trust preference must be intercept-required.
- External delegation remains disabled unless explicitly approved at runtime.

### Setup Decision Rule

Given configuration candidates C, choose:

argmax local_quality(c)

subject to:

- local_latency(c) <= user_latency_limit
- local_memory(c) <= user_memory_limit
- privacy_risk(c) <= user_privacy_limit
- external_cost(c) <= user_cost_limit

If no candidate satisfies constraints, setup must keep local-safe defaults and mark escalation as high-likelihood at runtime.

## Phase 2: Runtime Policy (Local-First Execution)

### Runtime Sequence

1. Execute local retrieval and local inference.
2. Estimate local confidence score and task complexity score.
3. Determine sufficiency against thresholds.
4. If sufficient: return local response.
5. If insufficient: generate escalation proposal.
6. Apply trust preference:
- If intercept-required: request explicit confirmation.
- If always-trust-cloud: proceed to external delegation.
7. Persist outcome and trust updates.

### Degraded Mode Requirement

When local inference or retrieval services are unavailable, timing out, or intermittently failing:

- Runtime must enter degraded mode.
- Degraded mode must return deterministic local utility output instead of hard failure.
- Runtime should apply cooldown-based retry behavior to avoid repeated blocking latency spikes.

### Runtime Sufficiency Rule

Insufficient local execution is true when any condition holds:

- local_confidence < confidence_threshold
- task_complexity >= complexity_threshold
- estimated_local_resource_need > user_compute_budget
- requested_operation violates local capability boundary

### Security Intercept Requirements

Before any external call, Escalation Proposal must include:

- Reason local execution is insufficient
- Metadata-only sharing summary
- Explicit user confirmation request

The intercept must not execute external computation before confirmation when intercept-required is active.

## Trust Preference Persistence

Trust preference is a persisted state field with at least:

- intercept_required
- always_trust_cloud

Rules:

- Default value: intercept_required
- User may toggle always_trust_cloud on or off
- Toggle persists across future requests and sessions

## Compliance Invariants

The controller is compliant only if all invariants hold:

- Every request attempts local route first.
- No external execution occurs without insufficiency determination.
- No external execution occurs without intercept confirmation when intercept_required is active.
- Trust override is explicit, persisted, and user-controlled.
- Service outages do not cause complete interaction failure; degraded local output remains available.
- Startup defaults avoid heavy indexing unless explicitly requested.

## Mapping to Current Implementation

- Delegation controller and escalation logic: agent/delegated_controller.py
- Runtime execution and controlled turn flow: agent/execution_loop.py
- Trust preference state and persistence fields: agent/session_state.py
- Runtime intercept and trust commands: main.py
- Persistent state location: config.py
- Setup optimization and capability detection: agent/setup_optimizer.py
- Degraded local fallback and cooldown handling: agent/execution_loop.py
- Incremental indexing and startup ingestion behavior: core/pipeline.py and main.py
- Local readiness diagnostics: main.py (`doctor` command)

## Future Extensions

- Add setup calibration routine to benchmark local model options on representative tasks.
- Add dynamic threshold tuning from accepted or rejected escalation outcomes.
- Add data-classification policy to redact sensitive metadata before proposal rendering.