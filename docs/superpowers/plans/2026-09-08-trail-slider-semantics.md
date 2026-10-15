# Trail Node Slider and Semantic Template Implementation Plan

> Functional source: `Pasted text.txt` from the referenced ChatGPT conversation and the
> canonical trail architecture specification `Spec - Node Slider Budgeting und Semantic
> Templates.md`.

## Architecture decision

Implement the server-side domain logic in `wcr-robots`. The trail architecture assigns prompt
assembly and runtime variable resolution to this service. No trail node model, trusted
Super-Admin claim contract, or semantic-template persistence adapter currently exists here, so
this change adds one domain module and a repository protocol instead of inventing public admin
routes or a second database. The concrete product backend can implement the protocol once its
ownership and authorization contract are fixed.

Values are represented externally as one-decimal floats and internally as integer half-steps
(`-10..10`). This makes the `-5.0..+5.0` range, `0.5` step, and separate `5.0` budgets exact and
prevents floating-point drift during repeated rebalancing.

## Task 1: Budget behavior

**Files:**

- Create `src/trail/models.py`
- Create `src/trail/budget.py`
- Create `tests/trail/test_budget.py`

Write failing tests for two, three, and many sliders; independent positive/negative budgets;
neutral values; snapping/clamping; proportional spring behavior with `alpha=0.3`; and repeated
updates. Implement deterministic largest-remainder distribution in half-step units so every
result remains on-grid and each directional sum stays at or below ten half-steps.

## Task 2: Semantic templates, inheritance, and authorization

**Files:**

- Create `src/trail/semantics.py`
- Create `tests/trail/test_semantics.py`

Provide the complete default mapping for all 21 half-steps. Model global, organization, project,
and user scopes. Resolve the most-specific permitted template. Require a Super Admin actor to
create or change central templates and to view hidden templates; allow the same actor to override
locked templates. Let users clone permitted templates as named, reusable presets and choose a
default preset. Keep storage behind a repository protocol and provide an in-memory adapter for
unit tests and local composition.

## Task 3: Runtime prompt resolution

**Files:**

- Create `src/trail/runtime.py`
- Create `src/trail/__init__.py`
- Modify `src/agents/ask_ai_agent.py`
- Modify `docker/Dockerfile.service`
- Create `tests/trail/test_runtime.py`
- Modify `tests/agents/test_ask_ai_agent.py`

Resolve each final, budgeted slider value through the inherited semantic template into a payload
containing slider identity, final value, instruction, template ID, and version. Format those
payloads as a bounded system-instruction block. Teach the existing Ask AI prompt path to include
the block from trusted runtime configuration; do not accept an untrusted role flag from a public
request. Include the new package in the existing production service image.

## Task 4: Verification

Run the targeted trail and Ask AI tests, then the relevant unit suite, Ruff, and the repository's
available build/type checks. Record any pre-existing environment limitation separately from code
failures. The locked full environment currently cannot install `timescale-vector` on Python 3.13
because its transitive `psycopg2==2.9.10` build cannot find `pg_config`; use a minimal isolated
test environment for the new pure-domain tests if that remains unresolved.
