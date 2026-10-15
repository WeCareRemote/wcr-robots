# Trail node sliders and semantic templates

This service owns the deterministic AI-facing part of trail slider processing. The implementation
lives in `src/trail` and follows the trail specification for node-local score sliders and semantic
templates.

## Processing contract

1. A node supplies any number of `NodeSlider` values.
2. `rebalance_node_sliders` snaps the active value to a `0.5` step in `-5.0..+5.0`.
3. Positive and negative values consume separate budgets of `5.0` each.
4. On overflow, `alpha=0.3` is assigned to the active slider's spring-back and the rest is removed
   proportionally from other same-direction sliders. Integer half-step units and deterministic
   largest-remainder distribution keep all invariants exact.
5. `resolve_node_runtime` resolves each final value through the inherited semantic template.
6. `process_node_slider_change` composes both steps so AI instructions can only be generated from
   post-budget values.
7. `format_runtime_instructions` creates the bounded system-instruction block consumed by the
   existing Ask AI graph through `UserInput.trail_runtime`.

The optional `deadzone` parameter maps `-0.5`, `0.0`, and `+0.5` to neutral when enabled. It is
disabled by default because the product specification leaves that behavior optional.

## Templates and permissions

`default_semantic_mapping` contains the exact 21 mappings from `-5.0` through `+5.0` in `0.5`
steps. `SemanticTemplateService` resolves layers in this order:

`global -> organization -> project -> user`

The most specific permitted layer wins. A locked layer or one with `allow_override=false` stops
normal inheritance below it. A Super Admin may create or replace central templates, inspect hidden
templates, and resolve through locked layers. User presets are private to their owner, are saved
through the repository, can be made the user's default, explicitly reused for several sliders of
the same semantic type, and reverted without deleting the preset.

`SemanticTemplateRepository` is the persistence boundary. The included in-memory adapter is for
tests and local composition. The product backend must provide the durable adapter after the Trail
backend owner and trusted organization/project/Super-Admin claim contract are fixed. This service
does not expose an Admin mutation endpoint or accept a client-provided Super-Admin flag.

## Integration assumptions

- `wcr-robots` owns prompt assembly and semantic runtime resolution; this is why the AI-facing
  domain core is implemented here.
- The current repositories do not yet contain the Trail node editor/navigation model. Slider UI,
  the 150-250 ms spring animation, and the optional budget meter therefore belong to the future
  Trail frontend integration, not this service.
- The authenticated upstream service is responsible for supplying the already resolved
  `trail_runtime` payload. Pydantic validation rejects off-grid, out-of-range, duplicate, or
  over-budget payloads before an agent runs.
- The production image explicitly copies `src/trail`; local source success alone is not treated as
  deployment readiness.

## Verification coverage

Tests cover two, three, and 25 dynamic sliders; repeated updates; positive, negative, mixed, and
neutral values; snapping, clamping, deadzone, and spring behavior; all mapping steps; every
inheritance layer; hidden/locked templates; Super-Admin rights; private reusable user presets;
runtime payload validation; post-spring instruction resolution; prompt insertion; and service
request validation.
