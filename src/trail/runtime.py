from html import escape

from trail.budget import rebalance_node_sliders
from trail.models import NodeSlider, ResolvedNodeRuntime, ResolvedSliderInstruction
from trail.semantics import Actor, SemanticTemplateService, TemplateContext


def resolve_node_runtime(
    *,
    node_id: str,
    sliders: list[NodeSlider],
    templates: SemanticTemplateService,
    context: TemplateContext,
    actor: Actor | None = None,
    preset_ids: dict[str, str] | None = None,
) -> ResolvedNodeRuntime:
    """Resolve final slider values to concrete, versioned AI instructions."""

    selected_presets = preset_ids or {}
    resolved: list[ResolvedSliderInstruction] = []
    for slider in sliders:
        template = templates.resolve(
            slider.semantic_template_key,
            context,
            actor=actor,
            preset_id=selected_presets.get(slider.id),
        )
        resolved.append(
            ResolvedSliderInstruction(
                slider_id=slider.id,
                slider_name=slider.name,
                final_value=slider.value,
                instruction=template.instruction_for(slider.value),
                template_id=template.id,
                template_version=template.version,
            )
        )
    return ResolvedNodeRuntime(node_id=node_id, sliders=resolved)


def process_node_slider_change(
    *,
    node_id: str,
    sliders: list[NodeSlider],
    active_slider_id: str,
    requested_value: float,
    templates: SemanticTemplateService,
    context: TemplateContext,
    actor: Actor | None = None,
    preset_ids: dict[str, str] | None = None,
    positive_budget: float = 5.0,
    negative_budget: float = 5.0,
    alpha: float = 0.3,
    deadzone: bool = False,
) -> tuple[list[NodeSlider], ResolvedNodeRuntime]:
    """Apply one interaction, then resolve only the stable post-budget values."""

    balanced = rebalance_node_sliders(
        sliders,
        active_slider_id=active_slider_id,
        requested_value=requested_value,
        positive_budget=positive_budget,
        negative_budget=negative_budget,
        alpha=alpha,
        deadzone=deadzone,
    )
    runtime = resolve_node_runtime(
        node_id=node_id,
        sliders=balanced,
        templates=templates,
        context=context,
        actor=actor,
        preset_ids=preset_ids,
    )
    return balanced, runtime


def format_runtime_instructions(runtime: ResolvedNodeRuntime | None) -> str:
    """Create a bounded prompt block from resolved semantic instructions."""

    if runtime is None or not runtime.sliders:
        return "No trail slider instructions are active."
    lines = [f'<trail_semantic_instructions node_id="{escape(runtime.node_id, quote=True)}">']
    lines.extend(
        f"- {escape(item.slider_name)} ({item.final_value:+.1f}): {escape(item.instruction)}"
        for item in runtime.sliders
    )
    lines.append("</trail_semantic_instructions>")
    return "\n".join(lines)
