from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_half_step_value(value: float) -> float:
    if not -5.0 <= value <= 5.0:
        raise ValueError("slider value must be between -5.0 and +5.0")
    if not (value * 2).is_integer():
        raise ValueError("slider value must use 0.5 steps")
    return float(value)


class NodeSlider(BaseModel):
    """One dynamically configured score slider belonging to a trail node."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    value: float = 0.0
    semantic_template_key: str = "priority"

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        return _validate_half_step_value(value)


class ResolvedSliderInstruction(BaseModel):
    """Auditable runtime payload passed from slider semantics to an AI prompt."""

    model_config = ConfigDict(frozen=True)

    slider_id: str
    slider_name: str
    final_value: float
    instruction: str
    template_id: str
    template_version: int

    @field_validator("final_value")
    @classmethod
    def validate_final_value(cls, value: float) -> float:
        return _validate_half_step_value(value)


class ResolvedNodeRuntime(BaseModel):
    """Resolved semantic instructions for every slider in one node."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    sliders: list[ResolvedSliderInstruction]

    @model_validator(mode="after")
    def validate_budget_invariants(self) -> "ResolvedNodeRuntime":
        slider_ids = [slider.slider_id for slider in self.sliders]
        if len(slider_ids) != len(set(slider_ids)):
            raise ValueError("slider IDs must be unique within a node")
        positive_total = sum(
            slider.final_value for slider in self.sliders if slider.final_value > 0
        )
        negative_total = sum(
            -slider.final_value for slider in self.sliders if slider.final_value < 0
        )
        if positive_total > 5.0:
            raise ValueError("positive node slider budget exceeds 5.0")
        if negative_total > 5.0:
            raise ValueError("negative node slider budget exceeds 5.0")
        return self
