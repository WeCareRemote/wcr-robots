from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal

from trail.models import NodeSlider

MIN_UNITS = -10
MAX_UNITS = 10


def _to_units(value: float) -> int:
    snapped = int((Decimal(str(value)) * 2).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return min(MAX_UNITS, max(MIN_UNITS, snapped))


def _allocate_proportionally(total: int, capacities: dict[int, int]) -> dict[int, int]:
    """Allocate exact integer reductions using stable largest remainders."""

    allocation = {index: 0 for index in capacities}
    remaining = min(total, sum(capacities.values()))
    active_capacities = dict(capacities)

    while remaining and active_capacities:
        capacity_sum = sum(active_capacities.values())
        exact = {
            index: Decimal(remaining) * Decimal(capacity) / Decimal(capacity_sum)
            for index, capacity in active_capacities.items()
        }
        floors = {
            index: min(
                active_capacities[index],
                int(share.quantize(Decimal(1), rounding=ROUND_FLOOR)),
            )
            for index, share in exact.items()
        }
        used = sum(floors.values())
        for index, amount in floors.items():
            allocation[index] += amount
            active_capacities[index] -= amount
        remaining -= used

        if not remaining:
            break

        ranked = sorted(
            active_capacities,
            key=lambda index: (-(exact[index] - floors[index]), index),
        )
        progressed = False
        for index in ranked:
            if remaining == 0:
                break
            if active_capacities[index] > 0:
                allocation[index] += 1
                active_capacities[index] -= 1
                remaining -= 1
                progressed = True
        active_capacities = {
            index: capacity for index, capacity in active_capacities.items() if capacity > 0
        }
        if not progressed:
            break

    return allocation


def _enforce_direction_budget(
    units: list[int],
    *,
    direction: int,
    budget_units: int,
    active_index: int,
    alpha: Decimal,
) -> None:
    indices = [index for index, value in enumerate(units) if value * direction > 0]
    magnitudes = {index: abs(units[index]) for index in indices}
    overflow = sum(magnitudes.values()) - budget_units
    if overflow <= 0:
        return

    reductions = {index: 0 for index in indices}
    active_is_in_direction = active_index in magnitudes
    if active_is_in_direction:
        requested_active = int(
            (Decimal(overflow) * alpha).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        )
        active_reduction = min(magnitudes[active_index], requested_active)
        reductions[active_index] = active_reduction

        other_capacities = {
            index: magnitude for index, magnitude in magnitudes.items() if index != active_index
        }
        other_reductions = _allocate_proportionally(
            overflow - active_reduction,
            other_capacities,
        )
        reductions.update(other_reductions)

        still_needed = overflow - sum(reductions.values())
        if still_needed:
            active_capacity = magnitudes[active_index] - reductions[active_index]
            extra_active = min(still_needed, active_capacity)
            reductions[active_index] += extra_active
            still_needed -= extra_active
        if still_needed:
            residual_capacities = {
                index: magnitudes[index] - reductions[index]
                for index in indices
                if index != active_index
            }
            extra = _allocate_proportionally(still_needed, residual_capacities)
            for index, amount in extra.items():
                reductions[index] += amount
    else:
        reductions.update(_allocate_proportionally(overflow, magnitudes))

    for index, reduction in reductions.items():
        units[index] = direction * (magnitudes[index] - reduction)


def rebalance_node_sliders(
    sliders: list[NodeSlider],
    *,
    active_slider_id: str,
    requested_value: float,
    positive_budget: float = 5.0,
    negative_budget: float = 5.0,
    alpha: float = 0.3,
    deadzone: bool = False,
) -> list[NodeSlider]:
    """Set one slider and softly rebalance overflowing same-direction values.

    All calculations use integer half-step units. This keeps the range, step, and both
    directional budget invariants stable across repeated updates.
    """

    ids = [slider.id for slider in sliders]
    if len(ids) != len(set(ids)):
        raise ValueError("slider IDs must be unique within a node")
    try:
        active_index = ids.index(active_slider_id)
    except ValueError as error:
        raise ValueError(f"active slider {active_slider_id!r} does not exist") from error
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")

    positive_budget_units = _to_units(positive_budget)
    negative_budget_units = _to_units(negative_budget)
    if positive_budget_units < 0 or negative_budget_units < 0:
        raise ValueError("budgets must be positive magnitudes")

    units = [_to_units(slider.value) for slider in sliders]
    requested_units = _to_units(requested_value)
    units[active_index] = 0 if deadzone and abs(requested_units) <= 1 else requested_units
    spring_alpha = Decimal(str(alpha))

    _enforce_direction_budget(
        units,
        direction=1,
        budget_units=positive_budget_units,
        active_index=active_index,
        alpha=spring_alpha,
    )
    _enforce_direction_budget(
        units,
        direction=-1,
        budget_units=negative_budget_units,
        active_index=active_index,
        alpha=spring_alpha,
    )

    return [slider.model_copy(update={"value": unit / 2}) for slider, unit in zip(sliders, units)]
