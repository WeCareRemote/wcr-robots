from trail import NodeSlider, rebalance_node_sliders


def _values(sliders: list[NodeSlider]) -> dict[str, float]:
    return {slider.id: slider.value for slider in sliders}


def _assert_invariants(sliders: list[NodeSlider]) -> None:
    values = [slider.value for slider in sliders]
    assert all(-5.0 <= value <= 5.0 for value in values)
    assert all((value * 2).is_integer() for value in values)
    assert sum(value for value in values if value > 0) <= 5.0
    assert sum(-value for value in values if value < 0) <= 5.0


def test_two_sliders_spring_back_active_and_other_proportionally() -> None:
    sliders = [NodeSlider(id="a", name="A", value=2.0), NodeSlider(id="b", name="B", value=2.0)]

    result = rebalance_node_sliders(sliders, active_slider_id="a", requested_value=5.0)

    assert _values(result) == {"a": 4.5, "b": 0.5}
    _assert_invariants(result)


def test_three_sliders_use_alpha_point_three_and_proportional_reduction() -> None:
    sliders = [
        NodeSlider(id="a", name="A", value=3.0),
        NodeSlider(id="b", name="B", value=2.0),
        NodeSlider(id="c", name="C", value=0.0),
    ]

    result = rebalance_node_sliders(sliders, active_slider_id="c", requested_value=4.0)

    assert _values(result) == {"a": 1.0, "b": 1.0, "c": 3.0}
    _assert_invariants(result)


def test_many_dynamic_sliders_keep_invariants_through_repeated_updates() -> None:
    sliders = [NodeSlider(id=str(index), name=f"Slider {index}") for index in range(25)]

    for index in range(100):
        active = str(index % len(sliders))
        requested = ((index * 7) % 23 - 11) / 2
        sliders = rebalance_node_sliders(
            sliders,
            active_slider_id=active,
            requested_value=requested,
        )
        _assert_invariants(sliders)


def test_positive_and_negative_budgets_are_independent() -> None:
    sliders = [
        NodeSlider(id="positive-a", name="Positive A", value=3.0),
        NodeSlider(id="positive-b", name="Positive B", value=2.0),
        NodeSlider(id="negative-a", name="Negative A", value=-4.0),
        NodeSlider(id="negative-b", name="Negative B", value=-1.0),
    ]

    result = rebalance_node_sliders(
        sliders,
        active_slider_id="positive-b",
        requested_value=5.0,
    )

    values = _values(result)
    assert values["negative-a"] == -4.0
    assert values["negative-b"] == -1.0
    assert values["positive-a"] + values["positive-b"] == 5.0
    _assert_invariants(result)


def test_negative_overflow_only_rebalances_negative_sliders() -> None:
    sliders = [
        NodeSlider(id="a", name="A", value=-3.0),
        NodeSlider(id="b", name="B", value=-2.0),
        NodeSlider(id="positive", name="Positive", value=4.0),
    ]

    result = rebalance_node_sliders(sliders, active_slider_id="b", requested_value=-5.0)

    values = _values(result)
    assert values["positive"] == 4.0
    assert values["a"] + values["b"] == -5.0
    _assert_invariants(result)


def test_neutral_value_releases_budget_without_moving_other_sliders() -> None:
    sliders = [
        NodeSlider(id="a", name="A", value=3.0),
        NodeSlider(id="b", name="B", value=2.0),
        NodeSlider(id="c", name="C", value=-5.0),
    ]

    result = rebalance_node_sliders(sliders, active_slider_id="a", requested_value=0.0)

    assert _values(result) == {"a": 0.0, "b": 2.0, "c": -5.0}
    _assert_invariants(result)


def test_requested_value_is_clamped_and_snapped_to_half_steps() -> None:
    sliders = [NodeSlider(id="a", name="A")]

    assert (
        rebalance_node_sliders(sliders, active_slider_id="a", requested_value=8.2)[0].value == 5.0
    )
    assert (
        rebalance_node_sliders(sliders, active_slider_id="a", requested_value=-8.2)[0].value == -5.0
    )
    assert (
        rebalance_node_sliders(sliders, active_slider_id="a", requested_value=1.24)[0].value == 1.0
    )
    assert (
        rebalance_node_sliders(sliders, active_slider_id="a", requested_value=1.25)[0].value == 1.5
    )


def test_optional_deadzone_can_make_neutral_more_stable() -> None:
    sliders = [NodeSlider(id="a", name="A")]

    result = rebalance_node_sliders(
        sliders,
        active_slider_id="a",
        requested_value=0.5,
        deadzone=True,
    )

    assert result[0].value == 0.0


def test_active_slider_absorbs_remaining_overflow_when_no_other_slider_can_move() -> None:
    sliders = [NodeSlider(id="a", name="A")]

    result = rebalance_node_sliders(
        sliders,
        active_slider_id="a",
        requested_value=5.0,
        positive_budget=2.0,
    )

    assert result[0].value == 2.0


def test_unknown_or_duplicate_slider_ids_are_rejected() -> None:
    sliders = [NodeSlider(id="a", name="A")]

    try:
        rebalance_node_sliders(sliders, active_slider_id="missing", requested_value=1.0)
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing active slider must fail")

    duplicate = [NodeSlider(id="a", name="A"), NodeSlider(id="a", name="Again")]
    try:
        rebalance_node_sliders(duplicate, active_slider_id="a", requested_value=1.0)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate slider IDs must fail")
