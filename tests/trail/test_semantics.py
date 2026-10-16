import pytest

from trail import (
    Actor,
    InMemorySemanticTemplateRepository,
    SemanticTemplate,
    SemanticTemplateService,
    TemplateContext,
    TemplateScope,
    default_semantic_mapping,
)


def _template(
    template_id: str,
    scope: TemplateScope,
    scope_id: str | None = None,
    *,
    mapping: dict[float, str] | None = None,
    visible: bool = True,
    allow_override: bool = True,
    locked: bool = False,
) -> SemanticTemplate:
    return SemanticTemplate(
        id=template_id,
        key="priority",
        version=1,
        scope=scope,
        scope_id=scope_id,
        mapping=mapping or default_semantic_mapping(),
        visible=visible,
        allow_override=allow_override,
        locked=locked,
    )


def test_default_template_maps_every_half_step_from_minus_five_to_plus_five() -> None:
    mapping = default_semantic_mapping()

    assert list(mapping) == [step / 2 for step in range(-10, 11)]
    assert mapping[-5.0].startswith("Hard constraint")
    assert mapping[0.0] == "No preference: treat as neutral; do not prioritize or penalize."
    assert mapping[5.0].startswith("Primary objective")


def test_template_inheritance_uses_most_specific_allowed_scope() -> None:
    repository = InMemorySemanticTemplateRepository()
    service = SemanticTemplateService(repository)
    admin = Actor(user_id="admin", is_super_admin=True)
    mapping = default_semantic_mapping()

    for template in (
        _template("global", TemplateScope.GLOBAL),
        _template("org", TemplateScope.ORGANIZATION, "org-1", mapping={**mapping, 1.0: "org"}),
        _template(
            "project", TemplateScope.PROJECT, "project-1", mapping={**mapping, 1.0: "project"}
        ),
        _template("user", TemplateScope.USER, "user-1", mapping={**mapping, 1.0: "user"}),
    ):
        service.save_template(admin, template)

    context = TemplateContext(organization_id="org-1", project_id="project-1", user_id="user-1")
    assert service.resolve("priority", context).instruction_for(1.0) == "user"

    context = TemplateContext(organization_id="org-1", project_id="project-1", user_id="other")
    assert service.resolve("priority", context).instruction_for(1.0) == "project"

    context = TemplateContext(organization_id="org-1", project_id="other", user_id="other")
    assert service.resolve("priority", context).instruction_for(1.0) == "org"


def test_only_super_admin_can_save_central_templates_or_view_hidden_templates() -> None:
    service = SemanticTemplateService(InMemorySemanticTemplateRepository())
    admin = Actor(user_id="admin", is_super_admin=True)
    user = Actor(user_id="user")
    hidden = _template("hidden", TemplateScope.GLOBAL, visible=False)

    with pytest.raises(PermissionError):
        service.save_template(user, hidden)

    service.save_template(admin, hidden)
    with pytest.raises(PermissionError):
        service.view_template(user, hidden.id)
    assert service.view_template(admin, hidden.id) == hidden


def test_user_cannot_replace_central_template_by_reusing_its_id() -> None:
    service = SemanticTemplateService(InMemorySemanticTemplateRepository())
    admin = Actor(user_id="admin", is_super_admin=True)
    user = Actor(user_id="user-1")
    service.save_template(admin, _template("global", TemplateScope.GLOBAL))
    collision = _template("global", TemplateScope.USER, "user-1")

    with pytest.raises(PermissionError):
        service.save_template(user, collision)

    assert service.view_template(admin, "global").scope == TemplateScope.GLOBAL


def test_user_presets_are_private_but_super_admin_can_view_them() -> None:
    service = SemanticTemplateService(InMemorySemanticTemplateRepository())
    admin = Actor(user_id="admin", is_super_admin=True)
    owner = Actor(user_id="owner")
    other = Actor(user_id="other")
    service.save_template(admin, _template("global", TemplateScope.GLOBAL))
    preset = service.create_user_preset(
        owner,
        key="priority",
        name="Private",
        mapping=default_semantic_mapping(),
    )

    assert service.view_template(owner, preset.id) == preset
    with pytest.raises(PermissionError):
        service.view_template(other, preset.id)
    assert service.view_template(admin, preset.id) == preset


def test_locked_global_template_blocks_inheritance_but_super_admin_can_override_it() -> None:
    repository = InMemorySemanticTemplateRepository()
    service = SemanticTemplateService(repository)
    admin = Actor(user_id="admin", is_super_admin=True)
    mapping = default_semantic_mapping()
    service.save_template(admin, _template("global", TemplateScope.GLOBAL, locked=True))
    service.save_template(
        admin,
        _template(
            "project", TemplateScope.PROJECT, "project-1", mapping={**mapping, 1.0: "project"}
        ),
    )

    context = TemplateContext(project_id="project-1", user_id="user-1")
    assert service.resolve("priority", context).id == "global"
    assert service.resolve("priority", context, actor=admin).id == "project"


def test_locked_project_template_blocks_user_default_below_it() -> None:
    service = SemanticTemplateService(InMemorySemanticTemplateRepository())
    admin = Actor(user_id="admin", is_super_admin=True)
    user = Actor(user_id="user-1")
    mapping = default_semantic_mapping()
    service.save_template(admin, _template("global", TemplateScope.GLOBAL))
    service.save_template(
        admin,
        _template("project", TemplateScope.PROJECT, "project-1", locked=True),
    )
    preset = service.create_user_preset(
        user,
        key="priority",
        name="Default",
        mapping={**mapping, 1.0: "user"},
        make_default=True,
    )
    context = TemplateContext(project_id="project-1", user_id="user-1")

    assert service.resolve("priority", context, actor=user).id == "project"
    assert service.resolve("priority", context, actor=admin).id == preset.id


def test_user_can_save_and_reuse_named_preset_when_overrides_are_allowed() -> None:
    repository = InMemorySemanticTemplateRepository()
    service = SemanticTemplateService(repository)
    admin = Actor(user_id="admin", is_super_admin=True)
    user = Actor(user_id="user-1")
    mapping = default_semantic_mapping()
    service.save_template(admin, _template("global", TemplateScope.GLOBAL))

    preset = service.create_user_preset(
        user,
        key="priority",
        name="My planning preset",
        mapping={**mapping, 2.0: "Prefer the long-term option."},
        make_default=False,
    )
    context = TemplateContext(user_id="user-1")

    assert service.resolve("priority", context).id == "global"
    reused = service.resolve("priority", context, preset_id=preset.id)
    assert reused.id == preset.id
    assert reused.instruction_for(2.0) == "Prefer the long-term option."

    with pytest.raises(PermissionError):
        service.resolve(
            "priority",
            TemplateContext(user_id="other"),
            actor=user,
            preset_id=preset.id,
        )


def test_default_user_preset_and_revert_to_inherited_default() -> None:
    service = SemanticTemplateService(InMemorySemanticTemplateRepository())
    admin = Actor(user_id="admin", is_super_admin=True)
    user = Actor(user_id="user-1")
    mapping = default_semantic_mapping()
    service.save_template(admin, _template("global", TemplateScope.GLOBAL))
    preset = service.create_user_preset(
        user,
        key="priority",
        name="Default preset",
        mapping={**mapping, 0.5: "slightly custom"},
        make_default=True,
    )
    context = TemplateContext(user_id="user-1")

    assert service.resolve("priority", context).id == preset.id
    service.revert_user_default(user, key="priority")
    assert service.resolve("priority", context).id == "global"


def test_user_override_is_rejected_when_global_policy_disallows_it() -> None:
    service = SemanticTemplateService(InMemorySemanticTemplateRepository())
    admin = Actor(user_id="admin", is_super_admin=True)
    user = Actor(user_id="user-1")
    service.save_template(admin, _template("global", TemplateScope.GLOBAL, allow_override=False))

    with pytest.raises(PermissionError):
        service.create_user_preset(
            user,
            key="priority",
            name="Not allowed",
            mapping=default_semantic_mapping(),
        )


def test_template_mapping_rejects_missing_or_off_grid_values() -> None:
    mapping = default_semantic_mapping()
    mapping.pop(4.5)
    with pytest.raises(ValueError):
        _template("missing", TemplateScope.GLOBAL, mapping=mapping)

    mapping = default_semantic_mapping()
    mapping[1.25] = "off grid"
    with pytest.raises(ValueError):
        _template("off-grid", TemplateScope.GLOBAL, mapping=mapping)
