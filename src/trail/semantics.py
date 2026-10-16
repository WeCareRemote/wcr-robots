from collections.abc import Iterable
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def default_semantic_mapping() -> dict[float, str]:
    """Canonical AI instructions for every supported half-step."""

    return {
        -5.0: (
            "Hard constraint: strongly avoid this criterion. If trade-offs exist, sacrifice "
            "this criterion first."
        ),
        -4.5: "Very strongly avoid: treat as highly undesirable; only accept if unavoidable.",
        -4.0: "Strongly avoid: penalize this criterion significantly in decisions.",
        -3.5: (
            "Avoid: prefer options that reduce this criterion; accept only with clear benefits "
            "elsewhere."
        ),
        -3.0: "Moderately avoid: lean away from this criterion when choosing among options.",
        -2.5: "Slightly avoid: small penalty; do not optimize for it.",
        -2.0: ("Weak avoid: minor negative preference; keep an eye on it but don’t overreact."),
        -1.5: "Very light avoid: only a subtle nudge away.",
        -1.0: "Barely avoid: almost neutral, but slightly unfavorable.",
        -0.5: "Near-neutral: treat as effectively neutral unless it’s easy to improve.",
        0.0: "No preference: treat as neutral; do not prioritize or penalize.",
        0.5: "Near-neutral: slight positive nudge if it’s easy/low-cost.",
        1.0: "Barely prefer: small positive preference; minor tie-breaker.",
        1.5: "Very light prefer: gentle nudge toward options that satisfy it.",
        2.0: "Weak prefer: minor priority; consider it but don’t over-optimize.",
        2.5: "Slightly prefer: moderate bonus; good to have.",
        3.0: "Moderately prefer: meaningful priority; actively optimize where possible.",
        3.5: (
            "Prefer: strong positive; choose options that support it unless there’s a strong "
            "reason not to."
        ),
        4.0: "Strongly prefer: major priority; trade off other positives before giving this up.",
        4.5: (
            "Very strongly prefer: treat as near-primary; only compromise with explicit "
            "justification."
        ),
        5.0: (
            "Primary objective: strongly prioritize this criterion. Optimize for it first and "
            "justify any deviation."
        ),
    }


class TemplateScope(StrEnum):
    GLOBAL = "global"
    ORGANIZATION = "organization"
    PROJECT = "project"
    USER = "user"


class Actor(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    is_super_admin: bool = False


class TemplateContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    organization_id: str | None = None
    project_id: str | None = None
    user_id: str | None = None


class SemanticTemplate(BaseModel):
    """Versioned semantic mapping at one inheritance scope."""

    model_config = ConfigDict(frozen=True)

    id: str
    key: str
    version: int = Field(ge=1)
    scope: TemplateScope
    scope_id: str | None = None
    mapping: dict[float, str]
    visible: bool = True
    allow_override: bool = True
    locked: bool = False
    preset_name: str | None = None
    is_default: bool = False

    @model_validator(mode="after")
    def validate_scope_and_mapping(self) -> "SemanticTemplate":
        if self.scope == TemplateScope.GLOBAL and self.scope_id is not None:
            raise ValueError("global templates cannot have a scope_id")
        if self.scope != TemplateScope.GLOBAL and not self.scope_id:
            raise ValueError(f"{self.scope.value} templates require a scope_id")
        expected = {step / 2 for step in range(-10, 11)}
        if set(self.mapping) != expected:
            raise ValueError("semantic mapping must define every 0.5 step from -5.0 to +5.0")
        if any(not instruction.strip() for instruction in self.mapping.values()):
            raise ValueError("semantic instructions cannot be blank")
        return self

    def instruction_for(self, value: float) -> str:
        if value not in self.mapping:
            raise ValueError(f"value {value} is not a supported 0.5 step")
        return self.mapping[value]


class SemanticTemplateRepository(Protocol):
    def save(self, template: SemanticTemplate) -> None: ...

    def get(self, template_id: str) -> SemanticTemplate | None: ...

    def list(self, key: str | None = None) -> Iterable[SemanticTemplate]: ...


class InMemorySemanticTemplateRepository:
    """Small adapter for tests/local composition; production storage implements the protocol."""

    def __init__(self) -> None:
        self._templates: dict[str, SemanticTemplate] = {}

    def save(self, template: SemanticTemplate) -> None:
        self._templates[template.id] = template

    def get(self, template_id: str) -> SemanticTemplate | None:
        return self._templates.get(template_id)

    def list(self, key: str | None = None) -> Iterable[SemanticTemplate]:
        templates = self._templates.values()
        if key is None:
            return tuple(templates)
        return tuple(template for template in templates if template.key == key)


class SemanticTemplateService:
    def __init__(self, repository: SemanticTemplateRepository) -> None:
        self.repository = repository

    def save_template(self, actor: Actor, template: SemanticTemplate) -> SemanticTemplate:
        existing = self.repository.get(template.id)
        if not actor.is_super_admin:
            if template.scope != TemplateScope.USER or template.scope_id != actor.user_id:
                raise PermissionError("only a Super Admin may save central semantic templates")
            if existing and (
                existing.scope != TemplateScope.USER or existing.scope_id != actor.user_id
            ):
                raise PermissionError("a user cannot replace a template owned by another scope")
            self._ensure_user_override_allowed(template.key)
        self.repository.save(template)
        return template

    def view_template(self, actor: Actor, template_id: str) -> SemanticTemplate:
        template = self.repository.get(template_id)
        if template is None:
            raise KeyError(template_id)
        if (
            template.scope == TemplateScope.USER
            and template.scope_id != actor.user_id
            and not actor.is_super_admin
        ):
            raise PermissionError("a user preset can only be viewed by its owner")
        if not template.visible and not actor.is_super_admin:
            raise PermissionError("this semantic template is hidden")
        return template

    def create_user_preset(
        self,
        actor: Actor,
        *,
        key: str,
        name: str,
        mapping: dict[float, str],
        make_default: bool = False,
    ) -> SemanticTemplate:
        self._ensure_user_override_allowed(key, actor=actor)
        if make_default:
            self.revert_user_default(actor, key=key)
        versions = [template.version for template in self.repository.list(key)]
        preset = SemanticTemplate(
            id=f"user-preset-{uuid4()}",
            key=key,
            version=max(versions, default=0) + 1,
            scope=TemplateScope.USER,
            scope_id=actor.user_id,
            mapping=mapping,
            preset_name=name,
            is_default=make_default,
        )
        self.repository.save(preset)
        return preset

    def revert_user_default(self, actor: Actor, *, key: str) -> None:
        for template in self.repository.list(key):
            if (
                template.scope == TemplateScope.USER
                and template.scope_id == actor.user_id
                and template.is_default
            ):
                self.repository.save(template.model_copy(update={"is_default": False}))

    def resolve(
        self,
        key: str,
        context: TemplateContext,
        *,
        actor: Actor | None = None,
        preset_id: str | None = None,
    ) -> SemanticTemplate:
        candidates = list(self.repository.list(key))
        global_templates = [
            template for template in candidates if template.scope == TemplateScope.GLOBAL
        ]
        if not global_templates:
            raise LookupError(f"no global semantic template exists for {key!r}")
        global_template = max(global_templates, key=lambda template: template.version)
        super_admin = bool(actor and actor.is_super_admin)
        if actor and not super_admin and actor.user_id != context.user_id:
            raise PermissionError("an actor can only resolve templates for their own user context")

        central_template = global_template
        for scope, scope_id in (
            (TemplateScope.ORGANIZATION, context.organization_id),
            (TemplateScope.PROJECT, context.project_id),
        ):
            if not super_admin and (central_template.locked or not central_template.allow_override):
                break
            scoped = [
                template
                for template in candidates
                if template.scope == scope and template.scope_id == scope_id
            ]
            if scoped:
                central_template = max(scoped, key=lambda template: template.version)

        if preset_id is not None:
            preset = self.repository.get(preset_id)
            if preset is None or preset.key != key or preset.scope != TemplateScope.USER:
                raise LookupError(f"user preset {preset_id!r} does not exist for {key!r}")
            if not super_admin and preset.scope_id != context.user_id:
                raise PermissionError("a user preset can only be reused by its owner")
            if not super_admin and (central_template.locked or not central_template.allow_override):
                raise PermissionError("semantic overrides are disabled for this template")
            return preset

        if not super_admin and (central_template.locked or not central_template.allow_override):
            return central_template

        user_templates = [
            template
            for template in candidates
            if template.scope == TemplateScope.USER
            and template.scope_id == context.user_id
            and (template.is_default or template.preset_name is None)
        ]
        if user_templates:
            return max(user_templates, key=lambda template: template.version)
        return central_template

    def _ensure_user_override_allowed(self, key: str, actor: Actor | None = None) -> None:
        global_templates = [
            template
            for template in self.repository.list(key)
            if template.scope == TemplateScope.GLOBAL
        ]
        if not global_templates:
            raise LookupError(f"no global semantic template exists for {key!r}")
        policy = max(global_templates, key=lambda template: template.version)
        if actor and actor.is_super_admin:
            return
        if policy.locked or not policy.allow_override:
            raise PermissionError("semantic overrides are disabled for this template")
