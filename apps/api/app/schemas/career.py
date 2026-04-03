from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class CareerAnalyzeRequest(BaseModel):
    intent: str = Field(default="auto", description="score | recommend | improve_cv | auto")
    query: str = Field(default="", description="Optional free-text user request/context")
    preferred_role: str | None = Field(default=None, description="Optional user-selected target role")
    cv_text: str | None = Field(default=None, description="Raw CV text for new users")
    cv_file_path: str | None = Field(default=None, description="Local file path (.pdf/.docx/.txt)")
    cv_filename: str | None = Field(default=None, description="Optional filename hint")
    debug: bool = Field(default=False, description="Return extra debug payload for admin usage")

    @model_validator(mode="after")
    def validate_cv_source(self) -> "CareerAnalyzeRequest":
        text_ok = bool((self.cv_text or "").strip())
        path_ok = bool((self.cv_file_path or "").strip())
        if not text_ok and not path_ok:
            raise ValueError("One of `cv_text` or `cv_file_path` is required.")
        return self


class RoleCandidate(BaseModel):
    role: str
    confidence: float
    reasons: list[str] = Field(default_factory=list)


class RoleInferenceResult(BaseModel):
    selected_role: str
    confidence: float
    requires_confirmation: bool
    candidates: list[RoleCandidate] = Field(default_factory=list)


class CVSnapshot(BaseModel):
    internal_cv_id: str
    source_type: str
    extracted_target_role: str
    experience_years: str
    skills: list[str] = Field(default_factory=list)
    projects_count: int = 0


class OrchestrationPlan(BaseModel):
    intent: str
    next_actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CareerAnalyzeResponse(BaseModel):
    cv_snapshot: CVSnapshot
    role_inference: RoleInferenceResult
    orchestration: OrchestrationPlan
    debug_payload: dict | None = None
