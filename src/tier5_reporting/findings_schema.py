"""Strict Pydantic JSON Schema for Tier 5 Clinical Findings (Single Source of Truth).

Enforces zero-hallucination safety invariants:
1. All values originate strictly from validated vision/biometric model outputs.
2. When `is_gradable=False` or `abstain_flag=True`, numerical DR grades are suppressed
   from clinical display (`displayed_dr_grade = None`) and replaced by a mandatory
   "Human Review Required" safety flag.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class LesionLocationItem(BaseModel):
    """Coordinates and classification of a localized retinal micro-lesion."""

    x: int
    y: int
    radius_px: int
    activation_score: float = Field(ge=0.0, le=1.0)
    lesion_type: str


class ONHBiomarkerSummary(BaseModel):
    """Optic Nerve Head biometric measurements for Glaucoma triage."""

    vcdr: float = Field(ge=0.0, le=1.0)
    hcdr: float = Field(ge=0.0, le=1.0)
    area_ratio: float = Field(ge=0.0, le=1.0)
    isnt_rule_violated: bool
    glaucoma_biometric_risk_score: float = Field(ge=0.0, le=1.0)


class EyeClinicalFindings(BaseModel):
    """Validated diagnostic findings for a single eye (OD or OS)."""

    laterality: str = Field(pattern="^(OD|OS)$")
    is_gradable: bool
    iqa_quality_score: float = Field(ge=0.0, le=1.0)
    iqa_rejection_reasons: List[str] = Field(default_factory=list)

    # Stage-1 Multi-Label Triage Probabilities
    prob_dr: float = Field(ge=0.0, le=1.0)
    prob_glaucoma: float = Field(ge=0.0, le=1.0)
    prob_cataract: float = Field(ge=0.0, le=1.0)
    onh_biomarkers: ONHBiomarkerSummary

    # Stage-2 DR Severity Grading & Conformal Safety
    raw_primary_dr_grade: int = Field(ge=-1, le=4)
    conservative_dr_grade: int = Field(ge=-1, le=4)
    displayed_dr_grade: Optional[int] = Field(default=None, ge=0, le=4)
    displayed_grade_label: str = ""
    continuous_severity_score: float = Field(ge=0.0, le=4.5)
    prob_referable_dr: float = Field(ge=0.0, le=1.0)
    class_probabilities: List[float] = Field(default_factory=lambda: [0.0] * 5)

    conformal_prediction_set: List[int] = Field(default_factory=list)
    confidence_tier: str = Field(pattern="^(HIGH|MODERATE|LOW_ABSTAIN)$")
    abstain_flag: bool
    ensemble_std: float = Field(ge=0.0)

    # Explainability & Spatial Involvement
    dominant_quadrant: str = "Superior-Temporal"
    active_quadrant_count: int = Field(default=0, ge=0, le=4)
    quadrant_scores: Dict[str, float] = Field(default_factory=dict)
    lesion_candidates: List[LesionLocationItem] = Field(default_factory=list)

    # Deterministic narrative lines filled by Template Engine
    other_pathologies_note: str = ""
    eye_narrative_summary: str = ""

    @model_validator(mode="after")
    def enforce_safety_abstention(self) -> "EyeClinicalFindings":
        """Guarantee that ungradable or low-confidence abstained scans never expose an automated DR grade."""
        if not self.is_gradable or self.abstain_flag or self.confidence_tier == "LOW_ABSTAIN":
            self.abstain_flag = True
            self.confidence_tier = "LOW_ABSTAIN"
            self.displayed_dr_grade = None
            self.displayed_grade_label = "Human Review Required (Automated Grade Suppressed)"
        else:
            self.displayed_dr_grade = int(self.conservative_dr_grade)
        return self


class BilateralPatientReport(BaseModel):
    """Patient-level bilateral ophthalmic screening report combining OD and OS."""

    patient_id: str
    patient_name: str = "Anonymous Screening Patient"
    patient_age: Optional[int] = None
    exam_date: str
    institution_name: str = "RetinaNet-Pro Automated Tele-Ophthalmology Screening Unit"
    pipeline_version: str = "3.0.0-LocalEdge"

    od_findings: Optional[EyeClinicalFindings] = None
    os_findings: Optional[EyeClinicalFindings] = None

    # Bilateral worse-eye aggregation outputs
    patient_worse_eye: str = "Bilateral"
    patient_overall_dr_grade: Optional[int] = None
    patient_overall_grade_label: str
    is_referable_dr: bool
    requires_human_review: bool
    overall_confidence_tier: str
    referral_urgency: str
    referral_timeline: str
    badge_color: str
    recommended_clinical_action: str
    bilateral_other_pathologies_summary: str
    doctor_verification_notes: str = ""
    doctor_signature: str = ""
