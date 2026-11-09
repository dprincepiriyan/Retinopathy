"""Deterministic Zero-Hallucination Medical Report Template Engine.

Generates clinical text strictly from validated numerical outputs in `EyeClinicalFindings`
and `BilateralPatientReport` without using any generative LLMs.
"""

from pathlib import Path
from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.tier5_reporting.findings_schema import (
    BilateralPatientReport,
    EyeClinicalFindings,
)


GRADE_LABELS: Dict[int, str] = {
    0: "Grade 0 — No Apparent Diabetic Retinopathy",
    1: "Grade 1 — Mild Non-Proliferative Diabetic Retinopathy (NPDR)",
    2: "Grade 2 — Moderate Non-Proliferative Diabetic Retinopathy (NPDR)",
    3: "Grade 3 — Severe Non-Proliferative Diabetic Retinopathy (NPDR)",
    4: "Grade 4 — Proliferative Diabetic Retinopathy (PDR)",
}

REFERRAL_RULES: Dict[int | str, Dict[str, str]] = {
    0: {
        "grade_label": GRADE_LABELS[0],
        "urgency": "Routine Rescreening",
        "timeline": "12 Months",
        "badge_color": "green",
        "clinical_action": (
            "Maintain annual dilated fundus screening and systemic glycemic/blood pressure control."
        ),
    },
    1: {
        "grade_label": GRADE_LABELS[1],
        "urgency": "Enhanced Monitoring",
        "timeline": "6–12 Months",
        "badge_color": "blue",
        "clinical_action": (
            "Repeat fundus examination in 6–12 months; optimize HbA1c and systemic lipid profile."
        ),
    },
    2: {
        "grade_label": GRADE_LABELS[2],
        "urgency": "Referable DR — Non-Urgent Ophthalmic Referral",
        "timeline": "3–6 Months",
        "badge_color": "amber",
        "clinical_action": (
            "Refer to ophthalmologist within 3–6 months for comprehensive dilated retinal evaluation and macular OCT."
        ),
    },
    3: {
        "grade_label": GRADE_LABELS[3],
        "urgency": "Urgent Ophthalmic Referral",
        "timeline": "2–4 Weeks",
        "badge_color": "orange",
        "clinical_action": (
            "Urgent referral to retinal specialist within 2–4 weeks; high risk of progression to proliferative retinopathy."
        ),
    },
    4: {
        "grade_label": GRADE_LABELS[4],
        "urgency": "Immediate Sight-Threatening Emergency Referral",
        "timeline": "< 48 Hours",
        "badge_color": "red",
        "clinical_action": (
            "Immediate referral (<48 hours) to vitreoretinal specialist for panretinal photocoagulation (PRP) or anti-VEGF therapy."
        ),
    },
    "abstain": {
        "grade_label": "Human Review Required (Automated Grade Suppressed)",
        "urgency": "Manual Specialist Review / Recapture Required",
        "timeline": "Immediate Recapture or Manual Triage",
        "badge_color": "gray",
        "clinical_action": (
            "Automated grading suppressed by clinical safety gate. Re-acquire mydriatic fundus photograph or route directly to an ophthalmologist."
        ),
    },
}


class DeterministicClinicalTemplateEngine:
    """Fills deterministic clinical narratives and renders the HTML/CSS medical report card."""

    def __init__(self, templates_dir: str | Path | None = None) -> None:
        if templates_dir is None:
            templates_dir = Path(__file__).resolve().parent / "templates"
        self.templates_dir = Path(templates_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    @staticmethod
    def build_other_pathologies_note(eye: EyeClinicalFindings) -> str:
        """Build the deterministic 1-line 'Note on Other Pathologies' from Stage-1 outputs."""
        vcdr = eye.onh_biomarkers.vcdr
        isnt_violated = eye.onh_biomarkers.isnt_rule_violated

        if eye.prob_glaucoma >= 0.45 or vcdr >= 0.55:
            rim_note = "ISNT neuroretinal rim rule violated" if isnt_violated else "borderline rim thinning"
            glaucoma_clause = (
                f"Elevated glaucoma suspicion (vCDR: {vcdr:.2f}, P={eye.prob_glaucoma:.2f}, {rim_note})"
            )
        else:
            glaucoma_clause = f"Optic Cup-to-Disc ratio is normal (vCDR: {vcdr:.2f}, low glaucoma suspicion)"

        if eye.prob_cataract >= 0.50:
            cataract_clause = f"media opacification detected suggestive of cataract (P={eye.prob_cataract:.2f})."
        else:
            cataract_clause = "no significant lens opacification (cataract negative)."

        return f"Note on Other Pathologies ({eye.laterality}): {glaucoma_clause}; {cataract_clause}"

    @classmethod
    def populate_eye_narratives(cls, eye: EyeClinicalFindings) -> EyeClinicalFindings:
        """Populate deterministic text fields on an `EyeClinicalFindings` instance in-place."""
        eye.other_pathologies_note = cls.build_other_pathologies_note(eye)

        if not eye.is_gradable:
            reasons_str = "; ".join(eye.iqa_rejection_reasons) or "Sub-threshold image quality."
            eye.displayed_grade_label = REFERRAL_RULES["abstain"]["grade_label"]
            eye.eye_narrative_summary = (
                f"{eye.laterality} scan rejected by Tier-0 Quality Gate (IQA Score: {eye.iqa_quality_score:.2f}). "
                f"Reason: {reasons_str} Human review or image recapture required."
            )
            return eye

        if eye.abstain_flag or eye.displayed_dr_grade is None:
            set_str = ", ".join(f"Grade {g}" for g in eye.conformal_prediction_set) or "Ambiguous"
            eye.displayed_grade_label = REFERRAL_RULES["abstain"]["grade_label"]
            eye.eye_narrative_summary = (
                f"{eye.laterality} scan flagged for manual specialist review due to high predictive uncertainty "
                f"(95% Conformal Set: {{{set_str}}}, Ensemble Std: {eye.ensemble_std:.2f})."
            )
            return eye

        grade = int(eye.displayed_dr_grade)
        eye.displayed_grade_label = GRADE_LABELS[grade]
        set_str = ", ".join(f"Grade {g}" for g in eye.conformal_prediction_set)
        lesion_count = len(eye.lesion_candidates)

        eye.eye_narrative_summary = (
            f"{eye.laterality} assessed as {GRADE_LABELS[grade]} "
            f"(Continuous Severity: {eye.continuous_severity_score:.2f}/4.50, "
            f"Referable DR Prob: {eye.prob_referable_dr * 100:.1f}%, "
            f"95% Conformal Set: {{{set_str}}}, Confidence: {eye.confidence_tier}). "
            f"Localized {lesion_count} focal lesion candidate(s), predominantly in the {eye.dominant_quadrant} quadrant."
        )
        return eye

    def render_html_report(
        self,
        report: BilateralPatientReport,
        extra_context: Dict[str, Any] | None = None,
    ) -> str:
        """Render the complete hospital-grade HTML report card."""
        css_path = self.templates_dir / "style.css"
        inline_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
        template = self.env.get_template("report_card.html")
        ctx: Dict[str, Any] = {
            "report": report,
            "inline_css": inline_css,
        }
        if extra_context:
            ctx.update(extra_context)
        return template.render(**ctx)
