"""Bilateral Patient-Level Aggregator (Right Eye OD & Left Eye OS).

Adheres to international ophthalmic screening guidelines:
- Patient-level referral urgency is dictated by the WORSE eye.
- If either eye triggers safety abstention (ungradable IQA or high conformal uncertainty),
  `requires_human_review` is set to True so no sight-threatening pathology is missed.
"""

from datetime import datetime, timezone
from typing import Optional

from src.tier5_reporting.findings_schema import (
    BilateralPatientReport,
    EyeClinicalFindings,
)
from src.tier5_reporting.template_engine import (
    DeterministicClinicalTemplateEngine,
    REFERRAL_RULES,
)


class BilateralClinicalAggregator:
    """Aggregates OD (Right) and OS (Left) eye findings into a unified patient-level decision."""

    def aggregate(
        self,
        patient_id: str,
        od_findings: Optional[EyeClinicalFindings] = None,
        os_findings: Optional[EyeClinicalFindings] = None,
        patient_name: str = "Anonymous Screening Patient",
        patient_age: Optional[int] = None,
        exam_date: Optional[str] = None,
        doctor_verification_notes: str = "",
        doctor_signature: str = "",
    ) -> BilateralPatientReport:
        """Combine OD and OS findings into a `BilateralPatientReport`."""
        if od_findings is None and os_findings is None:
            raise ValueError("At least one eye (OD or OS) must be provided for aggregation.")

        if exam_date is None:
            exam_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        if od_findings is not None:
            DeterministicClinicalTemplateEngine.populate_eye_narratives(od_findings)
        if os_findings is not None:
            DeterministicClinicalTemplateEngine.populate_eye_narratives(os_findings)

        available_eyes = [e for e in (od_findings, os_findings) if e is not None]
        gradable_confident_eyes = [
            e for e in available_eyes if e.is_gradable and not e.abstain_flag and e.displayed_dr_grade is not None
        ]
        any_abstained = any(e.abstain_flag or not e.is_gradable for e in available_eyes)

        other_notes = [e.other_pathologies_note for e in available_eyes if e.other_pathologies_note]
        bilateral_other_summary = " | ".join(other_notes)

        if not gradable_confident_eyes:
            # Both eyes (or the only provided eye) abstained / ungradable
            rule = REFERRAL_RULES["abstain"]
            return BilateralPatientReport(
                patient_id=patient_id,
                patient_name=patient_name,
                patient_age=patient_age,
                exam_date=exam_date,
                od_findings=od_findings,
                os_findings=os_findings,
                patient_worse_eye="Ungradable / Ambiguous",
                patient_overall_dr_grade=None,
                patient_overall_grade_label=rule["grade_label"],
                is_referable_dr=True,  # Safety-first: ungradable scans require manual referral/recapture
                requires_human_review=True,
                overall_confidence_tier="LOW_ABSTAIN",
                referral_urgency=rule["urgency"],
                referral_timeline=rule["timeline"],
                badge_color=rule["badge_color"],
                recommended_clinical_action=rule["clinical_action"],
                bilateral_other_pathologies_summary=bilateral_other_summary,
                doctor_verification_notes=doctor_verification_notes,
                doctor_signature=doctor_signature,
            )

        # Determine the worse eye among gradable confident eyes
        worse_eye_obj = max(
            gradable_confident_eyes,
            key=lambda e: (int(e.displayed_dr_grade or 0), e.continuous_severity_score, e.prob_referable_dr),
        )
        worse_grade = int(worse_eye_obj.displayed_dr_grade or 0)

        if (
            len(gradable_confident_eyes) == 2
            and gradable_confident_eyes[0].displayed_dr_grade == gradable_confident_eyes[1].displayed_dr_grade
        ):
            worse_eye_label = f"Bilateral Symmetric ({worse_eye_obj.laterality} lead)"
        else:
            worse_eye_label = f"{worse_eye_obj.laterality} ({'Right Eye' if worse_eye_obj.laterality == 'OD' else 'Left Eye'})"

        rule = REFERRAL_RULES[worse_grade]
        is_referable = bool(
            worse_grade >= 2
            or any(e.prob_referable_dr >= 0.38 for e in gradable_confident_eyes)
            or any_abstained
        )

        # Overall confidence is the minimum confidence across evaluated eyes
        tiers = {e.confidence_tier for e in available_eyes}
        if "LOW_ABSTAIN" in tiers:
            overall_conf = "LOW_ABSTAIN"
        elif "MODERATE" in tiers:
            overall_conf = "MODERATE"
        else:
            overall_conf = "HIGH"

        action_text = rule["clinical_action"]
        if any_abstained:
            action_text += (
                " NOTE: One eye scan was ungradable/ambiguous and requires manual slit-lamp or mydriatic recapture."
            )

        return BilateralPatientReport(
            patient_id=patient_id,
            patient_name=patient_name,
            patient_age=patient_age,
            exam_date=exam_date,
            od_findings=od_findings,
            os_findings=os_findings,
            patient_worse_eye=worse_eye_label,
            patient_overall_dr_grade=worse_grade,
            patient_overall_grade_label=rule["grade_label"],
            is_referable_dr=is_referable,
            requires_human_review=any_abstained,
            overall_confidence_tier=overall_conf,
            referral_urgency=rule["urgency"],
            referral_timeline=rule["timeline"],
            badge_color=rule["badge_color"],
            recommended_clinical_action=action_text,
            bilateral_other_pathologies_summary=bilateral_other_summary,
            doctor_verification_notes=doctor_verification_notes,
            doctor_signature=doctor_signature,
        )
