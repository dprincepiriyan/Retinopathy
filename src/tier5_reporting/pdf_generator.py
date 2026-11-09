"""WeasyPrint HTML-to-PDF Clinical Report Generator with Pure-Python PDF Fallback.

Converts the deterministic HTML report card into a printable hospital-grade PDF
with 1 click. Uses WeasyPrint when system Cairo/Pango libraries are available,
and falls back to a clean self-contained PDF 1.4 writer when running in minimal environments.
"""

from pathlib import Path
from typing import Optional

from src.tier5_reporting.findings_schema import BilateralPatientReport
from src.tier5_reporting.template_engine import DeterministicClinicalTemplateEngine


class ClinicalPDFGenerator:
    """Generates printable PDF clinical reports from `BilateralPatientReport`."""

    def __init__(self, template_engine: Optional[DeterministicClinicalTemplateEngine] = None) -> None:
        self.template_engine = template_engine or DeterministicClinicalTemplateEngine()

    @staticmethod
    def _build_minimal_pdf_bytes(lines: list[str]) -> bytes:
        """Generate a valid standalone PDF-1.4 document without external C libraries."""
        escaped_lines = []
        for line in lines[:48]:
            clean = (
                line.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
                .replace("—", "-")
                .replace("–", "-")
            )
            clean_ascii = clean.encode("ascii", errors="replace").decode("ascii")
            escaped_lines.append(clean_ascii)

        content_ops = ["BT", "/F1 10 Tf", "45 790 Td", "14 TL"]
        for idx, text_line in enumerate(escaped_lines):
            if idx == 0:
                content_ops.append(f"({text_line}) Tj")
            else:
                content_ops.append(f"T* ({text_line}) Tj")
        content_ops.append("ET")
        stream_data = "\n".join(content_ops).encode("ascii")

        objects = [
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
            b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
            f"5 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("ascii")
            + stream_data
            + b"\nendstream\nendobj\n",
        ]

        pdf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for obj in objects:
            offsets.append(len(pdf))
            pdf.extend(obj)

        xref_start = len(pdf)
        pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
        pdf.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))

        pdf.extend(
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode(
                "ascii"
            )
        )
        return bytes(pdf)

    def generate_pdf_bytes(self, report: BilateralPatientReport) -> bytes:
        """Render the `BilateralPatientReport` to PDF bytes."""
        html_str = self.template_engine.render_html_report(report)
        try:
            from weasyprint import HTML  # type: ignore

            return HTML(string=html_str, base_url=str(self.template_engine.templates_dir)).write_pdf()
        except Exception:
            # Fallback to deterministic pure-Python PDF 1.4 generator if WeasyPrint/Pango isn't installed
            lines = [
                f"RETINANET-PRO CLINICAL SCREENING REPORT ({report.pipeline_version})",
                "=" * 72,
                f"Institution : {report.institution_name}",
                f"Patient ID  : {report.patient_id}    Name: {report.patient_name}",
                f"Exam Date   : {report.exam_date}",
                "-" * 72,
                "PATIENT-LEVEL BILATERAL SUMMARY (WORSE-EYE PRIORITY)",
                f"Overall Assessment : {report.patient_overall_grade_label}",
                f"Worse Eye Driver   : {report.patient_worse_eye}",
                f"Referral Urgency   : {report.referral_urgency} ({report.referral_timeline})",
                f"Confidence Tier    : {report.overall_confidence_tier}",
                f"Human Review Req.  : {'YES' if report.requires_human_review else 'NO'}",
                f"Clinical Action    : {report.recommended_clinical_action}",
                "-" * 72,
            ]
            for label, eye in [("RIGHT EYE (OD)", report.od_findings), ("LEFT EYE (OS)", report.os_findings)]:
                lines.append(label)
                if eye is None:
                    lines.append("  Not provided.")
                else:
                    lines.append(f"  Gradable IQA : {eye.is_gradable} (Score: {eye.iqa_quality_score:.2f})")
                    lines.append(f"  DR Grade     : {eye.displayed_grade_label}")
                    lines.append(f"  Conformal Set: {eye.conformal_prediction_set} ({eye.confidence_tier})")
                    lines.append(
                        f"  Stage-1 Probs: DR={eye.prob_dr:.2f}, Glaucoma={eye.prob_glaucoma:.2f}, Cataract={eye.prob_cataract:.2f}"
                    )
                    lines.append(
                        f"  ONH Biomarker: vCDR={eye.onh_biomarkers.vcdr:.2f}, ISNT Violated={eye.onh_biomarkers.isnt_rule_violated}"
                    )
                    lines.append(f"  {eye.other_pathologies_note}")
                lines.append("-" * 72)

            if report.doctor_verification_notes:
                lines.append(f"Doctor Notes: {report.doctor_verification_notes}")
            if report.doctor_signature:
                lines.append(f"Signed By   : {report.doctor_signature}")
            return self._build_minimal_pdf_bytes(lines)

    def save_pdf(self, report: BilateralPatientReport, output_path: str | Path) -> Path:
        """Render and save the printable clinical PDF to `output_path`."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf_bytes = self.generate_pdf_bytes(report)
        path.write_bytes(pdf_bytes)
        return path
