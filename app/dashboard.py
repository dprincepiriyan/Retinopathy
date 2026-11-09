"""Tier 5 Interactive Doctor Dashboard (100% Offline Local Execution).

Launch with:
    streamlit run app/dashboard.py
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.assets.synthetic_fundus import generate_synthetic_fundus_image
from src.pipeline import RetinaNetProPipeline
from src.tier5_reporting.pdf_generator import ClinicalPDFGenerator
from src.tier5_reporting.template_engine import DeterministicClinicalTemplateEngine


st.set_page_config(
    page_title="RetinaNet-Pro | Clinical Doctor Dashboard",
    page_icon="👁️",
    layout="wide",
)


@st.cache_resource
def get_pipeline() -> RetinaNetProPipeline:
    return RetinaNetProPipeline()


def main() -> None:
    st.title("👁️ RetinaNet-Pro: Automated Retinal Screening & DR Severity Dashboard")
    st.caption(
        "6-Tier Hybrid Architecture • Multi-Label Stage-1 Triage • ONH vCDR/ISNT Glaucoma Biometrics • "
        "ResViT-Pro + GeM Pooling • 95% Split Conformal Prediction • Zero-Hallucination PDF Engine"
    )

    pipeline = get_pipeline()
    template_engine = DeterministicClinicalTemplateEngine()
    pdf_generator = ClinicalPDFGenerator(template_engine)

    with st.sidebar:
        st.header("📋 Patient & Exam Metadata")
        patient_id = st.text_input("Patient ID", value="PT-2026-0842")
        patient_name = st.text_input("Patient Name", value="R. Venkataraman")
        patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=58)

        st.divider()
        st.header("🖼️ Input Mode")
        input_mode = st.radio(
            "Select Scan Source",
            options=["Upload Bilateral Fundus Scans", "Clinical Simulation Presets"],
            index=1,
        )

        preset_choice = "Bilateral Moderate NPDR + Elevated vCDR (OD)"
        if input_mode == "Clinical Simulation Presets":
            preset_choice = st.selectbox(
                "Preset Scenario",
                options=[
                    "Healthy Bilateral Screening (Grade 0)",
                    "Early Mild NPDR with <10px Microaneurysms (OS)",
                    "Bilateral Moderate NPDR + Elevated vCDR (OD)",
                    "Ungradable Severe Motion Blur (Safety Gate Abstention)",
                ],
            )

        st.divider()
        st.header("✍️ Clinician Sign-Off")
        doctor_notes = st.text_area(
            "Reviewing Ophthalmologist Notes",
            value="Correlate with macular OCT and HbA1c panel.",
        )
        doctor_signature = st.text_input(
            "Verified By (Doctor Signature)",
            value="Dr. A. Subramanian, MD (Ophthalmology)",
        )

    od_img: np.ndarray | None = None
    os_img: np.ndarray | None = None

    if input_mode == "Upload Bilateral Fundus Scans":
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            od_file = st.file_uploader("Upload Right Eye (OD) Fundus", type=["png", "jpg", "jpeg"])
            if od_file is not None:
                od_img = np.array(Image.open(od_file).convert("RGB"))
        with col_u2:
            os_file = st.file_uploader("Upload Left Eye (OS) Fundus", type=["png", "jpg", "jpeg"])
            if os_file is not None:
                os_img = np.array(Image.open(os_file).convert("RGB"))
    else:
        if preset_choice == "Healthy Bilateral Screening (Grade 0)":
            od_img = generate_synthetic_fundus_image(laterality="OD", vcdr=0.31, seed=10)
            os_img = generate_synthetic_fundus_image(laterality="OS", vcdr=0.33, seed=11)
        elif preset_choice == "Early Mild NPDR with <10px Microaneurysms (OS)":
            od_img = generate_synthetic_fundus_image(laterality="OD", vcdr=0.32, seed=20)
            os_img = generate_synthetic_fundus_image(
                laterality="OS", num_microaneurysms=6, vcdr=0.34, seed=21
            )
        elif preset_choice == "Bilateral Moderate NPDR + Elevated vCDR (OD)":
            od_img = generate_synthetic_fundus_image(
                laterality="OD", num_microaneurysms=10, num_hemorrhages=5, num_exudates=4, vcdr=0.64, seed=30
            )
            os_img = generate_synthetic_fundus_image(
                laterality="OS", num_microaneurysms=4, num_hemorrhages=1, vcdr=0.36, seed=31
            )
        else:
            od_img = generate_synthetic_fundus_image(
                laterality="OD", simulate_severe_blur=True, seed=40
            )
            os_img = generate_synthetic_fundus_image(
                laterality="OS", simulate_flash_glare=True, seed=41
            )

    if od_img is None and os_img is None:
        st.info("Please upload at least one fundus photograph (OD or OS) to run screening.")
        return

    report, visuals = pipeline.evaluate_bilateral_patient(
        patient_id=patient_id,
        od_image_rgb=od_img,
        os_image_rgb=os_img,
        patient_name=patient_name,
        patient_age=int(patient_age),
        doctor_verification_notes=doctor_notes,
        doctor_signature=doctor_signature,
    )

    # Top-level Bilateral Patient Banner
    st.subheader("🏥 Patient-Level Bilateral Triage Summary (Worse-Eye Priority)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Overall Assessment", report.patient_overall_grade_label.split("—")[0].strip())
    b2.metric("Worse Eye Driver", report.patient_worse_eye)
    b3.metric("Referral Timeline", report.referral_timeline)
    b4.metric("Confidence Tier", report.overall_confidence_tier)

    st.markdown(f"**Recommended Clinical Action:** {report.recommended_clinical_action}")
    st.markdown(f"**{report.bilateral_other_pathologies_summary}**")

    st.divider()
    col_od, col_os = st.columns(2)

    for col, lat, findings in [
        (col_od, "OD", report.od_findings),
        (col_os, "OS", report.os_findings),
    ]:
        with col:
            eye_title = "Right Eye (OD)" if lat == "OD" else "Left Eye (OS)"
            st.subheader(f"👁️ {eye_title}")
            if findings is None or lat not in visuals:
                st.write("No scan provided.")
                continue

            vis = visuals[lat]
            img_tabs = st.tabs(["Enhanced CLAHE + FDA", "Grad-CAM++ Micro-Lesions", "ONH Glaucoma ROI"])
            with img_tabs[0]:
                st.image(vis.enhanced_rgb, caption=f"{lat} Tier-0/1 Preprocessed Scan", use_container_width=True)
            with img_tabs[1]:
                st.image(
                    vis.gradcam_overlay_rgb,
                    caption=f"{lat} Stride-4 Grad-CAM++ Overlay ({len(findings.lesion_candidates)} focal candidates)",
                    use_container_width=True,
                )
            with img_tabs[2]:
                st.image(
                    vis.onh_roi_rgb,
                    caption=f"{lat} Optic Nerve Head ROI (vCDR: {findings.onh_biomarkers.vcdr:.2f})",
                    width=220,
                )

            st.markdown(f"**Displayed Grade:** `{findings.displayed_grade_label}`")
            st.markdown(
                f"- **95% Conformal Prediction Set:** `{findings.conformal_prediction_set}` "
                f"(`{findings.confidence_tier}`)\n"
                f"- **Referable DR (Grade ≥ 2) Probability:** `{findings.prob_referable_dr * 100:.1f}%`\n"
                f"- **Stage-1 Multi-Label Triage:** DR `{findings.prob_dr:.2f}` | "
                f"Glaucoma `{findings.prob_glaucoma:.2f}` | Cataract `{findings.prob_cataract:.2f}`\n"
                f"- **ONH Biometrics:** vCDR `{findings.onh_biomarkers.vcdr:.2f}`, "
                f"ISNT Violated: `{findings.onh_biomarkers.isnt_rule_violated}`\n"
                f"- **Narrative:** {findings.eye_narrative_summary}"
            )

    st.divider()
    st.subheader("📄 Printable Hospital Report Card & 1-Click PDF Export")
    pdf_bytes = pdf_generator.generate_pdf_bytes(report)
    st.download_button(
        label="⬇️ Download Clinical PDF Report (WeasyPrint / PDF-1.4)",
        data=pdf_bytes,
        file_name=f"RetinaNetPro_Report_{report.patient_id}.pdf",
        mime="application/pdf",
    )

    with st.expander("Preview Rendered HTML Clinical Report Card", expanded=False):
        html_str = template_engine.render_html_report(report)
        st.components.v1.html(html_str, height=680, scrolling=True)


if __name__ == "__main__":
    main()
