# RetinaNet-Pro: Automated Retinal Disease Screening & DR Severity Classification

An end-to-end, multi-stage deep learning framework for automated retinal pathology screening and fine-grained Diabetic Retinopathy (DR) severity classification.

Designed as an advanced evolution of the multi-stage hybrid paradigm, overcoming early-stage micro-lesion degradation, severe class imbalance, and cross-device domain shifts across multi-centre fundus imaging.

---

## 🏛️ System Architecture Overview

The system operates across a **6-Tier Modular Pipeline (Tiers 0–5)**:

1. **Tier 0: Deterministic Clinical Preprocessing & Quality Gate**
   - Automated circular Field-of-View (FOV) detection & aspect-preserving square padding.
   - Local image quality assessment (IQA) gatekeeper (rejection/recapture flag for ungradable scans).
   - Ben Graham local illumination subtraction + Green-channel & LAB CLAHE contrast normalization.

2. **Tier 1: Domain Harmonization & Micro-Lesion Preservation**
   - **Fourier Domain Adaptation (FDA):** Generator-free, low-frequency amplitude style alignment (<5ms latency, zero lesion hallucination risk).
   - **Residual Attention U-Net CDAE:** Predicts noise residuals ($\hat{x} = x + lpha \cdot \Delta x$) to preserve microaneurysms (<10px) while removing sensor/Poisson noise.
   - **Lesion-Locked CUT/CycleGAN:** High-pass structural lock used for training-time domain augmentation.

3. **Tier 2: Stage-1 Multi-Label Disease Triage & Glaucoma ONH Engine**
   - Multi-label binary classification (independent Sigmoids for **Diabetic Retinopathy**, **Glaucoma**, and **Cataract**).
   - **Optic Nerve Head (ONH) ROI Specialist:** Automated Optic Disc and Cup segmentation computing vertical Cup-to-Disc Ratio (vCDR) and ISNT neuroretinal rim rule adherence.
   - Model Zoo: StrongCNN baseline, ConvNeXt-V2-Tiny with Coordinate Attention, and RETFound-LoRA.

4. **Tier 3: Stage-2 Fine-Grained DR Severity Grading**
   - Evaluates **all gradable eye scans** (eliminating cascade false-negatives from Stage 1).
   - **ResViT-Pro:** Multi-scale FPN CNN stream (stride-4 feature maps sensitive to <10px microaneurysms) coupled with SwinV2/ViT global transformer stream via Bidirectional Cross-Attention.
   - **Generalized Mean (GeM) Pooling:** Learnable $-norm pooling preventing tiny focal microaneurysms from being washed out.
   - **Tri-Head Diagnostic Layer:** CORAL/CORN ordinal logits (Grades 0–4), binary referable DR head (Grade $\ge$ 2), and continuous severity regressor.

5. **Tier 4: Meta-Ensembling, Calibration & Explainability**
   - Out-of-fold stacking meta-learner (LightGBM/Ridge) + 8-view Dihedral Test-Time Augmentation (TTA).
   - Temperature scaling ( < 0.04$) and Split Conformal Prediction (guaranteed 95% coverage predictive sets).
   - Dual Explainability: High-resolution **Grad-CAM++** for micro-lesions + **Cross-Attention Rollout** for quadrant-level involvement.

6. **Tier 5: Clinical Reporting & Doctor Dashboard**
   - **100% Offline Local Execution:** Fast inference (<500ms per bilateral pair) on laptop GPUs (RTX 3050 / 4050 6GB VRAM) with zero cloud hosting costs.
   - **Zero-Hallucination Medical Report Engine:** Deterministic templating engine based on validated vision outputs.
   - **Bilateral Patient-Level Aggregation:** Evaluates Left (OS) and Right (OD) eyes, determining patient referral priority by the worse eye.
   - **Interactive UI & 1-Click PDF:** Local dashboard with side-by-side heatmaps and printable clinical PDF export via WeasyPrint.

---

## ⚡ Execution Strategy: Hybrid Cloud-Train & Local-Infer

- **Training:** Executed on **Google Colab GPUs (L4 / A100 / T4)** using mixed-precision FP16, EMA, and PyTorch Lightning checkpoints.
- **Inference & Demo:** Lightweight exported weights (.pth / .onnx ~80MB total) run entirely locally on laptop GPUs (<1.8 GB VRAM footprint).

---

## 📚 Complete Architectural Blueprint

For the full forensic paper audit, mathematical formulas, and phase-by-phase implementation details, refer to:
👉 **[ARCHITECTURE_AND_WORKFLOW.md](./ARCHITECTURE_AND_WORKFLOW.md)**
