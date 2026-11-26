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
   - **Default model:** ImageNet-pretrained `timm` backbone + GeM + Tri-Head (see Training section). ResViT-Pro below is kept as a compact from-scratch ablation.
   - **ResViT-Pro:** Multi-scale FPN CNN stream (stride-4 feature maps sensitive to <10px microaneurysms) coupled with SwinV2/ViT global transformer stream via Bidirectional Cross-Attention.
   - **Generalized Mean (GeM) Pooling:** Learnable $-norm pooling preventing tiny focal microaneurysms from being washed out.
   - **Tri-Head Diagnostic Layer:** CORAL/CORN ordinal logits (Grades 0–4), binary referable DR head (Grade $\ge$ 2), and continuous severity regressor.

5. **Tier 4: Meta-Ensembling, Calibration & Explainability**
   - Out-of-fold stacking meta-learner (LightGBM/Ridge) + 8-view Dihedral Test-Time Augmentation (TTA).
   - Temperature scaling and Split Conformal Prediction (target 95% coverage predictive sets, measured on the test ensemble in `test_report.conformal_coverage`).
   - Dual Explainability: High-resolution **Grad-CAM++** for micro-lesions + **Cross-Attention Rollout** for quadrant-level involvement.

6. **Tier 5: Clinical Reporting & Doctor Dashboard**
   - **100% Offline Local Execution:** Fast inference (<500ms per bilateral pair) on laptop GPUs (RTX 3050 / 4050 6GB VRAM) with zero cloud hosting costs.
   - **Zero-Hallucination Medical Report Engine:** Deterministic templating engine based on validated vision outputs.
   - **Bilateral Patient-Level Aggregation:** Evaluates Left (OS) and Right (OD) eyes, determining patient referral priority by the worse eye.
   - **Interactive UI & 1-Click PDF:** Local dashboard with side-by-side heatmaps and printable clinical PDF export via WeasyPrint.

---

## ⚡ Training & Evaluation Workflow

All training logic lives in `src/training/` and is driven by the YAML configs; the Colab notebooks are thin drivers.

| Step | Notebook / CLI | Config | Output (mirrored to Drive) |
|---|---|---|---|
| Stage-1 triage | `colab/01_…` or `python -m src.training.cli stage1` | `configs/tier2_stage1_triage.yaml` | `stage1_multi_label_best.pth`, `stage1_metrics.json` |
| Stage-2 grading (4-fold CV) | `colab/02_…` or `python -m src.training.cli stage2 [--folds 1 2]` | `configs/tier3_stage2_grading.yaml` | `stage2_grader_fold{1..4}.pth`, `stage2_oof.npz`, `stage2_test_ensemble.npz`, `stage2_cv_metrics.json` |
| Calibration | `colab/03_…` or `python -m src.training.cli calibrate` | – | `calibration_params.json` (thresholds, temperature, conformal `q_hat`, test report) |

Key properties:

- **ImageNet-pretrained `timm` backbones** (default EfficientNet-B3) with GeM pooling and the task heads; the from-scratch `ResViTPro` / StrongCNN+ConvNeXt ensemble remain available as `arch: resvit_pro` / `arch: legacy_ensemble` for ablations.
- **Stage-1 source-shortcut fix:** in the original Stage-1 data every DR-positive image is EyePACS and every negative is ODIR. Training uses `stage1_triage_combined_splits.csv` (built automatically), which adds APTOS DR-negative and DR-positive scans and masks unknown labels. Report `auc_dr_within_source_mean`, not just `auc_dr`.
- **True out-of-fold predictions:** one Stage-2 model per validation fold (1–4); fold 0 is the locked test split, evaluated once per model. The four fold models are ensembled at inference.
- **Correct mixed precision:** fp16 on T4 uses a `GradScaler`; bf16 where supported.
- **Resumable:** every epoch writes a full training state (model, optimizer, EMA, scaler); finished folds are skipped, so CV can span several Colab sessions.
- **Calibration without double-dipping:** thresholds on all OOF, temperature on OOF half A, conformal on OOF half B, and everything reported on the test split with bootstrap CIs.
- **Inference parity & safety:** inference resizes exactly like the training-data writer, checkpoints load with `strict=True`, untrained modules (CDAE denoiser, ONH disc/cup U-Net) are never applied, temperature + thresholds are used at inference, and Grad-CAM++ uses real gradients. Saliency regions are not labelled as specific lesions unless a lesion-supervised model is used.

Quick local smoke run (CPU, tiny):

```bash
python -m src.training.cli stage1 --out /tmp/smoke --set model.backbone_name=efficientnet_b0 model.pretrained=false \
    data.resolution=128 data.max_train_samples=64 data.max_eval_samples=48 training.epochs=1 training.num_workers=0
pytest tests -q
```

### Known limitations / next steps

- Stage-2 is trained on APTOS-2019 only (single source, ~3.4k images). Add EyePACS / DDR / Messidor-2 and keep **Messidor-2 or IDRiD as an external test set** never used for training.
- Stage-1 DR labels still partially correlate with source (EyePACS positives only); the within-source AUC is the trustworthy number.
- The ONH disc/cup U-Net, the CDAE denoiser and the lesion branch need their own labelled data (REFUGE/ORIGA, IDRiD/DDR lesion masks) before they are used clinically.

---

## 📚 Complete Architectural Blueprint

For the full forensic paper audit, mathematical formulas, and phase-by-phase implementation details, refer to:
👉 **[ARCHITECTURE_AND_WORKFLOW.md](./ARCHITECTURE_AND_WORKFLOW.md)**

---

## ⚖️ License & Disclaimer

- **Code:** released under the [MIT License](./LICENSE).
- **Datasets & trained weights:** the MIT license covers only the source code in this repository. The fundus image datasets (APTOS 2019, EyePACS, ODIR-5K and others) are **not** redistributed here and remain subject to their original licenses / terms of use. Model weights trained on them should be treated as subject to those same terms (several permit non-commercial research use only).
- **Not a medical device:** this is a research and educational project. It has not been clinically validated or approved by any regulatory body and must not be used to diagnose, treat or make referral decisions for real patients.
