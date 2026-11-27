# RetinaNet-Pro: Project Progress & Engineering Milestone Report

**Last Updated:** November 21, 2026  
**Repository:** `Retinopathy` (`RetinaNet-Pro`)  
**Execution Paradigm:** Hybrid Cloud-Train (Google Colab GPU/TPU) & Offline Local Inference (<500ms on Laptop GPU)

---

## 1. Executive Summary

`RetinaNet-Pro` is an end-to-end, 6-Tier clinical deep learning framework for automated multi-disease retinal screening (**Diabetic Retinopathy**, **Glaucoma**, **Cataract**) and fine-grained **ICDR Diabetic Retinopathy Severity Grading (Grades 0–4)**.

### Milestones Completed So Far:
- [x] **Milestone 1 — Forensic Reference Paper Audit & System Blueprint:** Identified 7 critical bottlenecks in the baseline paper (including the 52% missed Mild DR collapse, Glaucoma vs. Normal confusion, and pre-split augmentation leakage) and architected the 6-Tier `RetinaNet-Pro` pipeline (`ARCHITECTURE_AND_WORKFLOW.md`).
- [x] **Milestone 2 — Complete 6-Tier Modular Codebase Implementation:** Implemented all core modules across `src/` (`data`, `tier1_purifiers`, `tier2_triage`, `tier3_grading`, `losses`, `tier4_ensemble`, `tier5_reporting`, `explainability`, `utils`), interactive doctor dashboard (`app/dashboard.py`), and Google Colab training notebooks (`colab/`).
- [x] **Milestone 3 — Automated Verification & Safety Test Suite:** Built and verified 18 unit tests (`tests/`) covering zero patient leakage across folds, `<10px` micro-lesion retention under Fourier Domain Adaptation, differentiable Soft-QWK & Nelder-Mead threshold search, and zero-hallucination clinical reporting.
- [x] **Milestone 4 — Multi-Dataset Forensic Overlap & Label-Corruption Audit:** Audited 5 raw zip archives (`24,915` total raw files, `~11.3 GB`), uncovering **68.4% cross-archive redundancy** and **2,007 scrambled/corrupted labels** in secondary low-resolution archives (`dataset-2.zip` / `dataset-4.zip`).
- [x] **Milestone 5 — End-to-End Tier-0 Clinical Preprocessing & Patient-Grouped Splitting:** Executed multi-core circular FOV cropping (`cv2.convexHull` background floor detector), `IQAGatekeeper` quality filtering, Green-Channel + LAB CLAHE enhancement, 64-bit inner-vessel `pHash` deduplication, canonical FDA reference spectrum fitting, and 5-fold `StratifiedGroupKFold` splitting across **`7,545` clean, unique, gradable retinal scans**.
- [x] **Milestone 5b — Pre-Training Correctness Review & Fixes (Nov 21, 2026):** see Section 7.
- [ ] **Milestone 6 — Cloud Training on Google Colab (Next Step):** Upload `data/processed/` to Google Drive, run Stage-1 (`01_colab_stage1_training.ipynb`) and Stage-2 (`02_colab_stage2_training.ipynb`) training on Colab GPU/TPU, and export calibrated `.pth`/`.onnx` weights back to `checkpoints/`.

---

## 2. Implemented 6-Tier System Architecture (`src/`)

| Tier | Module Directory | Implemented Algorithms & Components | Status |
| :--- | :--- | :--- | :--- |
| **Tier 0** | `src/data/` & `src/utils/` | Automated Circular FOV Detection & 1:1 Aspect-Preserving Pad (`preprocessing.py`), `IQAGatekeeper` blur/glare/exposure filter (`iqa_gatekeeper.py`), Ben Graham illumination subtraction, Green-Channel + LAB Luminance CLAHE, 64-bit inner-square vessel DCT `pHash` deduplication & `StratifiedGroupKFold` splitter (`datasets.py`, `dataset_cleaner_cli.py`). | **Complete & Executed** |
| **Tier 1** | `src/tier1_purifiers/` | Generator-free Fast Fourier Domain Adaptation (`fda.py`), Residual Attention U-Net CDAE (`residual_cdae.py`), and Lesion-Locked CycleGAN/CUT (`cyclegan_lesion_locked.py`). | **Complete** |
| **Tier 2** | `src/tier2_triage/` | Multi-label independent Sigmoid triage (`multi_label_triage.py`), `StrongCNN` EfficientNet-SE baseline (`strong_cnn.py`), `ConvNeXtV2CoordinateTriage` (`convnext_triage.py`), and `OpticNerveHeadUNet` Disc/Cup segmentation with vertical Cup-to-Disc Ratio (`vCDR`) & ISNT rule verification (`glaucoma_onh_branch.py`). | **Complete** |
| **Tier 3** | `src/tier3_grading/` | `ResViTPro` hybrid architecture combining Stride-4 FPN local micro-lesion stream + Global Transformer stream with Bidirectional Cross-Attention (`resvit_pro.py`), Learnable Generalized Mean (`GeM`) pooling (`gem_pooling.py`), Auxiliary Lesion-Guided Attention (`lesion_guided_net.py`), and Tri-Head Diagnostic Output (CORAL Ordinal Logits + Referable DR Head + Continuous Regressor in `multihead_grading.py`). | **Complete** |
| **Tier 4** | `src/tier4_ensemble/` & `src/losses/` | `AsymmetricFocalLoss`, `CORALOrdinalFocalLoss`, differentiable `SoftQWKLoss`, Nelder-Mead Simplex QWK threshold optimizer (`qwk_threshold_optimizer.py`), Temperature Scaling (`temperature_scaling.py`), Split Conformal Prediction (`conformal_predictor.py`), LightGBM/Ridge OOF Stacking (`stacking_meta.py`), Grad-CAM++ (`gradcam_pp.py`), and Attention Rollout (`attention_rollout.py`). | **Complete** |
| **Tier 5** | `src/tier5_reporting/` & `app/` | Strict Pydantic clinical findings schema (`findings_schema.py`), Bilateral OD/OS Worse-Eye Patient Aggregator (`bilateral_aggregator.py`), deterministic zero-hallucination report engine (`template_engine.py`), WeasyPrint 1-click PDF exporter (`pdf_generator.py`), and offline Doctor Dashboard (`app/dashboard.py`). | **Complete** |

---

## 3. Forensic Audit of Raw Datasets (`datasets/dataset-1.zip` .. `dataset-5.zip`)

Before running preprocessing, all 5 raw zip archives (`24,915` files, `~11.3 GB`) were forensically audited via SHA-256 file hashing, filename stem cross-matching, ground-truth label verification, and 64-bit DCT perceptual hashing (`pHash`):

| Archive | Size | Files | Forensic Finding | Disposition |
| :--- | :--- | :--- | :--- | :--- |
| **`dataset-1.zip`** | 472 MB | 4,217 | Contains `seg_dataset/{cataract, glaucoma, diabetic_retinopathy, normal}` (`512x512`, ODIR-5K + EyePACS). Zero overlap with `dataset-3.zip`. Identified 32 filename collisions between distinct ODIR (`.jpg`) and EyePACS (`.jpeg`) patients and 2 identical scans (`1415_right`, `625_left`) having co-existing Cataract + Glaucoma. | **Kept & Preprocessed** for **Stage-1 Multi-Label Triage** (ODIR vs. EyePACS IDs disambiguated; multi-label comorbidities merged). |
| **`dataset-3.zip`** | 9.81 GB | 10,000 | Contains `2,000` files per folder (`0`–`4`), consisting of **3,662 authentic high-resolution (`1050x1050` to `2588x3388`) APTOS 2019 `.png` scans** plus **6,338 pre-augmented `aug_*.jpg` copies**. | **Kept 3,662 High-Res Originals** for **Stage-2 DR Grading** (`512x512` & `768x768`); **Discarded 6,338 `aug_*.jpg` copies** to prevent train/val/test data leakage. |
| **`dataset-5.zip`** | 342 MB | 3,590 | `224x224` downscaled `.png` files. **100% (3,590 / 3,590)** of its images are already present in `dataset-3.zip` at full clinical resolution. | **Excluded & Deleted** (100% redundant low-res duplicate of `dataset-3.zip`). |
| **`dataset-2.zip`** | 217 MB | 3,554 | `224x224` downscaled `.png` files. **88.5% (3,146 / 3,554)** are APTOS 2019 images already in `dataset-3.zip`, and **2,007 of those 3,146 scans have corrupted/scrambled folder labels** (e.g., Grade 0 Normal scans placed inside `Severe` and `Mild`, Grade 2 Moderate placed in `Proliferate_DR`). | **Excluded & Deleted** (prevents severe label poisoning). |
| **`dataset-4.zip`** | 217 MB | 3,554 | **100% byte-for-byte identical copy** of `dataset-2.zip` (`SHA-256: 1627932e858b5fd9752b01bbbb04b2fbec4babe999ed43867ed6acbf6e2a1c32`). | **Excluded & Deleted** (exact duplicate). |

---

## 4. Tier-0 Preprocessing & Clean Split Summary

All authoritative scans from `dataset-1.zip` (`4,217`) and `dataset-3.zip` (`3,662` un-augmented high-resolution originals) were processed through the multi-core Tier-0 pipeline (`src/data/dataset_cleaner_cli.py`):

1. **Circular Retinal Lens FOV Detection & 1:1 Aspect-Preserving Square Pad:** Uses background-floor thresholding + `cv2.convexHull` to preserve circular Optic Disc geometry (`median circularity = 0.89–0.90`).
2. **Tier-0 Image Quality Assessment (`IQAGatekeeper`):** Filtered out `183` ungradable scans exhibiting severe motion/defocus blur, flash glare overexposure, or lens occlusion.
3. **Green-Channel + LAB Luminance CLAHE:** Enhanced micro-vascular contrast at peak hemoglobin absorption (`540–570 nm`).
4. **Inner-Square Vessel 64-bit `pHash` Deduplication:** Computed 64-bit DCT perceptual hashes on the central 70% inscribed retinal square (`Hamming <= 2`), removing `151` near-duplicate scans and merging `38` multi-label comorbidity records.
5. **Zero-Leakage Patient-Stratified 5-Fold Splitting (`StratifiedGroupKFold`):** Grouped strictly by `patient_id` with **Fold 0 locked as an untouched test split** and **Folds 1–4 assigned to `train_val`**.

### Final Dataset Numbers (`data/processed/`):
- **Total Raw Files Inspected:** `24,915`
- **Redundant / Pre-Augmented / Corrupted Files Removed:** `17,036`
- **Authoritative Scans Evaluated by Tier-0 Pipeline:** `7,879`
- **Corrupt / Unreadable Files:** `0`
- **Rejected by Tier-0 IQA Gatekeeper:** `183`
- **Near-Duplicates Removed by 64-bit `pHash`:** `151` *(38 merged as multi-label comorbidities)*
- **Final Clean, Unique, Gradable Scans:** **`7,545`**
  - **Stage-1 Multi-Label Triage (`data/processed/stage1_triage_512/`):** **`4,150` scans** (`3,320` in `train_val` Folds 1–4 | `830` in `test` Fold 0)
  - **Stage-2 DR Severity Grading (`data/processed/tensors_512/` & `tensors_768/`):** **`3,395` scans** (`2,716` in `train_val` Folds 1–4 | `679` in `test` Fold 0)
    - `Grade 0 (No DR)`: **1,780** (`52.4%`)
    - `Grade 1 (Mild DR)`: **318** (`9.4%`)
    - `Grade 2 (Moderate DR)`: **867** (`25.5%`)
    - `Grade 3 (Severe DR)`: **170** (`5.0%`)
    - `Grade 4 (Proliferative DR)`: **260** (`7.7%`)
- **Canonical FDA Reference Spectrum:** Saved to `data/processed/fda_canonical_amplitude.npy` (fitted across the top 64 highest-IQA normal reference scans).

---

## 5. Verification & Unit Test Suite

All **31 unit tests** in `tests/` pass (`pytest tests -q`):
- `tests/test_no_patient_leakage.py`: Verifies perceptual hash deduplication, zero `patient_id` overlap across CV folds/test split, and bilateral OD/OS worse-eye aggregation.
- `tests/test_microlesion_retention.py`: Verifies Fourier Domain Adaptation (`FDA`) and Residual Attention U-Net CDAE preserve `<10px` microaneurysms without hallucination.
- `tests/test_ordinal_metrics.py`: Verifies `CORALOrdinalFocalLoss`, differentiable `SoftQWKLoss`, `AsymmetricFocalLoss`, and Nelder-Mead simplex QWK threshold optimization.
- `tests/test_report_safety.py`: Verifies deterministic clinical report generation, low-confidence abstention safety gates, and HTML/PDF output integrity.
- `tests/test_training_upgrades.py`: Verifies strict checkpoints, masked Stage-1 loss, source-shortcut manifest & leakage guard, augmentation, AMP stepping, LR schedule, calibration coverage, and an end-to-end pipeline run on trained checkpoints.

---

## 6. Next Steps (Google Colab Training Workflow)

1. **Upload `retinopathy_processed.tar.gz` to Google Drive** (unchanged archive; the combined Stage-1 manifest is built automatically on Colab).
2. **Notebook 01 — Stage-1 triage:** pretrained EfficientNet-B3, combined manifest, masked ASL loss. Watch `auc_dr_within_source_mean`.
3. **Notebook 02 — Stage-2 grading, 4-fold CV:** Phase A 512 px (12 epochs) → Phase B 768 px (14 epochs) per fold (~1 h 25 min per fold on a T4). Spread folds across sessions with `FOLDS = [1, 2]` / `[3, 4]`; everything resumes from Drive.
4. **Notebook 03 — Calibration:** thresholds / temperature / conformal from true OOF predictions; test-split report with bootstrap CIs.
5. Copy `calibration_params.json`, `stage1_multi_label_best.pth` and `stage2_grader_fold*.pth` into `checkpoints/` and run `streamlit run app/dashboard.py`.

---

## 7. Pre-Training Correctness Review (Milestone 5b)

| Issue found | Fix |
|---|---|
| Stage-1 DR label fully determined by source (all positives EyePACS, all negatives ODIR) | `src/data/stage1_manifest.py` adds APTOS DR-negative + DR-positive scans; unknown labels masked in `AsymmetricFocalLoss`; within-source DR AUC reported |
| Tiny from-scratch networks (~0.45M / ~1.4M params) despite configs naming pretrained backbones | `src/models/pretrained.py`: `timm` backbones + GeM + existing heads, lower backbone LR; legacy nets kept for ablation |
| fp16 autocast on T4 without loss scaling | `GradScaler` in `ColabTPUAccelerator.backward/clip_and_step`; losses computed in fp32 |
| Untrained CDAE applied to every inference image; untrained ONH U-Net fed glaucoma probability and vCDR | Both only used when a trained checkpoint exists; ONH biomarkers reported as unavailable otherwise |
| Train/inference resize mismatch (INTER_AREA + JPEG vs INTER_LINEAR) | Inference reproduces INTER_AREA + JPEG q96 |
| Temperature never applied, QWK thresholds loaded but unused, `strict=False` loading | Self-describing checkpoints loaded with `strict=True`; temperature + thresholds applied in `RetinaNetProPipeline` |
| Single validation fold used for model selection *and* calibration, mislabelled "OOF" | 4-fold CV with true OOF; calibration splits OOF (temperature vs conformal); test-split report |
| Sampler + focal alpha + threshold weights (triple imbalance correction) | Exactly one strategy (`imbalance.strategy`) |
| Soft-QWK on batch 8 | Gradient accumulation (effective batch 32); Soft-QWK only in Phase B, ramped |
| "Grad-CAM++" without gradients; unsupervised channels labelled MA/HE/EX/CWS | Real gradient Grad-CAM++; regions reported as attention foci unless lesion-supervised |
| YAML configs not used by training; EMA configured but absent | Configs drive `src/training/*`; EMA, warmup+cosine, seeding, resumable state implemented |
| D4-only augmentation | `RetinalTrainAugment`: mild colour / gamma / exposure / rotation / scale / blur |
| Cached preprocessing wrote placeholder IQA score 0.90 | IQA score stored in a sidecar and reused; unknown otherwise |

Test suite: **31 tests** (`pytest tests -q`), including checkpoint strictness, masked loss, manifest leakage, calibration coverage and an end-to-end pipeline run on trained checkpoints.
