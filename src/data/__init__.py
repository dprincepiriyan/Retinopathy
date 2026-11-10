"""Data loading, deterministic clinical preprocessing, deduplication, and samplers."""

from src.data.augmentations import (
    apply_dihedral_tta,
    lesion_copy_paste_synthesis,
    retinal_mixup,
)
from src.data.datasets import (
    BilateralPatientPair,
    PreprocessedManifestDataset,
    RetinalDatasetRecord,
    compute_perceptual_hash,
    create_patient_stratified_splits,
    deduplicate_records_by_phash,
)
from src.data.preprocessing import (
    Tier0Preprocessor,
    ben_graham_normalization,
    detect_and_crop_circular_fov,
    green_lab_clahe,
)
from src.data.samplers import CurriculumResolutionScheduler, EffectiveNumberSampler

__all__ = [
    "detect_and_crop_circular_fov",
    "ben_graham_normalization",
    "green_lab_clahe",
    "Tier0Preprocessor",
    "RetinalDatasetRecord",
    "BilateralPatientPair",
    "compute_perceptual_hash",
    "deduplicate_records_by_phash",
    "create_patient_stratified_splits",
    "retinal_mixup",
    "lesion_copy_paste_synthesis",
    "apply_dihedral_tta",
    "EffectiveNumberSampler",
    "CurriculumResolutionScheduler",
]
