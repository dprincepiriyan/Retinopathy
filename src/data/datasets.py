"""Multi-dataset loader, perceptual hash deduplication, and zero-leakage patient splitter.

Addresses Bottleneck 7 from the forensic paper audit:
- Eliminates cross-dataset duplicate images via 64-bit DCT perceptual hashing (pHash).
- Partitions strictly by `patient_id` using StratifiedGroupKFold so Left (OS) and
  Right (OD) eyes of the same individual never leak across train/val/test splits.
- Pairs bilateral scans (OD & OS) for patient-level worse-eye referral aggregation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class RetinalDatasetRecord:
    """Single eye scan record across aggregated ophthalmic datasets."""

    image_id: str
    patient_id: str
    laterality: str  # "OD" (Right Eye) or "OS" (Left Eye)
    dr_grade: int    # ICDR 0, 1, 2, 3, 4
    glaucoma: int = 0
    cataract: int = 0
    source_dataset: str = "eyepacs"
    image_path: str = ""
    phash: Optional[int] = None
    is_augmented: bool = False
    fold: int = -1


@dataclass
class BilateralPatientPair:
    """Bilateral fundus examination pairing Right (OD) and Left (OS) eyes."""

    patient_id: str
    od_record: Optional[RetinalDatasetRecord] = None
    os_record: Optional[RetinalDatasetRecord] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def patient_worse_dr_grade(self) -> int:
        """International ophthalmic screening standard: patient grade = worse eye grade."""
        grades = []
        if self.od_record is not None:
            grades.append(self.od_record.dr_grade)
        if self.os_record is not None:
            grades.append(self.os_record.dr_grade)
        if not grades:
            raise ValueError(f"Patient {self.patient_id} has neither OD nor OS record.")
        return int(max(grades))

    @property
    def is_referable_dr(self) -> bool:
        """Referable Diabetic Retinopathy is defined as ICDR Grade >= 2 in either eye."""
        return self.patient_worse_dr_grade >= 2


def compute_perceptual_hash(image_rgb: np.ndarray, hash_size: int = 8) -> int:
    """Compute a 64-bit DCT Perceptual Hash (pHash) for a fundus image.

    Args:
        image_rgb: RGB or grayscale image array.
        hash_size: Side of low-frequency DCT block (8 -> 64-bit integer hash).

    Returns:
        64-bit unsigned integer perceptual hash.
    """
    if image_rgb.ndim == 3:
        # Green channel provides peak contrast for retinal vessels and lesions
        gray = image_rgb[:, :, 1]
    else:
        gray = image_rgb

    # Crop inner inscribed square (central 70%) so circular black FOV corners
    # do not dominate the low-frequency DCT coefficients across distinct patients
    h, w = gray.shape[:2]
    if h >= 32 and w >= 32:
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        gray = gray[y1:y2, x1:x2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    img_size = hash_size * 4
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size]
    # Exclude DC component [0, 0] when computing median so global brightness shifts don't alter hash
    dct_flat = dct_low.flatten()
    median_val = float(np.median(dct_flat[1:]))
    bits = (dct_flat > median_val).astype(np.uint64)

    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)
    return int(hash_val)


def hamming_distance_64bit(hash_a: int, hash_b: int) -> int:
    """Compute Hamming distance between two 64-bit perceptual hashes."""
    return int(bin(hash_a ^ hash_b).count("1"))


def deduplicate_records_by_phash(
    records: List[RetinalDatasetRecord],
    max_hamming_distance: int = 4,
) -> Tuple[List[RetinalDatasetRecord], List[RetinalDatasetRecord]]:
    """Deduplicate dataset records using 64-bit perceptual hashes.

    Args:
        records: Input list of RetinalDatasetRecord (each must have `phash` populated).
        max_hamming_distance: Maximum Hamming distance to consider two scans near-duplicates.

    Returns:
        Tuple of (clean_unique_records, removed_duplicate_records).
    """
    unique_records: List[RetinalDatasetRecord] = []
    removed_duplicates: List[RetinalDatasetRecord] = []

    for rec in records:
        if rec.phash is None:
            unique_records.append(rec)
            continue
        is_dup = False
        for kept in unique_records:
            if kept.phash is not None:
                if hamming_distance_64bit(rec.phash, kept.phash) <= max_hamming_distance:
                    is_dup = True
                    break
        if is_dup:
            removed_duplicates.append(rec)
        else:
            unique_records.append(rec)

    return unique_records, removed_duplicates


def create_patient_stratified_splits(
    records: List[RetinalDatasetRecord],
    n_splits: int = 5,
    test_fold: int = 0,
    seed: int = 42,
) -> Dict[str, List[RetinalDatasetRecord]]:
    """Partition records into cross-validation folds and an untouched test split.

    Guarantees:
    1. All scans from the same `patient_id` (both OD and OS) stay in the same fold.
    2. Augmented records (`is_augmented=True`) are strictly barred from validation and test splits.
    3. Stratification preserves natural DR severity class distribution across folds.
    """
    if len(records) < n_splits:
        raise ValueError(f"Need at least {n_splits} records to create {n_splits} folds.")

    # Separate any pre-existing augmented records so they never enter val/test
    unaugmented = [r for r in records if not r.is_augmented]
    augmented = [r for r in records if r.is_augmented]

    labels = np.array([r.dr_grade for r in unaugmented], dtype=np.int64)
    groups = np.array([r.patient_id for r in unaugmented])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_x = np.zeros(len(unaugmented))

    for fold_idx, (_, val_indices) in enumerate(sgkf.split(dummy_x, labels, groups)):
        for idx in val_indices:
            unaugmented[idx].fold = fold_idx

    test_records = [r for r in unaugmented if r.fold == test_fold]
    test_patient_ids = {r.patient_id for r in test_records}

    train_val_records = [r for r in unaugmented if r.fold != test_fold]
    # Only attach augmented records whose source patient is strictly inside train_val_records
    safe_augmented = [
        r for r in augmented if r.patient_id not in test_patient_ids
    ]

    return {
        "train_val": train_val_records,
        "test": test_records,
        "augmented_train_only": safe_augmented,
    }


def group_bilateral_pairs(
    records: List[RetinalDatasetRecord],
) -> List[BilateralPatientPair]:
    """Group eye-level records by `patient_id` into BilateralPatientPair objects."""
    by_patient: Dict[str, BilateralPatientPair] = {}
    for rec in records:
        if rec.patient_id not in by_patient:
            by_patient[rec.patient_id] = BilateralPatientPair(patient_id=rec.patient_id)
        pair = by_patient[rec.patient_id]
        if rec.laterality.upper() == "OD":
            pair.od_record = rec
        else:
            pair.os_record = rec
    return list(by_patient.values())


class PreprocessedManifestDataset:
    """PyTorch-compatible dataset reading preprocessed 512/768 scans from CSV manifests."""

    def __init__(
        self,
        manifest_csv: str,
        split: Optional[str] = "train_val",
        exclude_fold: Optional[int] = None,
        only_fold: Optional[int] = None,
        resolution: int = 512,
        data_root_override: Optional[str] = None,
        augment: bool = False,
    ) -> None:
        import csv
        from pathlib import Path

        self.manifest_dir = Path(manifest_csv).resolve().parent
        self.resolution = resolution
        self.data_root_override = Path(data_root_override) if data_root_override else None
        self.augment = augment
        self.rows: List[Dict[str, str]] = []

        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split and row.get("split") != split:
                    continue
                fold_val = int(row.get("fold", -1))
                if exclude_fold is not None and fold_val == exclude_fold:
                    continue
                if only_fold is not None and fold_val != only_fold:
                    continue
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        from pathlib import Path
        import torch

        row = self.rows[idx]
        col = "path_768" if (self.resolution == 768 and row.get("path_768")) else "path_512"
        raw_path_str = row.get(col) or row.get("path_512") or row.get("image_path", "")
        img_path = Path(raw_path_str)
        if self.data_root_override is not None:
            img_path = self.data_root_override / img_path.parent.name / img_path.name
        elif not img_path.exists():
            candidate = self.manifest_dir / img_path.parent.name / img_path.name
            if candidate.exists():
                img_path = candidate

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Preprocessed image not found at {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != self.resolution or rgb.shape[1] != self.resolution:
            rgb = cv2.resize(rgb, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            # Lossless D4 dihedral group transforms preserve <10px microaneurysms without interpolation blur
            if np.random.rand() < 0.5:
                rgb = np.fliplr(rgb)
            if np.random.rand() < 0.5:
                rgb = np.flipud(rgb)
            k_rot = int(np.random.randint(0, 4))
            if k_rot > 0:
                rgb = np.rot90(rgb, k=k_rot)
            rgb = np.ascontiguousarray(rgb)

        tensor_chw = torch.from_numpy(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)))
        dr_grade = int(row.get("dr_grade", 0))
        glaucoma = int(row.get("glaucoma", 0))
        cataract = int(row.get("cataract", 0))
        multi_label = torch.tensor(
            [1.0 if dr_grade > 0 else 0.0, float(glaucoma), float(cataract)],
            dtype=torch.float32,
        )

        return {
            "image": tensor_chw,
            "dr_grade": torch.tensor(dr_grade, dtype=torch.long),
            "referable_dr": torch.tensor([1.0 if dr_grade >= 2 else 0.0], dtype=torch.float32),
            "multi_label": multi_label,
            "image_id": row.get("image_id", ""),
            "patient_id": row.get("patient_id", ""),
        }

