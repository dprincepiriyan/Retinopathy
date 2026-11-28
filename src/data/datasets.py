"""Multi-dataset loader, perceptual hash deduplication, and zero-leakage patient splitter.

Addresses Bottleneck 7 from the forensic paper audit:
- Eliminates cross-dataset duplicate images via 64-bit DCT perceptual hashing (pHash).
- Partitions strictly by `patient_id` using StratifiedGroupKFold so Left (OS) and
  Right (OD) eyes of the same individual never leak across train/val/test splits.
- Pairs bilateral scans (OD & OS) for patient-level worse-eye referral aggregation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
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


_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _field_get(item, key: str, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _field_set(item, key: str, value) -> None:
    if isinstance(item, dict):
        item[key] = value
    else:
        setattr(item, key, value)


def deduplicate_records_by_phash(
    records: List,
    max_hamming_distance: int = 2,
    delete_duplicate_files: bool = False,
) -> Tuple[List, List]:
    """Deduplicate records (`RetinalDatasetRecord` or `dict`) by 64-bit pHash.

    Uses a vectorized NumPy XOR + byte popcount pass over kept hashes. Matches
    within `max_hamming_distance` bits merge comorbidity labels (max of
    `dr_grade`, `glaucoma`, `cataract`) onto the kept record and, when
    `delete_duplicate_files=True`, remove redundant duplicate files on disk
    (`image_path`, `path_512`, `path_768`).
    """
    unique_records: List = []
    removed_duplicates: List = []
    kept_hashes = np.empty(len(records), dtype=np.uint64)
    kept_indices = np.empty(len(records), dtype=np.int64)
    n_kept = 0

    for rec in records:
        ph = _field_get(rec, "phash")
        if ph is None:
            unique_records.append(rec)
            continue

        ph_u64 = np.uint64(int(ph))
        matched = None
        if n_kept:
            xor = kept_hashes[:n_kept] ^ ph_u64
            dists = _POPCOUNT_LUT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
            hits = np.flatnonzero(dists <= max_hamming_distance)
            if hits.size:
                matched = unique_records[int(kept_indices[hits[0]])]

        if matched is not None:
            for field_name in ("dr_grade", "glaucoma", "cataract"):
                _field_set(
                    matched,
                    field_name,
                    max(_field_get(matched, field_name, 0), _field_get(rec, field_name, 0)),
                )
            if delete_duplicate_files:
                for p_key in ("image_path", "path_512", "path_768"):
                    p_str = _field_get(rec, p_key, "")
                    if p_str and p_str != _field_get(matched, p_key, ""):
                        Path(p_str).unlink(missing_ok=True)
            removed_duplicates.append(rec)
        else:
            kept_hashes[n_kept] = ph_u64
            kept_indices[n_kept] = len(unique_records)
            n_kept += 1
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


STAGE1_LABEL_MASK_COLUMNS = ("mask_dr", "mask_glaucoma", "mask_cataract")


class PreprocessedManifestDataset:
    """PyTorch-compatible dataset reading preprocessed 512/768 scans from CSV manifests.

    `augment` may be False (no augmentation), True (default `RetinalTrainAugment`)
    or any callable mapping a uint8 RGB array to an augmented uint8 RGB array.

    Optional manifest columns `mask_dr`, `mask_glaucoma`, `mask_cataract` (0/1) mark
    which Stage-1 labels are actually known for a row; they are returned as
    `multi_label_mask` so the loss can ignore unknown labels instead of treating
    them as negatives.
    """

    def __init__(
        self,
        manifest_csv: str,
        split: Optional[str] = "train_val",
        exclude_fold: Optional[int] = None,
        only_fold: Optional[int] = None,
        resolution: int = 512,
        data_root_override: Optional[str] = None,
        augment: Union[bool, Callable[[np.ndarray], np.ndarray]] = False,
    ) -> None:
        import csv
        from pathlib import Path

        self.manifest_dir = Path(manifest_csv).resolve().parent
        self.resolution = resolution
        self.data_root_override = Path(data_root_override) if data_root_override else None
        if augment is True:
            from src.data.augmentations import RetinalTrainAugment

            self.augment_fn: Optional[Callable[[np.ndarray], np.ndarray]] = RetinalTrainAugment("medium")
        elif callable(augment):
            self.augment_fn = augment
        else:
            self.augment_fn = None
        self.augment = self.augment_fn is not None
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

    def dr_grades(self) -> np.ndarray:
        """Integer DR grades of all rows (used for class-balanced sampling)."""
        return np.array([int(r.get("dr_grade", 0)) for r in self.rows], dtype=np.int64)

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

        if self.augment_fn is not None:
            rgb = self.augment_fn(rgb)

        tensor_chw = torch.from_numpy(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)))
        dr_grade = int(row.get("dr_grade", 0))
        glaucoma = int(row.get("glaucoma", 0))
        cataract = int(row.get("cataract", 0))
        multi_label = torch.tensor(
            [1.0 if dr_grade > 0 else 0.0, float(glaucoma), float(cataract)],
            dtype=torch.float32,
        )

        label_mask = torch.tensor(
            [float(row.get(col, "1") or 1) for col in STAGE1_LABEL_MASK_COLUMNS],
            dtype=torch.float32,
        )

        return {
            "image": tensor_chw,
            "multi_label_mask": label_mask,
            "source_dataset": row.get("source_dataset", ""),
            "dr_grade": torch.tensor(dr_grade, dtype=torch.long),
            "referable_dr": torch.tensor([1.0 if dr_grade >= 2 else 0.0], dtype=torch.float32),
            "multi_label": multi_label,
            "image_id": row.get("image_id", ""),
            "patient_id": row.get("patient_id", ""),
        }

