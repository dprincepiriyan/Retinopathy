"""Tests for Phase 1 Data Curation, Perceptual Hash Deduplication & Zero Patient Leakage."""

import numpy as np
import pytest

from app.assets.synthetic_fundus import generate_synthetic_fundus_image
from src.data.datasets import (
    RetinalDatasetRecord,
    compute_perceptual_hash,
    create_patient_stratified_splits,
    deduplicate_records_by_phash,
    group_bilateral_pairs,
    hamming_distance_64bit,
)


def test_perceptual_hash_deduplication_catches_near_duplicates() -> None:
    img_a = generate_synthetic_fundus_image(size=256, laterality="OD", num_microaneurysms=5, seed=101)
    # Near-duplicate: minor brightness shift (+2)
    img_a_dup = np.clip(img_a.astype(np.int32) + 2, 0, 255).astype(np.uint8)
    # Distinct patient scan
    img_b = generate_synthetic_fundus_image(
        size=256, laterality="OS", num_hemorrhages=8, num_exudates=6, vcdr=0.65, seed=202
    )

    hash_a = compute_perceptual_hash(img_a)
    hash_a_dup = compute_perceptual_hash(img_a_dup)
    hash_b = compute_perceptual_hash(img_b)

    assert hamming_distance_64bit(hash_a, hash_a_dup) <= 2
    assert hamming_distance_64bit(hash_a, hash_b) > 4

    records = [
        RetinalDatasetRecord(image_id="scan_1", patient_id="P1", laterality="OD", dr_grade=1, phash=hash_a),
        RetinalDatasetRecord(image_id="scan_1_copy", patient_id="P99", laterality="OD", dr_grade=1, phash=hash_a_dup),
        RetinalDatasetRecord(image_id="scan_2", patient_id="P2", laterality="OS", dr_grade=3, phash=hash_b),
    ]
    unique, removed = deduplicate_records_by_phash(records, max_hamming_distance=4)
    assert len(unique) == 2
    assert len(removed) == 1
    assert removed[0].image_id == "scan_1_copy"


def test_zero_patient_overlap_and_no_augmented_test_leakage() -> None:
    records = []
    # Create 40 patients, each with bilateral OD and OS scans across grades 0-4
    for p_idx in range(40):
        pid = f"PATIENT_{p_idx:03d}"
        grade = p_idx % 5
        records.append(
            RetinalDatasetRecord(image_id=f"{pid}_OD", patient_id=pid, laterality="OD", dr_grade=grade)
        )
        records.append(
            RetinalDatasetRecord(image_id=f"{pid}_OS", patient_id=pid, laterality="OS", dr_grade=grade)
        )
        # Add an augmented copy for minority class patients
        if grade >= 3:
            records.append(
                RetinalDatasetRecord(
                    image_id=f"{pid}_OD_aug",
                    patient_id=pid,
                    laterality="OD",
                    dr_grade=grade,
                    is_augmented=True,
                )
            )

    splits = create_patient_stratified_splits(records, n_splits=5, test_fold=0, seed=42)
    train_val = splits["train_val"]
    test = splits["test"]
    aug_train = splits["augmented_train_only"]

    train_val_pids = {r.patient_id for r in train_val}
    test_pids = {r.patient_id for r in test}
    aug_pids = {r.patient_id for r in aug_train}

    # 1. Zero patient overlap between train_val and test
    assert train_val_pids.isdisjoint(test_pids)

    # 2. Zero patient overlap across individual CV folds
    for f1 in range(5):
        for f2 in range(f1 + 1, 5):
            pids_f1 = {r.patient_id for r in (train_val + test) if r.fold == f1}
            pids_f2 = {r.patient_id for r in (train_val + test) if r.fold == f2}
            assert pids_f1.isdisjoint(pids_f2)

    # 3. No augmented sample ever enters test split or shares a patient_id with test split
    assert all(not r.is_augmented for r in test)
    assert aug_pids.isdisjoint(test_pids)


def test_bilateral_patient_worse_eye_aggregation() -> None:
    records = [
        RetinalDatasetRecord(image_id="P10_OD", patient_id="P10", laterality="OD", dr_grade=1),
        RetinalDatasetRecord(image_id="P10_OS", patient_id="P10", laterality="OS", dr_grade=3),
    ]
    pairs = group_bilateral_pairs(records)
    assert len(pairs) == 1
    assert pairs[0].patient_worse_dr_grade == 3
    assert pairs[0].is_referable_dr is True


def test_dataset_cleaning_pipeline_end_to_end(tmp_path) -> None:
    import cv2
    from src.data.dataset_cleaner_cli import DatasetCleaningPipeline

    raw_dir = tmp_path / "raw_dataset"
    for grade in [0, 2]:
        (raw_dir / str(grade)).mkdir(parents=True, exist_ok=True)
        for p_idx in range(3):
            img = generate_synthetic_fundus_image(
                size=256, laterality="OD", num_microaneurysms=grade * 3, seed=grade * 100 + p_idx
            )
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(raw_dir / str(grade) / f"pt{grade}_{p_idx}_right.png"), bgr)

    # Also write 1 blurry ungradable image and 1 exact duplicate
    blur_img = generate_synthetic_fundus_image(size=256, simulate_severe_blur=True, seed=999)
    cv2.imwrite(str(raw_dir / "0" / "blurry_bad_left.png"), cv2.cvtColor(blur_img, cv2.COLOR_RGB2BGR))

    dup_img = generate_synthetic_fundus_image(size=256, laterality="OD", num_microaneurysms=0, seed=0)
    cv2.imwrite(str(raw_dir / "0" / "duplicate_copy_right.png"), cv2.cvtColor(dup_img, cv2.COLOR_RGB2BGR))

    cleaner = DatasetCleaningPipeline(stage1_size=128, stage2_size=128, n_splits=2)
    discovered = cleaner.discover_folder_dataset(raw_dir, source_name="test_ds")
    assert len(discovered) == 8

    out_dir = tmp_path / "cleaned_out"
    summary, splits = cleaner.clean_and_split_records(discovered, output_dir=out_dir)
    assert summary.total_scanned == 8
    assert summary.rejected_by_iqa == 1
    assert summary.duplicates_removed == 1
    assert summary.clean_unique_gradable == 6
    assert summary.manifest_csv_path.exists()

