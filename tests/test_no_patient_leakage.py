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


def test_perceptual_hash_deduplication_catches_near_duplicates(tmp_path) -> None:
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

    kept_file = tmp_path / "scan_1.jpg"
    dup_file = tmp_path / "scan_1_copy.jpg"
    kept_file.write_bytes(b"a")
    dup_file.write_bytes(b"b")

    records = [
        RetinalDatasetRecord(
            image_id="scan_1", patient_id="P1", laterality="OD", dr_grade=1, glaucoma=1,
            image_path=str(kept_file), phash=hash_a,
        ),
        RetinalDatasetRecord(
            image_id="scan_1_copy", patient_id="P99", laterality="OD", dr_grade=2, cataract=1,
            image_path=str(dup_file), phash=hash_a_dup,
        ),
        RetinalDatasetRecord(image_id="scan_2", patient_id="P2", laterality="OS", dr_grade=3, phash=hash_b),
    ]
    unique, removed = deduplicate_records_by_phash(records, delete_duplicate_files=True)
    assert len(unique) == 2
    assert len(removed) == 1
    assert removed[0].image_id == "scan_1_copy"
    # Comorbidity labels merge onto kept record; duplicate file is removed on disk
    assert (unique[0].dr_grade, unique[0].glaucoma, unique[0].cataract) == (2, 1, 1)
    assert kept_file.exists() and not dup_file.exists()

    # Default 2-bit rule keeps 3-bit-distant hashes separate (incl. high-bit uint64 values)
    hi = (1 << 63) | 0b10101
    p1, p2 = tmp_path / "d1.jpg", tmp_path / "d2.jpg"
    p1.write_bytes(b"1")
    p2.write_bytes(b"2")
    dict_rows = [
        {"image_id": "a", "phash": hi, "dr_grade": 0, "glaucoma": 1, "cataract": 0, "path_512": str(p1), "path_768": ""},
        {"image_id": "b", "phash": hi ^ 0b11, "dr_grade": 0, "glaucoma": 0, "cataract": 1, "path_512": str(p2), "path_768": ""},
        {"image_id": "c", "phash": hi ^ 0b111, "dr_grade": 0, "glaucoma": 0, "cataract": 0, "path_512": "", "path_768": ""},
    ]
    u_rows, r_rows = deduplicate_records_by_phash(dict_rows, delete_duplicate_files=True)
    assert [r["image_id"] for r in u_rows] == ["a", "c"]
    assert [r["image_id"] for r in r_rows] == ["b"]
    assert (u_rows[0]["glaucoma"], u_rows[0]["cataract"]) == (1, 1)
    assert p1.exists() and not p2.exists()


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


def test_cached_iqa_score_is_numeric_so_mixed_rows_sort(tmp_path) -> None:
    from src.data.dataset_cleaner_cli import _read_iqa_sidecar, _write_iqa_sidecar

    cached = str(tmp_path / "a.jpg")
    _write_iqa_sidecar(cached, 0.8123456)
    score, _ = _read_iqa_sidecar(cached)
    assert score == 0.8123
    missing, _ = _read_iqa_sidecar(str(tmp_path / "no_sidecar.jpg"))
    rows = [{"quality_score": 0.9}, {"quality_score": score}, {"quality_score": missing}]  # new + cached + unknown
    assert [r["quality_score"] for r in sorted(rows, key=lambda x: float(x["quality_score"] or 0))] == [missing, score, 0.9]
