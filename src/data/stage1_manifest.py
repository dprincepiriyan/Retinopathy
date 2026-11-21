"""Build a Stage-1 triage manifest that breaks the dataset-source shortcut.

Problem: in `stage1_triage_splits.csv` every DR-positive scan comes from EyePACS and
every DR-negative scan from ODIR, so "is DR present?" is perfectly predictable from
camera/dataset style alone. A model can score DR AUC ~1.0 without learning any
pathology.

Fix (with the data already on disk): add the APTOS-2019 Stage-2 scans to the Stage-1
pool. APTOS contributes *both* DR-negative (grade 0) and DR-positive (grade >= 1)
images from the same source, so source no longer determines the DR label.

Labels that are genuinely unknown are masked instead of being assumed negative:

=============  ======  ==========  ==========
source         DR      glaucoma    cataract
=============  ======  ==========  ==========
odir_s1        known   known       known
eyepacs_s1     known   unknown     unknown
aptos2019      known   unknown     unknown
=============  ======  ==========  ==========

Fold / split assignments are copied from the source manifests, so APTOS fold-0 scans
remain in the locked test split for both stages (no cross-stage leakage).
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

MASK_COLUMNS = ("mask_dr", "mask_glaucoma", "mask_cataract")

# Which labels are trustworthy per source (dr, glaucoma, cataract)
SOURCE_LABEL_MASKS: Dict[str, tuple[int, int, int]] = {
    "odir_s1": (1, 1, 1),
    "eyepacs_s1": (1, 0, 0),
    "aptos2019_highres": (1, 0, 0),
}
DEFAULT_MASK = (1, 1, 1)

COMBINED_MANIFEST_NAME = "stage1_triage_combined_splits.csv"


def _read(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_stage1_combined_manifest(
    stage1_csv: str | Path,
    stage2_csv: str | Path,
    out_csv: str | Path,
    include_aptos: bool = True,
) -> Dict[str, object]:
    """Write the combined Stage-1 manifest and return a summary dict."""
    rows = [dict(r) for r in _read(Path(stage1_csv))]
    if include_aptos and Path(stage2_csv).exists():
        for r in _read(Path(stage2_csv)):
            r = dict(r)
            r["stage_role"] = "stage1_triage_aptos"
            r["path_768"] = ""  # Stage-1 trains at 512 px only
            rows.append(r)

    for r in rows:
        mask = SOURCE_LABEL_MASKS.get(r.get("source_dataset", ""), DEFAULT_MASK)
        for col, val in zip(MASK_COLUMNS, mask):
            r[col] = str(val)

    # Safety: no patient may be in both the locked test split and train_val.
    split_by_patient: Dict[str, set] = defaultdict(set)
    for r in rows:
        split_by_patient[r["patient_id"]].add(r.get("split", ""))
    leaking = [pid for pid, splits in split_by_patient.items() if len(splits) > 1]
    if leaking:
        raise RuntimeError(f"Patient leakage between splits for {len(leaking)} patients, e.g. {leaking[:3]}")

    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dr_by_source = Counter((r["source_dataset"], int(r.get("dr_grade", 0)) > 0) for r in rows)
    return {
        "out_csv": str(out),
        "n_rows": len(rows),
        "dr_label_by_source": {f"{src}|dr={int(pos)}": n for (src, pos), n in sorted(dr_by_source.items())},
    }


def ensure_stage1_combined_manifest(data_root: str | Path, include_aptos: bool = True) -> Path:
    """Return the combined manifest path inside `data_root`, building it if missing."""
    root = Path(data_root)
    out = root / COMBINED_MANIFEST_NAME
    if not out.exists():
        build_stage1_combined_manifest(
            root / "stage1_triage_splits.csv",
            root / "stage2_grading_splits.csv",
            out,
            include_aptos=include_aptos,
        )
    return out
