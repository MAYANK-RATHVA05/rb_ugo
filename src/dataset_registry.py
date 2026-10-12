"""Provenance and availability checks for historical benchmark candidates.

No candidate is asserted to belong to the paper's missing 74-dataset supplement.
Raw rows are never stored in or written back to the tracked registry.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from src.config import PROJECT_ROOT, RAW_DATA_DIR
from src.data_loader import DatasetError, load_dataset


REGISTRY_PATH = PROJECT_ROOT / "data" / "manifests" / "benchmark_candidates.json"


@dataclass(frozen=True)
class CandidateCheck:
    id: str
    status: str
    ugo_paper_membership: str
    reason: str


def load_candidate_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Read a manifest of historical evidence, rejecting fabricated membership."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("candidates"), list):
        raise ValueError("Unsupported benchmark candidate registry schema.")
    seen: set[str] = set()
    for entry in document["candidates"]:
        name = entry.get("id")
        filename = entry.get("filename")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or name in seen:
            raise ValueError("Candidate IDs must be unique safe lowercase names.")
        seen.add(name)
        if not isinstance(filename, str) or filename != Path(filename).name or not filename.endswith(".dat"):
            raise ValueError("Candidate filename must be a local .dat basename.")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Candidate SHA-256 must have 64 lowercase hex digits.")
        archive_digest = entry.get("archive_sha256")
        if archive_digest is not None and (not isinstance(archive_digest, str)
                                          or not re.fullmatch(r"[0-9a-f]{64}", archive_digest)):
            raise ValueError("Candidate archive SHA-256 must have 64 lowercase hex digits.")
        if entry.get("ugo_paper_membership") != "unverified":
            raise ValueError("This candidate registry cannot assert UGO paper membership.")
        if any(not isinstance(entry.get(field), int) or isinstance(entry[field], bool) or entry[field] < 0
               for field in ("rows", "features", "majority_count", "minority_count", "official_fold_pairs")):
            raise ValueError("Candidate counts must be nonnegative integers.")
        if entry["rows"] != entry["majority_count"] + entry["minority_count"]:
            raise ValueError("Candidate class counts must sum to rows.")
    return document


def check_candidate(entry: dict[str, Any], raw_dir: Path = RAW_DATA_DIR) -> CandidateCheck:
    """Inspect only a placed local raw file, guarded by its historical hash."""
    filename = entry["filename"]
    if filename != Path(filename).name or not filename.endswith(".dat"):
        raise ValueError("Candidate filename must be a local .dat basename.")
    path = Path(raw_dir) / filename
    if not path.is_file():
        return CandidateCheck(entry["id"], "raw_file_absent", "unverified", "Historical manifest only; no local raw file.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        return CandidateCheck(entry["id"], "sha256_mismatch", "unverified", "Local file differs from historically inspected bytes.")
    try:
        dataset = load_dataset(path)
    except (DatasetError, UnicodeError) as exc:
        return CandidateCheck(entry["id"], "load_failed", "unverified", str(exc))
    expected = (entry["rows"], entry["features"], entry["majority_count"],
                entry["minority_count"], entry["majority_label"],
                entry["minority_label"], entry["target"],
                tuple(entry["categorical_features"]))
    observed = (len(dataset.X), len(dataset.feature_names), dataset.majority_count,
                dataset.minority_count, dataset.majority_label,
                dataset.minority_label, dataset.target_name,
                dataset.categorical_feature_names)
    if observed != expected:
        return CandidateCheck(entry["id"], "metadata_mismatch", "unverified", "Parsed rows, labels or types disagree with historical inspection.")
    return CandidateCheck(entry["id"], "historical_bytes_verified", "unverified", "Matches historical inspection; paper membership and official folds remain unverified.")


def check_all_candidates(raw_dir: Path = RAW_DATA_DIR,
                         registry_path: Path = REGISTRY_PATH) -> list[CandidateCheck]:
    return [check_candidate(entry, raw_dir) for entry in load_candidate_registry(registry_path)["candidates"]]
