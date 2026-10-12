"""Atomic per-job records and strict run identity for sequential evaluation."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def json_value(value: Any) -> Any:
    """Convert diagnostic NumPy values; nonfinite diagnostics become explicit nulls."""
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported result value: {type(value).__name__}.")


def canonical_json(value: Any) -> str:
    return json.dumps(json_value(value), sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    # A killed process leaves either the old complete file or the new complete file.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".pending-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            import os
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ResultStore:
    """One writer per directory; completed records survive interruptions.

    A different dataset, fold, configuration, source fingerprint or library
    version must use a different directory. Corrupt records fail loudly.
    """

    def __init__(self, directory: str | Path, manifest: dict[str, Any]):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        expected = json_value(manifest)
        self.run_id = content_hash(expected)
        manifest_path = self.directory / "manifest.json"
        envelope = {"run_id": self.run_id, "manifest": expected}
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != envelope:
                raise ValueError("Result directory belongs to a different or corrupt run; use a new directory.")
        else:
            if any(self.directory.iterdir()):
                raise ValueError("Result directory is nonempty without a manifest.")
            _atomic_json(manifest_path, envelope)

    def _path(self, job: dict[str, Any]) -> Path:
        return self.directory / f"job-{content_hash(job)}.json"

    def read(self, job: dict[str, Any]) -> dict[str, Any] | None:
        path = self._path(job)
        if not path.exists():
            return None
        envelope = json.loads(path.read_text(encoding="utf-8"))
        record = envelope.get("record")
        if (not isinstance(record, dict) or envelope.get("sha256") != content_hash(record)
                or record.get("run_id") != self.run_id or record.get("job") != json_value(job)
                or record.get("status") not in {"success", "failed", "unavailable"}):
            raise ValueError(f"Invalid/corrupt result record: {path.name}.")
        return record

    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        result = json_value({**record, "run_id": self.run_id})
        if result.get("status") not in {"success", "failed", "unavailable"}:
            raise ValueError("Only terminal evaluation records can be saved.")
        _atomic_json(self._path(result["job"]), {"sha256": content_hash(result), "record": result})
        return result
