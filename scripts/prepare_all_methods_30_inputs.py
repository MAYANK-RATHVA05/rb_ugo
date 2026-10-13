"""Rebuild the fixed 30-study inputs from public source archives on disk.

Only the four original KEEL section ZIPs and the UCI Landsat ZIP must be
supplied. The two R data objects are read from source-dir when present, or
downloaded from pinned CRAN GitHub revisions. Every input is checked against
the committed SHA-256 registry before it is written. No models are fitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.request import urlopen
import zipfile

import numpy as np

from src.data_loader import _split_keel


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "data/manifests/all_methods_30.json"
REGISTRY = ROOT / "data/manifests/ugo_74_approximate.json"
KEEL_ARCHIVES = {
    "imb_IRlowerThan9.zip": "73376ffd570b54187fa0db72b85ef19ceb08b45ddd4e920edef466e6b957dcb3",
    "imb_IRhigherThan9p1.zip": "ea3fb6cca16e033ad71165eed78862ede9168df8c1c67841d64d09af0dcd5135",
    "imb_IRhigherThan9p2.zip": "ac8eba996bb891c996d020e81cd45f7938baa70b3d4404908fca1a1a88afae34",
    "imb_IRhigherThan9p3.zip": "80e9004707e09a8c333ab80bac380254f31aeeb515b93e21c3a6a1147dc4f040",
}
SATELLITE_ARCHIVE_SHA = "7c54e0e11c872a1b0b647da370d596dcb06746159cce4121d92ccd70b7d7ce3c"
R_SOURCES = {
    "banana": "https://raw.githubusercontent.com/cran/imbalance/dca75c7c3e5e7a4f1da30b74508bdd53ec93c61a/data/banana.rda",
    "abalone_rings_ge_18_vs_rest": "https://raw.githubusercontent.com/cran/AppliedPredictiveModeling/5505adba3a6d245ca4e2d84ee6f9d5150747a29e/data/abalone.RData",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def registered_inputs() -> list[dict]:
    names = json.loads(SELECTION.read_text(encoding="utf-8"))["datasets"]
    registry = {entry["dataset_id"]: entry for entry in
                json.loads(REGISTRY.read_text(encoding="utf-8"))["datasets"]}
    if len(names) != 30 or len(set(names)) != 30:
        raise ValueError("Expected exactly 30 distinct selected inputs.")
    return [registry[name] for name in names]


def locate_archives(source_dir: Path, required: set[str]) -> dict[str, Path]:
    """Match archives by bytes, allowing renamed downloads such as '(1)'."""
    wanted = {**KEEL_ARCHIVES,
              "statlog+landsat+satellite.zip": SATELLITE_ARCHIVE_SHA}
    located = {}
    for path in source_dir.rglob("*.zip"):
        if not (path.name.startswith("imb_IR") or
                path.name.startswith("statlog+landsat+satellite")):
            continue
        archive_hash = digest(path.read_bytes())
        for name, expected_hash in wanted.items():
            if name in required and archive_hash == expected_hash:
                located[name] = path
    missing = required - located.keys()
    if missing:
        raise FileNotFoundError(
            f"Missing source archives under {source_dir}: {', '.join(sorted(missing))}. "
            "Use the four KEEL imbalanced section ZIPs and the UCI Landsat ZIP."
        )
    return located


def original_keel_members(paths: dict[str, Path], expected_hashes: set[str]) -> dict[str, bytes]:
    """Read only original .dat members inside each dataset's nested ZIP."""
    found = {}
    for archive_name in KEEL_ARCHIVES:
        if archive_name not in paths:
            continue
        with zipfile.ZipFile(paths[archive_name]) as outer:
            for info in outer.infolist():
                if (info.is_dir() or not info.filename.lower().endswith(".zip")
                        or "-5-fold" in info.filename or "__MACOSX" in info.filename):
                    continue
                with zipfile.ZipFile(io.BytesIO(outer.read(info))) as nested:
                    members = [member for member in nested.infolist()
                               if not member.is_dir() and member.filename.lower().endswith(".dat")
                               and "__MACOSX" not in member.filename]
                    if len(members) != 1:
                        raise ValueError(f"Expected one original .dat in {info.filename}.")
                    data = nested.read(members[0])
                    key = digest(data)
                    if key in expected_hashes:
                        found[key] = data
    missing = expected_hashes - found.keys()
    if missing:
        raise ValueError(f"The KEEL archives do not contain {len(missing)} registered originals.")
    return found


def normalize_keel_header(data: bytes) -> bytes:
    """Apply only the audited loader-compatibility header fixes."""
    lines = []
    in_data = False
    for line in data.decode("utf-8-sig").splitlines(keepends=True):
        if not in_data:
            line = re.sub(r"^@(input|output)(?=\s)", r"@\1s", line, flags=re.I)
            if re.match(r"^@inputs\s", line, re.I):
                tokens = _split_keel(line.split(None, 1)[1].strip())
                if "" in tokens:
                    line = "@inputs " + ",".join(token for token in tokens if token) + "\n"
            if line.strip().lower() == "@data":
                in_data = True
        lines.append(line)
    return "".join(lines).encode("utf-8")


def satellite_variant(archive_path: Path) -> bytes:
    with zipfile.ZipFile(archive_path) as archive:
        rows = []
        for member in ("sat.trn", "sat.tst"):
            for line in archive.read(member).decode("utf-8-sig").splitlines():
                if line.strip():
                    row = line.split()
                    if len(row) != 37:
                        raise ValueError(f"Unexpected Landsat row width in {member}.")
                    rows.append(row)
    if len(rows) != 6435:
        raise ValueError("Unexpected Landsat row count.")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([f"feature_{i:02d}" for i in range(1, 37)] + ["class"])
    writer.writerows(row[:36] + ["1" if row[-1] == "4" else "0"] for row in rows)
    return output.getvalue().encode("utf-8")


def r_source(name: str, entry: dict, source_dir: Path, cache: Path) -> Path:
    expected = entry["provenance"]["original_source_sha256"]
    for path in source_dir.rglob(entry["provenance"]["original_source_filename"]):
        if digest(path.read_bytes()) == expected:
            return path
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / entry["provenance"]["original_source_filename"]
    if not path.exists() or digest(path.read_bytes()) != expected:
        with urlopen(R_SOURCES[name], timeout=30) as response:
            body = response.read()
        if digest(body) != expected:
            raise ValueError(f"Downloaded {name} R data differs from the registered source.")
        path.write_bytes(body)
    return path


def r_variant(name: str, path: Path) -> bytes:
    try:
        import pyreadr
    except ImportError as exc:
        raise RuntimeError("Install pyreadr first: python -m pip install pyreadr==0.5.7") from exc
    if name == "banana":
        frame = pyreadr.read_r(str(path))["banana"]
    else:
        source = pyreadr.read_r(str(path))["abalone"]
        numeric = list(source.select_dtypes("number").drop(columns="Rings").columns)
        frame = source[numeric].copy()
        frame["class"] = np.where(source.Rings >= 18, "positive", "negative")
    return frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def prepare(source_dir: Path, output_root: Path, *, cache: Path) -> int:
    entries = registered_inputs()
    pending = [entry for entry in entries if not (output_root / entry["relative_path"]).exists()]
    for entry in entries:
        path = output_root / entry["relative_path"]
        if path.exists() and digest(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"Existing {path} differs from the registered SHA-256; left unchanged.")
    needed_keel = {entry["provenance"]["original_source_sha256"] for entry in pending
                   if entry["dataset_id"] not in R_SOURCES
                   and entry["dataset_id"] != "satellite_class_4_vs_rest"}
    required = set(KEEL_ARCHIVES) if needed_keel else set()
    if any(entry["dataset_id"] == "satellite_class_4_vs_rest" for entry in pending):
        required.add("statlog+landsat+satellite.zip")
    archives = locate_archives(source_dir, required) if required else {}
    originals = original_keel_members(archives, needed_keel) if needed_keel else {}
    staged = []
    for entry in pending:
        name = entry["dataset_id"]
        if name in R_SOURCES:
            body = r_variant(name, r_source(name, entry, source_dir, cache))
        elif name == "satellite_class_4_vs_rest":
            body = satellite_variant(archives["statlog+landsat+satellite.zip"])
        else:
            original = originals[entry["provenance"]["original_source_sha256"]]
            body = original if digest(original) == entry["sha256"] else normalize_keel_header(original)
        if digest(body) != entry["sha256"]:
            raise ValueError(f"Prepared {name} differs from registered SHA-256; nothing new was written.")
        staged.append((output_root / entry["relative_path"], body))
    for path, body in staged:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        print(f"Prepared {path.name}")
    print(f"Verified {len(entries)} inputs ({len(staged)} newly prepared).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Folder containing your KEEL and Landsat ZIP downloads.")
    parser.add_argument("--output-root", type=Path, default=ROOT,
                        help="Repository root; override only for validation.")
    args = parser.parse_args()
    if not args.source_dir.is_dir():
        parser.error(f"Source directory does not exist: {args.source_dir}")
    return prepare(args.source_dir, args.output_root,
                   cache=args.output_root / "data/raw/source_cache")


if __name__ == "__main__":
    raise SystemExit(main())
