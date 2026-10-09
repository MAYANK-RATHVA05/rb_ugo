"""Strict binary CSV/KEEL loading without preprocessing or data splitting.

Original labels are source text, including numeric-looking labels. Only feature
columns are converted to numeric types. Missing features and duplicates remain.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


class DatasetError(ValueError):
    """A dataset violates the supported format or binary-data contract."""


@dataclass(frozen=True)
class LoadedDataset:
    """Data and metadata, with aligned y and original_y Series.

    label_mapping maps original text labels to integers: majority=0, minority=1.
    X preserves numeric/categorical columns and missing values. Feature names
    follow source order; source_filename is the resolved source path.
    """

    X: pd.DataFrame
    y: pd.Series
    original_y: pd.Series
    label_mapping: dict[str, int]
    feature_names: tuple[str, ...]
    numeric_feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]
    majority_label: str
    minority_label: str
    majority_count: int
    minority_count: int
    target_name: str
    source_filename: str
    source_format: str


@dataclass(frozen=True)
class _Attribute:
    name: str
    kind: str
    values: tuple[str, ...] = ()


def _split_keel(text: str) -> list[str]:
    """Split KEEL tokens; support single/double and doubled matching quotes."""
    tokens: list[str] = []
    i = 0
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] in "\"'":
            quote = text[i]
            i += 1
            value = ""
            while i < len(text):
                if text[i] == quote:
                    if i + 1 < len(text) and text[i + 1] == quote:
                        value += quote
                        i += 2
                        continue
                    i += 1
                    break
                value += text[i]
                i += 1
            else:
                raise DatasetError("Unterminated quoted KEEL value.")
            while i < len(text) and text[i].isspace():
                i += 1
            if i < len(text) and text[i] != ",":
                raise DatasetError("Unexpected characters after a quoted KEEL value.")
        else:
            start = i
            while i < len(text) and text[i] != ",":
                if text[i] in "\"'":
                    raise DatasetError("Quotes must surround the entire KEEL value.")
                i += 1
            value = text[start:i].strip()
        tokens.append(value)
        if i == len(text):
            return tokens
        i += 1


def _validate_names(names: list[str], context: str) -> None:
    if not names or any(not name.strip() or name == "?" for name in names):
        raise DatasetError(f"{context} must contain nonempty column names.")
    if len(set(names)) != len(names):
        raise DatasetError(f"Duplicate column names in {context}.")


def _frame(names: list[str], rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        raise DatasetError("Dataset is empty: no data records.")
    for number, row in enumerate(rows, 1):
        if len(row) != len(names):
            raise DatasetError(f"Data record {number} has {len(row)} fields; expected {len(names)}.")
    frame = pd.DataFrame(rows, columns=names)
    return frame.mask(frame.isin(["?", ""]), np.nan)


def _numeric(series: pd.Series, name: str, integer: bool = False) -> pd.Series:
    try:
        converted = pd.to_numeric(series, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DatasetError(f"Invalid numeric value in attribute {name!r}.") from exc
    present = converted.dropna()
    if not np.isfinite(present).all():
        raise DatasetError(f"Non-finite numeric value in attribute {name!r}.")
    if integer and ((present % 1) != 0).any():
        raise DatasetError(f"Non-integer value in integer attribute {name!r}.")
    return converted


def _read_csv(path: Path, target: str | None) -> tuple[pd.DataFrame, str, list[str], list[str]]:
    if target is None:
        raise DatasetError("CSV requires an explicit target column (--target).")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        try:
            records = list(csv.reader(stream, strict=True))
        except csv.Error as exc:
            raise DatasetError(f"Malformed CSV: {exc}") from exc
    if not records:
        raise DatasetError("Dataset is empty.")
    names, rows = records[0], records[1:]
    _validate_names(names, "CSV header")
    if target not in names:
        raise DatasetError(f"Target column {target!r} is not present.")
    data = _frame(names, rows)
    numeric: list[str] = []
    categorical: list[str] = []
    for name in names:
        if name == target:
            continue
        present = data[name].dropna()
        converted = pd.to_numeric(present, errors="coerce")
        if not present.empty and converted.notna().all():
            data[name] = _numeric(data[name], name)
            numeric.append(name)
        else:
            data[name] = data[name].astype("category")
            categorical.append(name)
    return data, target, numeric, categorical


def _attribute(body: str) -> _Attribute:
    match = re.fullmatch(r'''\s*("(?:[^"]|"")*"|'(?:[^']|'')*'|[^\s]+)\s+(.+?)\s*''', body)
    if match is None:
        raise DatasetError(f"Malformed @attribute declaration: {body!r}.")
    name_parts = _split_keel(match[1])
    if len(name_parts) != 1:
        raise DatasetError("Attribute names containing commas must be quoted.")
    name = name_parts[0]
    specification = match[2]
    if specification.startswith("{") and specification.endswith("}"):
        values = _split_keel(specification[1:-1])
        if any(not value or value == "?" for value in values) or len(set(values)) != len(values):
            raise DatasetError(f"Invalid nominal values for {name!r}.")
        return _Attribute(name, "nominal", tuple(values))
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    type_match = re.fullmatch(rf"(real|integer|numeric)(?:\s*\[\s*({number})\s*,\s*({number})\s*\])?", specification, re.I)
    if type_match:
        if type_match[2] is not None and float(type_match[2]) > float(type_match[3]):
            raise DatasetError(f"Reversed numeric range for {name!r}.")
        return _Attribute(name, type_match[1].lower())
    raise DatasetError(f"Unsupported or malformed type for attribute {name!r}: {specification!r}.")


def _read_keel(path: Path, override: str | None) -> tuple[pd.DataFrame, str, list[str], list[str]]:
    attributes: list[_Attribute] = []
    rows: list[list[str]] = []
    metadata: dict[str, str] = {}
    in_data = False
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        try:
            if in_data:
                if line.startswith("@"):
                    raise DatasetError("Unexpected metadata after @data.")
                rows.append(_split_keel(line))
                continue
            match = re.fullmatch(r"@(\w+)(?:\s+(.*))?", line)
            if not match:
                raise DatasetError("Expected a KEEL metadata directive before @data.")
            key, body = match[1].lower(), match[2] or ""
            if key == "attribute":
                attributes.append(_attribute(body))
            elif key in {"relation", "inputs", "outputs"}:
                if key in metadata or not body:
                    raise DatasetError(f"Missing or duplicate @{key} declaration.")
                metadata[key] = body
            elif key == "data" and not body:
                in_data = True
            else:
                raise DatasetError(f"Unsupported or malformed directive @{key}.")
        except DatasetError as exc:
            raise DatasetError(f"KEEL line {number}: {exc}") from exc
    if not in_data or "relation" not in metadata:
        raise DatasetError("KEEL requires @relation and @data declarations.")
    names = [attribute.name for attribute in attributes]
    _validate_names(names, "KEEL attributes")
    outputs = _split_keel(metadata["outputs"]) if "outputs" in metadata else []
    if len(outputs) > 1:
        raise DatasetError("Multiple KEEL output attributes are unsupported.")
    if any(output not in names for output in outputs):
        raise DatasetError("KEEL @outputs refers to an unknown attribute.")
    target = override if override is not None else (outputs[0] if outputs else None)
    if target is None:
        raise DatasetError("KEEL needs a single @outputs attribute or an explicit target override.")
    if target not in names:
        raise DatasetError(f"Target column {target!r} is not present.")
    inputs = _split_keel(metadata["inputs"]) if "inputs" in metadata else [name for name in names if name != target]
    _validate_names(inputs, "KEEL inputs")
    if any(name not in names for name in inputs):
        raise DatasetError("KEEL @inputs refers to an unknown attribute.")
    if target in inputs and override is None:
        raise DatasetError("KEEL target is also listed in @inputs.")
    # An explicit override makes all other declared attributes features.
    if override is not None:
        inputs = [name for name in names if name != target]
    data = _frame(names, rows)
    numeric: list[str] = []
    categorical: list[str] = []
    for attribute in attributes:
        name = attribute.name
        if attribute.kind == "nominal":
            invalid = set(data[name].dropna()) - set(attribute.values)
            if invalid:
                raise DatasetError(f"Undeclared nominal value in attribute {name!r}: {sorted(invalid)!r}.")
            if name in inputs:
                data[name] = pd.Categorical(data[name], categories=attribute.values)
                categorical.append(name)
        else:
            converted = _numeric(data[name], name, attribute.kind == "integer")
            if name in inputs:
                data[name] = converted
                numeric.append(name)
    return data[inputs + [target]], target, numeric, categorical


def load_dataset(
    filepath: str | Path,
    target_column: str | None = None,
    minority_label: str | None = None,
) -> LoadedDataset:
    """Load exactly two classes from UTF-8 CSV or KEEL .dat.

    CSV requires target_column; KEEL reads @outputs unless overridden. '?' and
    empty feature fields stay missing; missing targets fail. A tie requires an
    explicit minority_label. A more frequent class cannot be called minority.
    Labels match source text exactly, so '01' is distinct from '1'.
    """
    path = Path(filepath)
    if not path.is_file():
        raise DatasetError(f"Dataset file does not exist or is not a regular file: {path}")
    format_name = path.suffix.lower().lstrip(".")
    try:
        if format_name == "csv":
            data, target, numeric, categorical = _read_csv(path, target_column)
        elif format_name == "dat":
            data, target, numeric, categorical = _read_keel(path, target_column)
        else:
            raise DatasetError("Unsupported format; use a .csv or KEEL .dat file.")
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"Cannot read dataset {path}: {exc}") from exc
    original_y = data[target].copy()
    if original_y.isna().any():
        raise DatasetError("Missing target values are not permitted.")
    counts = original_y.value_counts()
    if len(counts) != 2:
        raise DatasetError(f"Binary classification requires exactly two target classes; found {len(counts)}.")
    if minority_label is None:
        if counts.iloc[0] == counts.iloc[1]:
            raise DatasetError("Equal class counts: supply an explicit minority label.")
        minority = str(counts.idxmin())
    else:
        minority = minority_label
        if minority not in counts.index:
            raise DatasetError(f"Minority label {minority!r} is not a target class.")
    majority = next(str(label) for label in counts.index if label != minority)
    if counts[minority] > counts[majority]:
        raise DatasetError("Explicit minority label is more frequent than the other class.")
    mapping = {majority: 0, minority: 1}
    X = data.drop(columns=target)
    if X.shape[1] == 0:
        raise DatasetError("Dataset must contain at least one feature column.")
    return LoadedDataset(
        X=X, y=original_y.map(mapping).astype("int64"), original_y=original_y,
        label_mapping=mapping, feature_names=tuple(X.columns),
        numeric_feature_names=tuple(numeric), categorical_feature_names=tuple(categorical),
        majority_label=majority, minority_label=minority,
        majority_count=int(counts[majority]), minority_count=int(counts[minority]),
        target_name=target, source_filename=str(path.resolve()), source_format=format_name,
    )
