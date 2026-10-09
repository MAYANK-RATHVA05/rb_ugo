"""Deterministic loader tests; toy data are not experimental evidence."""
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DatasetError, load_dataset

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("format_name", ["csv", "dat"])
def test_load_toy(format_name: str) -> None:
    path = FIXTURES / f"toy.{format_name}"
    dataset = load_dataset(path, "class" if format_name == "csv" else None)
    assert dataset.X.shape == (6, 2)
    assert dataset.feature_names == ("amount", "color")
    assert dataset.numeric_feature_names == ("amount",)
    assert dataset.categorical_feature_names == ("color",)
    assert pd.api.types.is_numeric_dtype(dataset.X.amount)
    assert isinstance(dataset.X.color.dtype, pd.CategoricalDtype)
    assert dataset.X.loc[1, "color"] == "blue, pale"
    assert dataset.X.isna().sum().sum() == 2
    assert dataset.original_y.tolist() == ["normal", "normal", "normal", "normal", "rare", "normal"]
    assert dataset.y.tolist() == [0, 0, 0, 0, 1, 0]
    assert dataset.label_mapping == {"normal": 0, "rare": 1}
    assert (dataset.majority_label, dataset.majority_count) == ("normal", 5)
    assert (dataset.minority_label, dataset.minority_count) == ("rare", 1)
    assert dataset.source_filename == str(path.resolve())
    assert dataset.source_format == format_name
    assert dataset.target_name == "class"


@pytest.mark.parametrize("body, message", [
    ("", "empty"),
    ("x,class\n", "empty"),
    ("x,class\n1,?\n2,a\n3,b\n", "Missing target"),
    ("x,class\n1,\n2,a\n3,b\n", "Missing target"),
    ("x,class\n1,a\n2,a\n", "exactly two"),
    ("x,class\n1,a\n2,b\n3,c\n", "exactly two"),
    ("x,class\n1,a\n2,b\n", "Equal class counts"),
    ("x,class\n1,a,extra\n2,b\n", "fields"),
    ("x,class\n1\n2,b\n", "fields"),
    ("x,x,class\n1,2,a\n", "Duplicate column"),
    ("x,class\n\"unterminated,a\n", "Malformed CSV"),
    ("x,class\ninf,a\n2,a\n3,b\n", "Non-finite"),
    ("class\na\na\nb\n", "at least one feature"),
])
def test_invalid_csv(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset(path, "class")


def test_csv_requires_target() -> None:
    with pytest.raises(DatasetError, match="explicit target"):
        load_dataset(FIXTURES / "toy.csv")
    with pytest.raises(DatasetError, match="not present"):
        load_dataset(FIXTURES / "toy.csv", "absent")


def test_minority_override_and_label_preservation(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("x,class\n1,01\n2,1\n", encoding="utf-8")
    dataset = load_dataset(path, "class", "01")
    assert dataset.original_y.tolist() == ["01", "1"]
    assert dataset.label_mapping == {"1": 0, "01": 1}
    assert dataset.y.tolist() == [1, 0]
    with pytest.raises(DatasetError, match="not a target class"):
        load_dataset(path, "class", "absent")
    with pytest.raises(DatasetError, match="more frequent"):
        load_dataset(FIXTURES / "toy.csv", "class", "normal")


@pytest.mark.parametrize("path", ["missing.csv", "."])
def test_invalid_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(DatasetError, match="not a regular file"):
        load_dataset(tmp_path / path, "class")


def test_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("x,class\n1,a\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="Unsupported format"):
        load_dataset(path, "class")


def test_csv_quoted_multiline_and_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "quoted.csv"
    path.write_text('text,missing,class\n"hello, \"\"world\"\"",?,a\n"two\nlines",?,a\nother,?,b\n', encoding="utf-8", newline="\n")
    dataset = load_dataset(path, "class")
    assert dataset.X.loc[0, "text"] == 'hello, "world"'
    assert dataset.X.loc[1, "text"] == "two\nlines"
    assert dataset.X.missing.isna().all()
    assert dataset.categorical_feature_names == ("text", "missing")


def test_keel_quotes_and_integer(tmp_path: Path) -> None:
    path = tmp_path / "quoted.dat"
    path.write_text('''@relation quotes
@attribute "numeric feature" integer [0, 4]
@attribute 'category name' {'it''s blue', "red, pale"}
@attribute class {a, b}
@inputs "numeric feature", 'category name'
@outputs class
@data
1,'it''s blue',a
2,"red, pale",a
?, ?,b
''', encoding="utf-8")
    dataset = load_dataset(path)
    assert dataset.feature_names == ("numeric feature", "category name")
    assert dataset.X.loc[0, "category name"] == "it's blue"
    assert dataset.X.loc[1, "category name"] == "red, pale"
    assert dataset.X.iloc[2].isna().all()


@pytest.mark.parametrize("old,new,message", [
    ("@outputs class", "@outputs class, color", "Multiple"),
    ("@outputs class", "@outputs absent", "unknown attribute"),
    ("@inputs amount, color", "@inputs amount, absent", "unknown attribute"),
    ("@inputs amount, color", "@inputs amount, class", "also listed"),
    ("@inputs amount, color", "@inputs amount, amount", "Duplicate"),
    ("@data", "", "before @data"),
    ("@relation toy", "", "requires @relation and @data"),
    ("@outputs class", "@outputs class\n@outputs class", "duplicate"),
    ("1,red,normal", "1,red,?", "Missing target"),
    ("1,red,normal", "1,green,normal", "Undeclared nominal"),
    ("1,red,normal", "bad,red,normal", "Invalid numeric"),
    ("1,red,normal", "1,red,normal,extra", "fields"),
    ("2,'blue, pale',normal", "2,'blue, pale,normal", "Unterminated"),
    ("2,'blue, pale',normal", "2,'blue, pale'junk,normal", "Unexpected characters"),
    ("real [0, 10]", "string", "Unsupported or malformed type"),
    ("real [0, 10]", "real [10, 0]", "Reversed"),
    ("@attribute color", "@attribute amount", "Duplicate column"),
])
def test_invalid_keel(tmp_path: Path, old: str, new: str, message: str) -> None:
    body = (FIXTURES / "toy.dat").read_text(encoding="utf-8").replace(old, new)
    path = tmp_path / "invalid.dat"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset(path)


def test_keel_target_override_and_inputs(tmp_path: Path) -> None:
    body = (FIXTURES / "toy.dat").read_text(encoding="utf-8")
    path = tmp_path / "override.dat"
    path.write_text(body.replace("@outputs class", ""), encoding="utf-8")
    with pytest.raises(DatasetError, match="single @outputs"):
        load_dataset(path)
    assert load_dataset(path, "class").label_mapping == {"normal": 0, "rare": 1}
    path.write_text(body.replace("@inputs amount, color", "@inputs amount"), encoding="utf-8")
    assert load_dataset(path).feature_names == ("amount",)
    # Override can select another target while retaining all other attributes.
    path.write_text(body.replace("4,?,normal", "4,red,normal"), encoding="utf-8")
    assert load_dataset(path, "color", "blue, pale").target_name == "color"


def test_keel_invalid_integer(tmp_path: Path) -> None:
    path = tmp_path / "integer.dat"
    body = (FIXTURES / "toy.dat").read_text(encoding="utf-8")
    path.write_text(body.replace("real [0, 10]", "integer [0, 10]").replace("2,'blue", "2.5,'blue"), encoding="utf-8")
    with pytest.raises(DatasetError, match="Non-integer"):
        load_dataset(path)
