import json

import numpy as np
import pytest

from src.results import ResultStore, canonical_json


def test_atomic_numpy_roundtrip_and_identity(tmp_path):
    store = ResultStore(tmp_path, {"input": "abc", "config": {"seed": 42}})
    job = {"fold": 0, "method": "nores"}
    written = store.write({"job": job, "status": "success", "predictions": np.array([0, 1]),
                           "diagnostic": np.nan})
    assert store.read(job) == written
    assert written["diagnostic"] is None
    assert written["predictions"] == [0, 1]
    assert ResultStore(tmp_path, {"input": "abc", "config": {"seed": 42}}).run_id == store.run_id
    assert not list(tmp_path.glob(".pending-*"))
    with pytest.raises(ValueError, match="different or corrupt"):
        ResultStore(tmp_path, {"input": "different"})


def test_corrupt_record_is_not_silently_reused(tmp_path):
    store = ResultStore(tmp_path, {"input": "abc"})
    job = {"fold": 0}
    store.write({"job": job, "status": "failed", "metrics": None})
    path = next(tmp_path.glob("job-*.json"))
    content = json.loads(path.read_text())
    content["record"]["status"] = "success"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="corrupt"):
        store.read(job)


def test_existing_unidentified_directory_is_rejected(tmp_path):
    (tmp_path / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="without a manifest"):
        ResultStore(tmp_path, {"input": "abc"})


def test_json_refuses_objects_and_nonterminal_records(tmp_path):
    with pytest.raises(TypeError):
        canonical_json(object())
    store = ResultStore(tmp_path, {"x": 1})
    with pytest.raises(ValueError, match="terminal"):
        store.write({"job": {"fold": 0}, "status": "running"})
