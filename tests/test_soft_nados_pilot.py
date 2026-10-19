import json

import numpy as np

from scripts.run_soft_nados_pilot import (
    datasets,
    design_hash,
    inject_symmetric_noise,
    load_config,
)


def test_builtin_dataset_definitions_are_binary_imbalanced_and_fixed():
    loaded = datasets()
    assert list(loaded) == ["breast_cancer", "wine_class_2_vs_rest", "digits_1_vs_rest_3_to_1"]
    assert {name: X.shape for name, (X, _) in loaded.items()} == {
        "breast_cancer": (569, 30),
        "wine_class_2_vs_rest": (178, 13),
        "digits_1_vs_rest_3_to_1": (728, 64),
    }
    for X, y in loaded.values():
        assert np.isfinite(X).all()
        assert set(y) == {0, 1}
        assert np.sum(y == 1) < np.sum(y == 0)


def test_noise_is_deterministic_and_flips_floor_ten_percent_per_class():
    y = np.r_[np.zeros(33, dtype=int), np.ones(17, dtype=int)]
    first, flags = inject_symmetric_noise(y, .10, 1001)
    second, again = inject_symmetric_noise(y, .10, 1001)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(flags, again)
    assert np.sum(flags & (y == 0)) == 3
    assert np.sum(flags & (y == 1)) == 1
    np.testing.assert_array_equal(first, np.where(flags, 1 - y, y))


def test_frozen_config_and_hash_are_stable(tmp_path):
    config = load_config()
    digest = design_hash(config)
    assert len(digest) == 64
    changed = json.loads(json.dumps(config))
    changed["scientific_design"]["soft_nados"]["temperature"] = .10
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        load_config(path)
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("Modified frozen temperature was accepted")
