"""Small-study export checks without committing or using raw datasets."""

from copy import deepcopy

import pytest

from scripts.run_ec_ebo_small_study import ec_ebo_diagnostics


def test_cluster_export_keeps_shortfalls_duplicates_and_intrusion_separate():
    record = {
        "job": {"classifier": "LogReg", "method": "ec_ebo", "fold": 0},
        "status": "success",
        "resampler_diagnostics": {
            "allocation": {"cluster_sizes": [2], "cluster_mean_entropies": [.3]},
            "cluster_allocations": [3], "cluster_generated_counts": [2],
            "cluster_shortfall_reasons": [{"no_positive_seed": 1}],
            "requested_count": 3, "generated_count": 2,
            "duplicate_generated_count": 1,
            "nearest_original_majority_proxy": {"count": 1, "tie_count": 0, "rate": .5},
        },
    }
    details, summaries = ec_ebo_diagnostics([record])
    assert details[0]["original_minority_rows"] == 2
    assert details[0]["quota"] == 3
    assert details[0]["generated"] == 2
    assert details[0]["shortfall"] == 1
    assert summaries[0]["shortfall_reason_counts"] == {"no_positive_seed": 1}
    assert summaries[0]["duplicate_generated_total"] == 1
    assert summaries[0]["intrusion_proxy_rate"] == .5

    corrupt = deepcopy(record)
    corrupt["resampler_diagnostics"]["cluster_allocations"] = [4]
    with pytest.raises(ValueError, match="totals disagree"):
        ec_ebo_diagnostics([corrupt])
