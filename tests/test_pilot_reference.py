import pytest

from scripts.pilot_paper_comparison import compare_pilot


def test_reference_difference_retains_missing_values_and_f1_limit():
    summary = {"datasets": [{"dataset": "linear", "summaries": [{"classifier": "GNB", "method": "nores",
                "classifier_fidelity": "python_approximation", "metrics": {
                    "iam": {"method_mean": .5, "paired_count": 10},
                    "f1": {"method_mean": None, "paired_count": 0}}}]}]}
    reference = {"methods_in_value_order": ["nores"], "source_tables": {"linear": {"table": 9, "page": 13}},
                 "datasets": {"linear": {"GNB": {"iam": [.25], "f1": [.6]}}}}
    rows = compare_pilot(summary, reference)
    assert rows[0]["pilot_minus_paper"] == .25
    assert "Different reconstruction" in rows[0]["comparison_status"]
    assert rows[1]["pilot_minus_paper"] is None
    assert rows[1]["paired_count"] == 0
    assert "F1 averaging unresolved" in rows[1]["comparison_status"]
    summary["datasets"][0]["summaries"][0]["method"] = "unknown"
    with pytest.raises(ValueError, match="no audited paper reference"):
        compare_pilot(summary, reference)
