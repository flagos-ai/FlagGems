"""Recompute every published median, ratio and A/A gate from full ROI samples."""
from pathlib import Path
import json
import math
import statistics

root = Path(__file__).resolve().parent
doc = json.loads((root / "measurements.json").read_text(encoding="utf8"))
counts = {"cases": 0, "complete": 0, "new_correct": 0, "fg_pairs": 0, "das_pairs": 0, "DAS90": 0}
for case in doc["cases"]:
    counts["cases"] += 1
    arms = case["arms"]
    medians = {}
    stable = {}
    for provider, repeat in [("old", "old_repeat"), ("new", "aa"), ("das", "das_repeat")]:
        base, again = arms[provider], arms[repeat]
        if base["timing_valid"] and again["timing_valid"]:
            for arm in (base, again):
                blocks = arm["device_blocks"]
                assert len(blocks) == 2 and all(len(b) == 25 for b in blocks)
                assert all(math.isfinite(x) and x >= 0 for b in blocks for x in b)
                assert arm["input_backing_hashes"] == base["input_backing_hashes"]
            medians[provider] = statistics.median(sum(base["device_blocks"], []))
            stable[provider] = all(
                statistics.fmean(x) > 0
                and abs(statistics.fmean(y) / statistics.fmean(x) - 1) <= 0.05
                for x, y in zip(base["device_blocks"], again["device_blocks"])
            )
            assert stable[provider] == case[provider + "_stable"]
    counts["complete"] += bool(case["input_match"])
    counts["new_correct"] += bool(case["new_correct"])
    counts["fg_pairs"] += bool(case["fg_pair_qualified"])
    counts["das_pairs"] += bool(case["fg_das_qualified"])
    counts["DAS90"] += bool(case["target90"])
    if case["old_over_new"] is not None:
        assert math.isclose(medians["old"] / medians["new"], case["old_over_new"], rel_tol=1e-12)
    if case["das_over_new"] is not None:
        assert math.isclose(medians["das"] / medians["new"], case["das_over_new"], rel_tol=1e-12)
    if case["fg_pair_qualified"]:
        assert case["old_correct"] and case["new_correct"]
        assert stable["old"] and stable["new"]
    if case["fg_das_qualified"]:
        assert case["new_correct"] and case["das_correct"]
        assert stable["new"] and stable["das"]
    assert bool(case["target90"]) == bool(case["fg_das_qualified"] and case["das_over_new"] >= 0.9)
print(json.dumps(counts, indent=2))
