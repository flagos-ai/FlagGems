"""Recompute all109 new-kernel attribution decisions from published evidence."""

import ast
import hashlib
import json
import runpy
from pathlib import Path


def classify_specialization(roles, source_witness, jit_names, expected):
    """Require producer-linked kernel identity; a Python entry alone proves no gain."""
    new, aa = roles["new"], roles["aa"]
    names = {
        k for k in new if k.startswith("kernel:") and k.split(":", 1)[1] in jit_names
    }
    aa_names = {
        k for k in aa if k.startswith("kernel:") and k.split(":", 1)[1] in jit_names
    }
    source_seen = any("/ops/linear_projection.py:" in k for k in source_witness)
    if names:
        assert source_seen, "kernel without implementation source witness"
        assert all(
            not any(k in roles[role] for k in names) for role in ["old", "old_repeat"]
        ), "kernel also in baseline; cannot attribute"
    observed = bool(names) and source_seen
    complete = (
        observed
        and names == aa_names
        and all(new[k] == aa.get(k) == expected for k in names)
    )
    return names, source_seen, observed, complete


P = Path(__file__).resolve().parent
runpy.run_path(str(P / "verify_results.py"), run_name="__main__")
measurements = json.loads((P / "measurements.json").read_bytes())
attribution = json.loads((P / "specialization-attribution.json").read_bytes())
witness = json.loads((P / "kernel-witness.json").read_bytes())
source = P.parents[3] / "src/flag_gems/runtime/backend/_hygon/ops/linear_projection.py"
assert hashlib.sha256(source.read_bytes()).hexdigest() == attribution["source_sha256"]
assert (
    hashlib.sha256((P / "kernel-witness.json").read_bytes()).hexdigest()
    == attribution["kernel_witness_sha256"]
)
tree = ast.parse(source.read_text(encoding="utf8"))
jit_names = {
    n.name
    for n in tree.body
    if isinstance(n, ast.FunctionDef)
    and any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in n.decorator_list)
}
claims = {c["id"]: c for c in attribution["cases_detail"]}
assert len(measurements["cases"]) == len(claims) == 109
assert set(claims) == set(witness["cases"]) == {c["id"] for c in measurements["cases"]}
gains = model_gains = specialized = 0
for case in measurements["cases"]:
    cid = case["id"]
    claim = claims[cid]
    roles = witness["cases"][cid]
    assert set(roles) == set(case["arms"])
    names, seen, observed, complete = classify_specialization(
        roles, case["source_witness"]["new"], jit_names, 50 * case["calls_per_roi"]
    )
    gain = complete and case["fg_pair_qualified"] and case["old_over_new"] > 1.05
    model = cid.startswith("model54-")
    assert claim["model_scope"] == model
    assert claim["source_entry_witness"] == seen
    assert claim["specialization_observed"] == observed
    assert claim["all50_samples_valid_and_expected_repeat_kernel_counts"] == complete
    assert claim["attributed_gain_above5percent"] == gain
    assert claim["correct_old_new_DAS"] == [
        case[k + "_correct"] for k in ["old", "new", "das"]
    ]
    assert claim["fg_pair_qualified"] == case["fg_pair_qualified"]
    assert claim["fg_das_qualified"] == case["fg_das_qualified"]
    assert claim["old_over_new"] == case["old_over_new"]
    assert claim["DAS_over_new"] == case["das_over_new"]
    assert claim["specialization_kernel_counts"] == {
        role: {k: roles[role].get(k, 0) for k in sorted(names)}
        for role in ["new", "aa"]
    }
    gains += gain
    model_gains += gain and model
    specialized += observed
assert gains == attribution["attributed_qualified_gains_above5percent"]
assert model_gains == attribution["model_attributed_qualified_gains_above5percent"]
assert specialized == attribution["actual_specialized_cases"]
print(
    json.dumps(
        {
            "cases": 109,
            "actual_specialized": specialized,
            "attributed_gains": gains,
            "model_attributed_gains": model_gains,
        },
        indent=2,
    )
)
