"""Verify backend scoping preserves every measured Hygon test and its assertions."""

import ast
import hashlib
import json
import types
import xml.etree.ElementTree as ET
from pathlib import Path

P = Path(__file__).resolve().parent
proof = json.loads((P / "ci-test-scope-proof.json").read_bytes())
source = (P.parents[3] / "tests/test_hygon_linear_projection.py").read_text(
    encoding="utf8"
)
assert hashlib.sha256(source.encode()).hexdigest() == proof["new_test_canonical_sha256"]
old_tree = ast.parse(proof["previous_test_source_utf8"])
new_tree = ast.parse(source)
marks = [
    node
    for node in new_tree.body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id == "pytestmark"
        for target in node.targets
    )
]
assert len(marks) == 1
mark = marks[0]
new_tree.body.remove(mark)
assert ast.dump(old_tree, include_attributes=False) == ast.dump(
    new_tree, include_attributes=False
)
assert isinstance(mark.value, ast.Call)
assert isinstance(mark.value.func, ast.Attribute) and mark.value.func.attr == "skipif"
condition = compile(ast.Expression(mark.value.args[0]), "<backend-scope>", "eval")
for vendor in ["hygon", "nvidia", "ascend"]:
    actual = eval(condition, {"flag_gems": types.SimpleNamespace(vendor_name=vendor)})
    assert actual == (vendor != "hygon")
junit = P / "ci-test-scope-junit.xml"
assert hashlib.sha256(junit.read_bytes()).hexdigest() == proof["published_junit_sha256"]
tests = list(ET.parse(junit).getroot().iter("testcase"))
assert len(tests) == 12
assert all(
    case.find("failure") is None
    and case.find("error") is None
    and case.find("skipped") is None
    for case in tests
)
assert proof["pytest_exit_code"] == 0
assert proof["terminal_job"]["State"] == "COMPLETED"
assert proof["terminal_job"]["ExitCode"] == "0:0"
assert proof["production_src_tree"] == "091642c92e6892ebc92ed57dbe066a347cdb99f0"
print(
    json.dumps(
        {
            "Hygon_tests_passed": 12,
            "skipped": 0,
            "all_existing_test_AST_unchanged": True,
        }
    )
)
