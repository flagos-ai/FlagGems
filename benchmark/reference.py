# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference-only benchmark options and reports, independent of correctness pytest."""

import re
import traceback
from types import ModuleType


def reference_failure(error):
    """Conservative diagnostic hints, not an authoritative capability table."""
    message = str(error)
    category = "UNKNOWN"
    obj = getattr(error, "obj", None)
    if (
        isinstance(error, AttributeError)
        and isinstance(obj, ModuleType)
        and (obj.__name__ == "torch" or obj.__name__.startswith("torch."))
        and getattr(error, "name", None)
    ):
        category = "API_MISSING"
    elif isinstance(error, (RuntimeError, NotImplementedError)) and re.search(
        r"not implemented for ['\"]"
        r"(?:Half|BFloat16|Float|Double|Char|Byte|Short|Int|Long|Bool|Float8_[A-Za-z0-9_]+)['\"]",
        message,
    ):
        category = "DTYPE_UNSUPPORTED"
    elif isinstance(error, NotImplementedError):
        category = "NOT_IMPLEMENTED"
    return {
        "category": category,
        "type": type(error).__name__,
        "message": message,
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }


def validate_reference_options(config):
    if not getattr(config.option, "reference_only", False):
        return False
    import pytest

    conflicts = [
        name
        for name in (
            "override",
            "override_config",
            "profile_only",
            "preflight_only",
            "list_cases",
            "query",
            "parallel",
            "numprocesses",
        )
        if getattr(config.option, name, None)
    ]
    if conflicts:
        raise pytest.UsageError(
            "--reference-only cannot be combined with "
            + ", ".join("--" + name.replace("_", "-") for name in conflicts)
        )
    return True


def reference_report(records, *, exitstatus=0):
    """Separate source skips, unsupported tests and real execution failures."""
    # pytest.skip aborts the entire original benchmark node, including any
    # remaining cases. Earlier calls remain evidence, not a completed node.
    skipped_nodes = {
        r.get("nodeid")
        for r in records
        if r.get("pytest_phase") and r["status"] == "SKIP"
    }
    statuses = [
        (
            "SKIP"
            if r["status"] in {"PASSED", "NOT_RUN"} and r.get("nodeid") in skipped_nodes
            else r["status"]
        )
        for r in records
    ]
    if exitstatus not in (0, 1, 5) or any(s in {"FAILED", "NOT_RUN"} for s in statuses):
        status = "FAILED"
    elif "UNSUPPORTED" in statuses:
        status = "UNSUPPORTED"
    elif exitstatus == 1:
        status = "FAILED"
    elif "PASSED" in statuses:
        status = "PASSED"
    elif statuses:
        status = "ALL_SKIP"
    else:
        status = "NO_CASES"
    return {
        "schema_version": "flaggems.reference/v1",
        "phase": "timing",
        "status": status,
        "records": records,
    }
