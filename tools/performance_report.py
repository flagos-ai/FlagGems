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

"""Pure benchmark report parsing, shared by the scheduler and host tests."""

import json

if __package__:
    from .consts import DTYPE_MAP
else:
    from consts import DTYPE_MAP


def _case_record(result, default=None):
    """Use the same comparison priority and provenance in both report paths."""
    compared = result.get("compared_speedup")
    record = {
        "base": result.get("latency_base", default),
        "gems": result.get("latency", default),
        "speedup": compared if compared is not None else result.get("speedup", default),
    }
    if compared is not None:
        record["native_speedup"] = result.get("speedup")
    for field in (
        "compared_speedup",
        "base_raw_speedup",
        "base_data_source",
        "base_shape_key",
        "base_latency_ms",
        "base_scaling_factor",
        "base_miss_reason",
    ):
        if result.get(field) is not None:
            record[field] = result[field]
    return record


def parse_perf_data(op, result_file):
    raw_data = {}
    try:
        with result_file.open("r") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {
            "status": "Error",
            "reason": f"Invalid JSON in {result_file}",
        }

    data = raw_data.get(op, {})
    if not data:
        return {"status": "NotFound"}

    result = data.get("result", "NotFound")
    if result in ["failed", "skipped"]:
        return {
            "status": result.title(),
            "reason": data.get("reason", "Unknown"),
            "test_case": data.get("test_case", "Unknown"),
        }

    bench_res = {}
    records = data.get("details", [])
    if not any(item.get("native_baseline_skip_reason") for item in records):
        for item in records:
            dtype = DTYPE_MAP.get(item["dtype"], item["dtype"])
            details = {}
            total = 0.0
            count = 0
            for res in item.get("result", []):
                shape = str(res.get("shape_detail", "Unknown")).replace(" ", "")
                details.setdefault(shape, {})
                record = _case_record(res, default=0.0)
                details[shape].update(record)
                speedup = record["speedup"]
                count += 1
                total += speedup

            if details:
                bench_res[dtype] = {
                    "result": "OK",
                    "details": details,
                    "speedup": total / count,
                }
            else:
                bench_res[dtype] = {
                    "result": "Unknown",
                    "details": {},
                    "speedup": 0,
                }

        return {
            "status": result.title(),
            "data": bench_res,
            "test_case": data.get("test_case", "Unknown"),
        }

    native_baseline_skip_reasons = []
    speedup_totals = {}
    speedup_counts = {}

    for item in records:
        dtype = DTYPE_MAP.get(item["dtype"], item["dtype"])
        skip_reason = item.get("native_baseline_skip_reason")
        if skip_reason and skip_reason not in native_baseline_skip_reasons:
            native_baseline_skip_reasons.append(skip_reason)
        dtype_result = bench_res.setdefault(
            dtype,
            {
                "result": "Unknown",
                "details": {},
                "speedup": None,
            },
        )
        details = dtype_result["details"]
        for res in item.get("result", []):
            shape = str(res.get("shape_detail", "Unknown")).replace(" ", "")
            details.setdefault(shape, {})
            record = _case_record(res)
            details[shape].update(record)
            speedup = record["speedup"]
            if isinstance(speedup, (int, float)):
                speedup_counts[dtype] = speedup_counts.get(dtype, 0) + 1
                speedup_totals[dtype] = speedup_totals.get(dtype, 0.0) + speedup

        if details:
            dtype_result["result"] = "OK"

    for dtype, dtype_result in bench_res.items():
        count = speedup_counts.get(dtype, 0)
        dtype_result["speedup"] = (
            speedup_totals.get(dtype, 0.0) / count if count else None
        )

    parsed_result = {
        "status": result.title(),
        "data": bench_res,
        "test_case": data.get("test_case", "Unknown"),
    }
    if native_baseline_skip_reasons:
        parsed_result["native_baseline_skip_reason"] = "; ".join(
            native_baseline_skip_reasons
        )
    return parsed_result
