#!/usr/bin/env python3
"""Validate and pair official index_put Operator benchmark output."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

BOOL_ORDINALS = (
    set(range(10, 14)) | set(range(24, 28)) | set(range(38, 42)) | {47, 48, 54, 55}
)


def _normal_shape(shape):
    return ast.literal_eval(shape) if isinstance(shape, str) else shape


def _add_metadata(row, ordinal):
    shape = _normal_shape(row["shape"])
    if not isinstance(shape, list) or len(shape) != 4:
        raise ValueError(f"invalid shape detail {shape!r}")
    row["shape_data"] = shape
    row["ordinal"] = ordinal
    # benchmark/test_index_put.py emits 42 acc_false then 14 acc_true rows
    # per API, in dtype-major order. shape_detail's final field is accumulate.
    row["accumulate"] = bool(shape[3])
    row["is_bool"] = ordinal in BOOL_ORDINALS


def load_csv(path: Path):
    with path.open(newline="") as f:
        raw = list(csv.DictReader(f))
    rows = []
    ordinals = defaultdict(int)
    for item in raw:
        ordinal = ordinals[item["operator"]]
        ordinals[item["operator"]] += 1
        row = {
            "operator": item["operator"],
            "dtype": item["dtype"],
            "shape": item["shape"],
            "torch": float(item["torch"]),
            "candidate": float(item.get("cand", item.get("candidate"))),
            "speedup": float(item["speedup"]),
        }
        _add_metadata(row, ordinal)
        rows.append(row)
    return rows


def load_json(path: Path):
    data = json.loads(path.read_text())
    rows = []
    for operator in ("index_put", "index_put_"):
        ordinal = 0
        for detail in data[operator]["details"]:
            dtype = detail["dtype"]
            if (
                detail.get("mode") != "operator"
                or detail.get("level") != "comprehensive"
            ):
                raise ValueError(f"{path}: unexpected benchmark mode/level")
            for result in detail["result"]:
                row = {
                    "operator": operator,
                    "dtype": dtype,
                    "shape": result["shape_detail"],
                    "torch": float(result["latency_base"]),
                    "candidate": float(result["latency"]),
                    "speedup": float(result["speedup"]),
                    "candidate_source": result.get("candidate_source"),
                }
                _add_metadata(row, ordinal)
                expected = (row["torch"] + 1e-7) / (row["candidate"] + 1e-7)
                if not math.isclose(
                    expected, row["speedup"], rel_tol=2e-5, abs_tol=2e-8
                ):
                    raise ValueError(f"{path}: speedup arithmetic mismatch: {row}")
                rows.append(row)
                ordinal += 1
    return rows


def validate_matrix(rows, source):
    counts = Counter(row["operator"] for row in rows)
    if counts != Counter({"index_put": 56, "index_put_": 56}):
        raise ValueError(f"{source}: expected 56 cases per API, got {counts}")
    for operator in ("index_put", "index_put_"):
        selected = [r for r in rows if r["operator"] == operator]
        if [r["ordinal"] for r in selected] != list(range(56)):
            raise ValueError(f"{source}: {operator} ordering has gaps or duplicates")
        if sum(r["accumulate"] for r in selected) != 14:
            raise ValueError(
                f"{source}: {operator} does not contain 14 accumulate=True cases"
            )
        if sum(r["is_bool"] for r in selected) != 16:
            raise ValueError(f"{source}: {operator} expected 16 official Bool cases")
        for row in selected:
            if not all(
                math.isfinite(row[k]) and row[k] > 0
                for k in ("torch", "candidate", "speedup")
            ):
                raise ValueError(f"{source}: invalid latency/speedup: {row}")


def current_gate_possible(row):
    input_shape, index_shapes, values_shape, accumulate = row["shape_data"]
    if row["is_bool"]:
        return (
            row["dtype"] in ("torch.float16", "torch.bfloat16", "torch.float32")
            and 1 <= len(input_shape) <= 3
            and index_shapes == [input_shape]
            and len(values_shape) == 1
            and 0 <= math.prod(values_shape) <= math.prod(input_shape)
            and math.prod(input_shape) <= (262144 if accumulate else 4096)
        )
    if accumulate:
        # The atomic routes accept contiguous 1-D FP16/FP32 updates with one
        # index tensor, or full-rank 2-D/3-D FP16/FP32 updates whose index
        # tensors broadcast to the values shape. BF16 remains a semantic
        # fallback because the MUSA extension has no verified BF16 atomicAdd.
        one_dim = (
            len(input_shape) == 1
            and len(index_shapes) == 1
            and len(index_shapes[0]) == 1
            and values_shape == index_shapes[0]
        )
        fullrank = (
            1 < len(input_shape) <= 3
            and len(index_shapes) == len(input_shape)
            and all(shape and len(shape) <= 3 for shape in index_shapes)
        )
        if not (one_dim or fullrank):
            return False
        if row["dtype"] not in ("torch.float16", "torch.float32"):
            return False
        if math.prod(index_shapes[0]) <= 0:
            return False
        if math.prod(index_shapes[0]) > 65535 * 256:
            return False
        if one_dim:
            return True
        rank = max(map(len, index_shapes))
        result = [1] * rank
        for shape in index_shapes:
            padded = [1] * (rank - len(shape)) + shape
            for dim, size in enumerate(padded):
                if size != 1 and result[dim] != 1 and result[dim] != size:
                    return False
                result[dim] = max(result[dim], size)
        if math.prod(result) > 65535 * 256 or len(values_shape) > rank:
            return False
        padded_values = [1] * (rank - len(values_shape)) + values_shape
        return all(
            size == 1 or size == result[dim] for dim, size in enumerate(padded_values)
        )
    if len(input_shape) == 2 and len(index_shapes) == 1:
        index_shape = index_shapes[0]
        if not index_shape or math.prod(index_shape) > 65535:
            return False
        m = math.prod(index_shape)
        n = input_shape[1]
        return values_shape == [n] or values_shape == [m, n]

    if not (1 <= len(input_shape) <= 3 and len(index_shapes) == len(input_shape)):
        return False
    if any(not shape or len(shape) > 3 for shape in index_shapes):
        return False
    rank = max(map(len, index_shapes))
    result = [1] * rank
    for shape in index_shapes:
        padded = [1] * (rank - len(shape)) + shape
        for dim, size in enumerate(padded):
            if size != 1 and result[dim] != 1 and result[dim] != size:
                return False
            result[dim] = max(result[dim], size)
    if math.prod(result) > 65535 * 256 or len(values_shape) > rank:
        return False
    padded_values = [1] * (rank - len(values_shape)) + values_shape
    return all(
        size == 1 or size == result[dim] for dim, size in enumerate(padded_values)
    )


def native_kernel_kind(row):
    if not current_gate_possible(row):
        return "fallback"
    if row["is_bool"]:
        return "bool-mask-prefix"
    input_shape, index_shapes, _, _ = row["shape_data"]
    if row["accumulate"]:
        return (
            "atomic-1d-accumulate"
            if len(input_shape) == 1
            else "atomic-fullrank-accumulate"
        )
    if (
        row["operator"] == "index_put"
        and not row["accumulate"]
        and len(input_shape) == 2
        and len(index_shapes) == 1
        and math.prod(input_shape) <= 4096
        and math.prod(index_shapes[0]) <= 256
    ):
        return "row-clone-scatter"
    if len(input_shape) == 2 and len(index_shapes) == 1:
        return "row-tail"
    return "fullrank-N1"


def case_category(row):
    input_shape, index_shapes, _, accumulate = row["shape_data"]
    if row["is_bool"]:
        return "bool_accumulate" if accumulate else "bool_mask"
    if accumulate:
        return "accumulate"
    if row["operator"] == "index_put" and math.prod(input_shape) >= 16 * 1024 * 1024:
        return "large_out_of_place"
    if len(index_shapes) == len(input_shape):
        if (
            any(len(shape) > 1 for shape in index_shapes)
            or len({tuple(s) for s in index_shapes}) > 1
        ):
            return "fullrank_broadcast"
        return "fullrank_n1"
    if len(index_shapes) == 1:
        return "continuous_tail"
    return "partial_multidim"


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def export_matrix(rows, path):
    fields = [
        "operator",
        "case_ordinal",
        "dtype",
        "is_bool",
        "accumulate",
        "input_shape",
        "index_shapes",
        "values_shape",
        "torch_ms",
        "candidate_ms",
        "speedup",
        "native_gate_possible",
        "native_kernel_kind",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            input_shape, index_shapes, values_shape, _ = row["shape_data"]
            writer.writerow(
                {
                    "operator": row["operator"],
                    "case_ordinal": row["ordinal"],
                    "dtype": row["dtype"],
                    "is_bool": row["is_bool"],
                    "accumulate": row["accumulate"],
                    "input_shape": repr(input_shape),
                    "index_shapes": repr(index_shapes),
                    "values_shape": repr(values_shape),
                    "torch_ms": row["torch"],
                    "candidate_ms": row["candidate"],
                    "speedup": row["speedup"],
                    "native_gate_possible": current_gate_possible(row),
                    "native_kernel_kind": native_kernel_kind(row),
                }
            )


def export_legacy_csv(rows, path):
    """Write the compact per-run CSV used for review in chat and spreadsheets."""
    fields = ["operator", "dtype", "shape", "torch", "cand", "speedup"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            input_shape, index_shapes, values_shape, accumulate = row["shape_data"]
            writer.writerow(
                {
                    "operator": row["operator"],
                    "dtype": row["dtype"],
                    "shape": repr(
                        [input_shape, index_shapes, values_shape, accumulate]
                    ),
                    "torch": row["torch"],
                    "cand": row["candidate"],
                    "speedup": row["speedup"],
                }
            )


def row_signature(row):
    input_shape, index_shapes, values_shape, accumulate = row["shape_data"]
    return (
        row["operator"],
        row["dtype"],
        tuple(input_shape),
        tuple(tuple(shape) for shape in index_shapes),
        tuple(values_shape),
        bool(accumulate),
    )


def _trace_signature(
    api, dtype, input_shape, index_shapes, values_shape, accumulate, symbol
):
    if symbol == "try_native_index_put_bool_":
        # Mask True count is random and appears as Values.shape[0]; match the
        # official logical case without requiring that per-run dynamic K.
        return (
            "bool",
            api,
            dtype,
            tuple(input_shape),
            tuple(tuple(shape) for shape in index_shapes),
            bool(accumulate),
        )
    if symbol in ("native_index_put_", "try_native_index_put_row_clone_"):
        # The public row path flattens multidimensional indices before the C++
        # call. Preserve the logical update count when pairing it to the
        # official shape instead of requiring the flattened rank to match.
        return (
            "row-tail",
            api,
            dtype,
            tuple(input_shape),
            math.prod(index_shapes[0]) if index_shapes else 0,
            tuple(values_shape),
            bool(accumulate),
        )
    return (
        "exact",
        api,
        dtype,
        tuple(input_shape),
        tuple(tuple(shape) for shape in index_shapes),
        tuple(values_shape),
        bool(accumulate),
    )


def load_actual_hits(path):
    hits = defaultdict(
        lambda: {"symbols": set(), "attempts": 0, "dispatches": 0, "launches": 0}
    )
    if not path.exists():
        return hits
    with path.open(newline="") as stream:
        for item in csv.DictReader(stream):
            signature = _trace_signature(
                item["api"],
                item["dtype"],
                ast.literal_eval(item["input_shape"]),
                ast.literal_eval(item["index_shapes"]),
                ast.literal_eval(item["values_shape"]),
                item.get("accumulate", "False") == "True",
                item["extension_symbol"],
            )
            hits[signature]["symbols"].add(item["extension_symbol"])
            hits[signature]["attempts"] += int(item["attempts"])
            hits[signature]["dispatches"] += int(
                item.get("successful_dispatches", item["kernel_launches"])
            )
            hits[signature]["launches"] += int(item["kernel_launches"])
    return hits


def actual_hit_for_case(row, hits):
    input_shape, index_shapes, values_shape, accumulate = row["shape_data"]
    if row["is_bool"]:
        bool_key = (
            "bool",
            row["operator"],
            row["dtype"],
            tuple(input_shape),
            tuple(tuple(shape) for shape in index_shapes),
            bool(accumulate),
        )
        return hits.get(
            bool_key, {"symbols": set(), "attempts": 0, "dispatches": 0, "launches": 0}
        )
    exact = (
        "exact",
        row["operator"],
        row["dtype"],
        tuple(input_shape),
        tuple(tuple(shape) for shape in index_shapes),
        tuple(values_shape),
        bool(accumulate),
    )
    result = hits.get(exact)
    if result is not None and result["launches"] > 0:
        return result
    if not accumulate and len(input_shape) == 2 and len(index_shapes) == 1:
        row_key = (
            "row-tail",
            row["operator"],
            row["dtype"],
            tuple(input_shape),
            math.prod(index_shapes[0]),
            tuple(values_shape),
            bool(accumulate),
        )
        return hits.get(
            row_key, {"symbols": set(), "attempts": 0, "dispatches": 0, "launches": 0}
        )
    return result or {"symbols": set(), "attempts": 0, "dispatches": 0, "launches": 0}


def pair_matrices(off, on, out_dir):
    paired = []
    for before, after in zip(off, on):
        identity_before = (
            before["operator"],
            before["ordinal"],
            before["dtype"],
            before["accumulate"],
        )
        identity_after = (
            after["operator"],
            after["ordinal"],
            after["dtype"],
            after["accumulate"],
        )
        if identity_before != identity_after:
            raise ValueError(
                f"OFF/ON case order mismatch: {identity_before} != {identity_after}"
            )
        # Bool values length K is random, so pair by official case ordinal and
        # semantic index/input shape, not by the values shape.
        if before["shape_data"][:2] != after["shape_data"][:2]:
            raise ValueError(f"OFF/ON input/index shape mismatch: {identity_before}")
        paired.append((before, after))

    fields = [
        "operator",
        "case_ordinal",
        "dtype",
        "is_bool",
        "accumulate",
        "input_shape",
        "index_shapes",
        "values_shape_off",
        "values_shape_on",
        "torch_off_ms",
        "torch_on_ms",
        "base_off_ms",
        "candidate_on_ms",
        "speedup_off",
        "speedup_on",
        "candidate_on_over_off",
        "native_gate_possible",
        "native_kernel_kind",
    ]
    with (out_dir / "operator_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for before, after in paired:
            ishape, iidx, ivals, _ = before["shape_data"]
            _, _, on_vals, _ = after["shape_data"]
            writer.writerow(
                {
                    "operator": before["operator"],
                    "case_ordinal": before["ordinal"],
                    "dtype": before["dtype"],
                    "is_bool": before["is_bool"],
                    "accumulate": before["accumulate"],
                    "input_shape": repr(ishape),
                    "index_shapes": repr(iidx),
                    "values_shape_off": repr(ivals),
                    "values_shape_on": repr(on_vals),
                    "torch_off_ms": before["torch"],
                    "torch_on_ms": after["torch"],
                    "base_off_ms": before["candidate"],
                    "candidate_on_ms": after["candidate"],
                    "speedup_off": before["speedup"],
                    "speedup_on": after["speedup"],
                    "candidate_on_over_off": after["candidate"] / before["candidate"],
                    "native_gate_possible": current_gate_possible(after),
                    "native_kernel_kind": native_kernel_kind(after),
                }
            )

    actual_hits = load_actual_hits(out_dir / "native_hit_cases.csv")
    coverage_fields = [
        "operator",
        "case_ordinal",
        "dtype",
        "is_bool",
        "accumulate",
        "input_shape",
        "index_shapes",
        "values_shape",
        "predicted_native_kind",
        "actual_native_hit",
        "actual_native_symbols",
        "trace_attempts",
        "trace_dispatches",
        "trace_launches",
        "torch_on_ms",
        "base_off_ms",
        "candidate_on_ms",
        "speedup_on",
        "candidate_on_over_off",
    ]
    with (out_dir / "case_coverage.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=coverage_fields, lineterminator="\n")
        writer.writeheader()
        for before, after in paired:
            hit = actual_hit_for_case(after, actual_hits)
            input_shape, index_shapes, values_shape, accumulate = after["shape_data"]
            writer.writerow(
                {
                    "operator": after["operator"],
                    "case_ordinal": after["ordinal"],
                    "dtype": after["dtype"],
                    "is_bool": after["is_bool"],
                    "accumulate": accumulate,
                    "input_shape": repr(input_shape),
                    "index_shapes": repr(index_shapes),
                    "values_shape": repr(values_shape),
                    "predicted_native_kind": native_kernel_kind(after),
                    "actual_native_hit": hit["launches"] > 0,
                    "actual_native_symbols": ";".join(sorted(hit["symbols"])),
                    "trace_attempts": hit["attempts"],
                    "trace_dispatches": hit["dispatches"],
                    "trace_launches": hit["launches"],
                    "torch_on_ms": after["torch"],
                    "base_off_ms": before["candidate"],
                    "candidate_on_ms": after["candidate"],
                    "speedup_on": after["speedup"],
                    "candidate_on_over_off": after["candidate"] / before["candidate"],
                }
            )

    priority_rows = []
    for before, after in paired:
        hit = actual_hit_for_case(after, actual_hits)
        speedup = after["speedup"]
        deficit = max(0.0, math.log(0.80 / speedup))
        priority_rows.append(
            {
                "case_id": f"{after['operator']}:{after['ordinal']}",
                "api": after["operator"],
                "dtype": after["dtype"],
                "input_shape": repr(after["shape_data"][0]),
                "index_shapes": repr(after["shape_data"][1]),
                "values_shape": repr(after["shape_data"][2]),
                "accumulate": after["accumulate"],
                "category": case_category(after),
                "torch_ms": after["torch"],
                "base_ms": before["candidate"],
                "candidate_ms": after["candidate"],
                "speedup": speedup,
                "native_hit": hit["launches"] > 0,
                "native_symbols": ";".join(sorted(hit["symbols"])),
                "log_deficit_to_0_80": deficit,
                # This is a ceiling: it assumes every lagging case reaches 0.80x.
                "max_recoverable_log_gm_per_case": deficit / 56.0,
            }
        )
    priority_rows.sort(
        key=lambda row: (-row["log_deficit_to_0_80"], row["api"], row["case_id"])
    )
    priority_fields = list(priority_rows[0]) if priority_rows else []
    with (out_dir / "case_priority.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=priority_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(priority_rows)

    category_groups = defaultdict(list)
    for row in priority_rows:
        category_groups[(row["api"], row["category"])].append(row)
    category_fields = [
        "api",
        "category",
        "case_count",
        "native_hits",
        "torch_candidate_geomean",
        "base_candidate_geomean",
        "total_log_deficit_to_0_80",
        "upper_bound_log_gm_gain",
        "worst_case_id",
        "worst_speedup",
    ]
    with (out_dir / "category_priority.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=category_fields, lineterminator="\n")
        writer.writeheader()
        for (api, category), rows in sorted(
            category_groups.items(),
            key=lambda item: -sum(r["log_deficit_to_0_80"] for r in item[1]),
        ):
            worst = min(rows, key=lambda r: r["speedup"])
            writer.writerow(
                {
                    "api": api,
                    "category": category,
                    "case_count": len(rows),
                    "native_hits": sum(r["native_hit"] for r in rows),
                    "torch_candidate_geomean": geomean([r["speedup"] for r in rows]),
                    "base_candidate_geomean": geomean(
                        [r["base_ms"] / r["candidate_ms"] for r in rows]
                    ),
                    "total_log_deficit_to_0_80": sum(
                        r["log_deficit_to_0_80"] for r in rows
                    ),
                    "upper_bound_log_gm_gain": sum(
                        r["max_recoverable_log_gm_per_case"] for r in rows
                    ),
                    "worst_case_id": worst["case_id"],
                    "worst_speedup": worst["speedup"],
                }
            )

    lines = [
        "# Official index_put Operator benchmark audit",
        "",
        (
            "Source: `benchmark/test_index_put.py`, Comprehensive Operator mode. CSV latencies are milliseconds; "
            "multiply by 1000 for microseconds. Both runs used the same source and software environment with "
            "`--warmup 10 --iter 10`; only `FLAGGEMS_MTHREADS_NATIVE_INDEX_PUT` differed."
        ),
        "",
        (
            "| API | Cases | OFF Torch/Candidate GM | ON Torch/Candidate GM | Candidate ON/OFF GM | "
            "Torch ON/OFF GM | Actual Native hits |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    case_change_lines = []
    for operator in ("index_put", "index_put_"):
        pairs = [(a, b) for a, b in paired if a["operator"] == operator]
        off_sp = geomean([a["speedup"] for a, _ in pairs])
        on_sp = geomean([b["speedup"] for _, b in pairs])
        cand_ratio = geomean([b["candidate"] / a["candidate"] for a, b in pairs])
        torch_ratio = geomean([b["torch"] / a["torch"] for a, b in pairs])
        actual_count = sum(
            1
            for _, after in pairs
            if actual_hit_for_case(after, actual_hits)["launches"] > 0
        )
        lines.append(
            f"| `{operator}` | 56 | {off_sp:.6f}x | {on_sp:.6f}x | "
            f"{cand_ratio:.6f}x | {torch_ratio:.6f}x | {actual_count}/56 |"
        )
        ratios = [b["candidate"] / a["candidate"] for a, b in pairs]
        case_change_lines.append(
            f"`{operator}` ON/OFF Candidate cases: improved {sum(x < 0.95 for x in ratios)}, "
            f"within ±5% {sum(0.95 <= x <= 1.05 for x in ratios)}, "
            f"regressed >5% {sum(x > 1.05 for x in ratios)}."
        )
    lines += [
        "",
        *case_change_lines,
        "",
        "## Native-hit and fallback contributions",
        "",
        "| API | Path | Cases | Torch/Candidate GM | Candidate ON/OFF GM |",
        "|---|---|---:|---:|---:|",
    ]
    for operator in ("index_put", "index_put_"):
        pairs = [(a, b) for a, b in paired if a["operator"] == operator]
        for label, selected in (
            (
                "Native hit",
                [
                    (a, b)
                    for a, b in pairs
                    if actual_hit_for_case(b, actual_hits)["launches"] > 0
                ],
            ),
            (
                "Fallback",
                [
                    (a, b)
                    for a, b in pairs
                    if actual_hit_for_case(b, actual_hits)["launches"] == 0
                ],
            ),
        ):
            if not selected:
                continue
            gem_torch = geomean([b["speedup"] for _, b in selected])
            gem_on_off = geomean([b["candidate"] / a["candidate"] for a, b in selected])
            lines.append(
                f"| `{operator}` | {label} | {len(selected)} | {gem_torch:.6f}x | {gem_on_off:.6f}x |"
            )
    lines += ["", "## Five slowest Candidate/Torch cases per API", ""]
    for operator in ("index_put", "index_put_"):
        rows = sorted(
            (r for r in on if r["operator"] == operator), key=lambda r: r["speedup"]
        )
        lines += [
            f"### `{operator}`",
            "",
            "| Case | dtype | accumulate | Bool | Input | Indices | Values | Torch/Candidate | Candidate μs | Native |",
            "|---:|---|---:|---:|---|---|---|---:|---:|---|",
        ]
        for row in rows[:5]:
            input_shape, index_shapes, values_shape, accumulate = row["shape_data"]
            hit = actual_hit_for_case(row, actual_hits)["launches"] > 0
            lines.append(
                f"| {row['ordinal']} | {row['dtype']} | {accumulate} | {row['is_bool']} | "
                f"`{input_shape}` | `{index_shapes}` | `{values_shape}` | "
                f"{row['speedup']:.4f}x | {row['candidate'] * 1000:.3f} | {hit} |"
            )
        lines.append("")
    lines += [
        "",
        (
            "The ON/OFF Torch ratio exposes baseline drift. Attribute an ON speedup to Native only when Candidate "
            "latency improves; a Torch-only shift is not a Native gain."
        ),
        "",
        "## Current candidate hit scope",
        "",
        (
            "`case_coverage.csv` pairs each official case with the dispatch gate, traced extension symbol/count, and "
            "performance. `native_hit_cases.csv` records extension attempts, successful dispatches, and expected "
            "device launches. The experimental Bool prefix path is limited to contiguous FP16/BF16/FP32 inputs "
            "of at most 4096 elements for `accumulate=False`, or 262144 for `accumulate=True`; larger masks and "
            "unsupported shapes fall back."
        ),
        "",
    ]
    (out_dir / "performance_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--off", type=Path, default=Path(__file__).parent / "current_off.json"
    )
    parser.add_argument(
        "--on", type=Path, default=Path(__file__).parent / "current_on.json"
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--flat-csv-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.flat_csv_dir.mkdir(parents=True, exist_ok=True)
    loader = load_json if args.off.suffix == ".json" else load_csv
    off = loader(args.off)
    on = load_json(args.on) if args.on.suffix == ".json" else load_csv(args.on)
    validate_matrix(off, args.off)
    validate_matrix(on, args.on)
    export_matrix(off, args.out_dir / "current_off.csv")
    export_matrix(on, args.out_dir / "current_on.csv")
    export_legacy_csv(off, args.flat_csv_dir / "final_off.csv")
    export_legacy_csv(on, args.flat_csv_dir / "final_on.csv")
    if args.off.suffix == ".json":
        shutil.copyfile(args.off, args.flat_csv_dir / "final_off.json")
    if args.on.suffix == ".json":
        shutil.copyfile(args.on, args.flat_csv_dir / "final_on.json")
    pair_matrices(off, on, args.out_dir)
    print("OFF/ON rows:", len(off), len(on))
    print((args.out_dir / "performance_summary.md").read_text())


if __name__ == "__main__":
    main()
