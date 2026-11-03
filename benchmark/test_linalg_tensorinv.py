import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base

# Shapes for linalg_tensorinv benchmark - these are 2D square matrices for ind=1
TENSORINV_SHAPES = [
    (2, 2),  # 2x2 matrix
    (4, 4),  # 4x4 matrix
    (8, 8),  # 8x8 matrix
    (16, 16),  # 16x16 matrix
    (24, 24),  # 24x24 matrix
]


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr76:op_pytorch_linalg_tensorinv_out_h800_perf_baseline": {
        "sha256": "1e4cdc201a90c32835eb26f87661dfed4cd286e83736849ee2e6beb39379af97",
        "artifact": "op_pytorch_linalg_tensorinv_out_h800_perf_baseline.json",
        "operators": ["linalg_tensorinv_out_current_core"],
        "provenance": {},
    }
}


def _aten_unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate consolidated baseline key: " + key)
        result[key] = value
    return result


def _aten_load_json(raw):
    def reject(value):
        raise ValueError("Nonfinite consolidated baseline constant: " + value)

    return json.loads(raw, object_pairs_hook=_aten_unique_keys, parse_constant=reject)


def _aten_project_snapshot(data, source_id):
    spec = _ATEN_SNAPSHOT_SPECS[source_id]
    manifest = data["_aten_consolidation"]
    if spec.get("auxiliary"):
        entry = manifest["auxiliary_artifacts"][spec["auxiliary"]]
        raw = entry["raw_json"].encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("Consolidated baseline manifest digest differs")
        if _aten_load_json(raw) != entry["content"]:
            raise ValueError(
                "Consolidated manifest content differs from original bytes"
            )
        return raw, None
    source = manifest["sources"][source_id]
    raw = source["raw_json"].encode("utf-8")
    if (
        hashlib.sha256(raw).hexdigest() != spec["sha256"]
        or source["artifact_sha256"] != spec["sha256"]
    ):
        raise ValueError("Consolidated native snapshot digest differs")
    original = _aten_load_json(raw)
    original_ops = {
        key
        for key, value in original.items()
        if isinstance(value, dict) and "shapes" in value
    }
    if set(source["operators"]) != set(spec["operators"]):
        raise ValueError("Consolidated native operator scope differs")
    if source["original_metadata"] != {
        key: value for key, value in original.items() if key not in original_ops
    }:
        raise ValueError("Consolidated collection metadata differs")
    for operator in spec["operators"]:
        entry = source["operators"][operator]
        expected = original[operator]
        if entry["entry_metadata"] != {
            key: value for key, value in expected.items() if key != "shapes"
        } or set(entry["shape_keys"]) != set(expected["shapes"]):
            raise ValueError("Consolidated native entry or case scope differs")
        seen = set()
        for old_key, new_key in entry["shape_keys"].items():
            if (
                new_key in seen
                or new_key != source_id + ":" + old_key
                or data[operator]["shapes"][new_key] != expected["shapes"][old_key]
            ):
                raise ValueError("Consolidated native timing/input record differs")
            seen.add(new_key)
    return raw, source


@lru_cache(maxsize=4)
def _aten_cached_payload(raw, source_id):
    if source_id is None:
        return raw, None
    data = _aten_load_json(raw)
    if "_aten_consolidation" not in data:
        return raw, None
    return _aten_project_snapshot(data, source_id)


def _aten_existing_path(path, source_id=None):
    path = Path(path)
    spec = _ATEN_SNAPSHOT_SPECS.get(source_id, {})
    original_name = Path(spec.get("artifact", spec.get("auxiliary", ""))).name
    if not path.exists() and path.name == original_name:
        unified = path.parent / "op_pytorch_h800_perf_baseline.json"
        if unified.is_file():
            return unified
    return path


def _aten_read_bytes(path, source_id):
    raw = _aten_existing_path(path, source_id).read_bytes()
    return _aten_cached_payload(raw, source_id)[0]


@lru_cache(maxsize=1)
def _aten_container_sha(raw):
    return (
        hashlib.sha256(raw).hexdigest()
        if "_aten_consolidation" in _aten_load_json(raw)
        else None
    )


def _aten_qualify_metadata(metadata, source_id):
    if source_id is None:
        return metadata
    path = _aten_existing_path(metadata["artifact"], source_id)
    container_sha = _aten_container_sha(path.read_bytes())
    if container_sha is None:
        return metadata
    selected_sha = _ATEN_SNAPSHOT_SPECS[source_id]["sha256"]
    reported_sha = metadata["artifact_sha256"]
    if reported_sha not in (selected_sha, container_sha):
        raise ValueError("Reported native baseline digest has no matching scope")
    return {
        **metadata,
        "artifact_sha256_scope": (
            "container_file" if reported_sha == container_sha else "selected_snapshot"
        ),
        "snapshot_source_id": source_id,
        "snapshot_sha256": selected_sha,
        "container_sha256": container_sha,
    }


class LinalgTensorinvBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = TENSORINV_SHAPES

    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        with_out = self.op_name == "linalg_tensorinv_out"
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        # Create an invertible matrix
        # For 2D case, use ind=1 (matrix inverse)
        A = torch.randn(shape, dtype=case.dtype, device=self.device)
        # Make it invertible
        m = shape[0]
        A = A @ A.T + torch.eye(m, dtype=case.dtype, device=self.device) * 0.1
        if with_out:
            # ind=1 on a 2D input yields a same-shape inverse
            out = torch.empty_like(A)
            return A, 1, {"out": out}
        else:
            return A, 1  # ind=1 for matrix inverse


@pytest.mark.linalg_tensorinv
def test_linalg_tensorinv():
    bench = LinalgTensorinvBenchmark(
        op_name="linalg_tensorinv",
        torch_op=torch.linalg.tensorinv,
        # torch.linalg.tensorinv benchmark uses float32 because PyTorch does not support half here.
        dtypes=[torch.float32],
    )
    bench.run()


@dataclass
class _TensorinvOutMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _AscendTensorinvOutReference:
    @cached_property
    def _h800_reference(self):
        path = Path(os.environ["FLAGGEMS_TENSORINV_OUT_H800_BASELINE"]).resolve(
            strict=True
        )
        payload = _aten_read_bytes(
            path, ("pr76:op_pytorch_linalg_tensorinv_out_h800_perf_baseline")
        )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != "1e4cdc201a90c32835eb26f87661dfed4cd286e83736849ee2e6beb39379af97":
            raise ValueError("Expected the reviewed native H800 tensorinv output data")
        data = json.loads(payload)
        key = "linalg_tensorinv_out_current_core"
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        if (
            task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or data[key]["native_api"] != "ops." + key
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "ee71b38185ddfe9214f2e616781637df2c18b37f"
            or environment["source"]["git_dirty"] is not False
            or data["_collection"]["configuration"]["repo"] != "pytorch"
            or data["_collection"]["configuration"]["ncu_enabled"] is not False
            or set(data[key]["shapes"])
            != {str(list(shape)) for shape in TENSORINV_SHAPES}
            or any(
                set(rows) != {"torch.float32"} for rows in data[key]["shapes"].values()
            )
        ):
            raise ValueError(
                "Expected all five clean native H800 tensorinv output records"
            )
        return data[key]["shapes"], _aten_qualify_metadata(
            dict(
                artifact=str(path),
                artifact_sha256=digest,
                operator=key,
                collection_task=task,
                environment=environment,
                input_identity="Same original recipe and full configuration; random payloads differ across devices",
                rng_policy=(
                    "Original builder consumes caller RNG with no per-case reset; "
                    "actual producer checkpoints recorded separately"
                ),
                ratio="Raw native H800 mean latency / actual local candidate latency; no scaling",
                qualification="Shared functional collection, not an isolated performance comparison",
            ),
            ("pr76:op_pytorch_linalg_tensorinv_out_h800_perf_baseline"),
        )

    def get_latency(self, op, *args, **kwargs):
        if not getattr(self, "_tensorinv_native_pending", False):
            return super().get_latency(op, *args, **kwargs)
        self._tensorinv_native_pending = False
        if (
            op is not self.torch_op
            or len(args) != 2
            or type(args[1]) is not int
            or args[1] != 1
            or set(kwargs) != {"out"}
        ):
            raise ValueError("Static tensorinv requires the original native out call")
        inp, ind = args
        out = kwargs["out"]
        if (
            not isinstance(inp, torch.Tensor)
            or not isinstance(out, torch.Tensor)
            or tuple(inp.shape) not in TENSORINV_SHAPES
            or inp.dtype != torch.float32
            or out.dtype != inp.dtype
            or inp.device != out.device
            or inp.device.type != torch.device(self.device).type
            or tuple(out.shape) != tuple(inp.shape)
            or not inp.is_contiguous()
            or not out.is_contiguous()
            or inp.storage_offset() != 0
            or out.storage_offset() != 0
            or inp.requires_grad
            or out.requires_grad
            or inp.untyped_storage()._cdata == out.untyped_storage()._cdata
        ):
            raise ValueError(
                "Tensorinv input/out differs from the native H800 contract"
            )
        tensor = dict(
            shape=list(inp.shape),
            dtype=str(inp.dtype),
            stride=list(inp.stride()),
            requires_grad=False,
        )
        config = dict(
            inputs=dict(args=[tensor, ind], kwargs=dict(out=dict(tensor))),
            source_commit="2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
            benchmark="benchmark/test_linalg_tensorinv.py",
            scope="core",
            descriptor=list(inp.shape),
            variant=0,
            input_recipe=dict(
                matrix="torch.randn(shape,dtype,device); matrix@matrix.T+0.1*eye(n)",
                rng_policy="consume caller RNG stream; no per-case seed reset",
                out="torch.empty_like(matrix)",
                native="torch.linalg.tensorinv(matrix,1,out=out)",
            ),
        )
        records, provenance = self._h800_reference
        matches = [
            record
            for rows in records.values()
            for dtype, record in rows.items()
            if dtype == str(inp.dtype)
            and {k: v for k, v in record["config"].items() if k != "producer_sample"}
            == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one H800 record matching the complete tensorinv call"
            )
        record = matches[0]
        sample = record["config"]["producer_sample"]
        if (
            sample["rng_device_type"] != "cuda"
            or sample["input_bytes"] != inp.numel() * inp.element_size()
            or len(sample["input_sha256"]) != 64
            or len(bytes.fromhex(sample["rng_state_before_hex"])) < 16
            or record["timing"]
            != dict(
                method="triton.testing.do_bench",
                return_mode="mean",
                warmup_ms=25,
                rep_ms=100,
            )
        ):
            raise ValueError("Native tensorinv sample or timing provenance differs")
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (int, float))
            or not math.isfinite(us)
            or us <= 0
            or not math.isclose(
                record["latency_ms"], us / 1000, rel_tol=1e-9, abs_tol=1e-12
            )
        ):
            raise ValueError(
                "Native tensorinv latency must be positive with consistent units"
            )
        self._tensorinv_imported_reference = dict(
            **provenance, config=config, producer_sample=sample
        )
        return us / 1000

    def _measure_input(self, input, case_id=None):
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        self._tensorinv_native_pending = (
            flag_gems.vendor_name == "ascend"
            and self.op_name == "linalg_tensorinv_out"
            and needs_native
        )
        self._tensorinv_imported_reference = None
        try:
            if self._tensorinv_native_pending and (
                self.is_backward or "tflops" in self.to_bench_metrics
            ):
                raise ValueError(
                    "Static tensorinv covers original forward latency only"
                )
            if self._tensorinv_native_pending and not os.environ.get(
                "FLAGGEMS_TENSORINV_OUT_H800_BASELINE"
            ):
                raise ValueError("Missing native H800 tensorinv output reference")
            metric = super()._measure_input(input, case_id)
            if self._tensorinv_imported_reference is None:
                return metric
            if metric.latency is not None and (
                not math.isfinite(metric.latency) or metric.latency <= 0
            ):
                raise ValueError(
                    "Local tensorinv candidate latency must be positive and finite"
                )
            if "speedup" in self.to_bench_metrics:
                metric.speedup = metric.latency_base / metric.latency
            return _TensorinvOutMetrics(
                **asdict(metric), baseline_source=self._tensorinv_imported_reference
            )
        finally:
            self._tensorinv_native_pending = False


class LinalgTensorinvOutBenchmark(
    _AscendTensorinvOutReference, LinalgTensorinvBenchmark
):
    pass


@pytest.mark.linalg_tensorinv_out
def test_linalg_tensorinv_out():
    bench = LinalgTensorinvOutBenchmark(
        op_name="linalg_tensorinv_out",
        torch_op=torch.linalg.tensorinv,
        # torch.linalg.tensorinv benchmark uses float32 because PyTorch does not support half here.
        dtypes=[torch.float32],
    )
    bench.run()
