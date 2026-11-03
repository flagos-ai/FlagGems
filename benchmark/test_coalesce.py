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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# (sparse shape, nnz) workload descriptors. Coalescing work scales with nnz, so
# the tensors are sized for a meaningful measurement while staying small enough
# to build repeatedly; nnz is always > numel(shape), which guarantees duplicate
# coordinates and therefore real sort/merge work.
_COALESCE_SHAPES = [
    ((1024, 1024), 65536),
    ((1024, 1024), 262144),
    ((1024, 1024), 1048576),
    ((4096, 4096), 1048576),
    ((2048, 2048), 2097152),
    ((256, 256, 256), 1048576),
]


_OP_NAME = "coalesce"
_NV_ARTIFACT_SHA256 = "ed09649e4b5e64b94c6c610cc01422117f792ee8316524f3222711e4c5f3f4c3"
_NV_ORIGINAL = {
    "environment": {
        "schema_version": 1,
        "collected_at_utc": "2026-10-16T12:08:09.647837+00:00",
        "hardware": {
            "gpu": {
                "name": "NVIDIA H800",
                "uuid": "f76856dc-d02f-95a0-dc05-054190d1ea7d",
                "total_memory_bytes": 85017624576,
                "total_memory_gib": 79.18,
                "compute_capability": "9.0",
                "multiprocessor_count": 132,
                "cuda_device_index": 0,
                "driver_version": "580.105.08",
            },
            "cpu_model": "Intel(R) Xeon(R) Platinum 8468V",
            "cpu_architecture": "x86_64",
            "cpu_socket_count": 2,
            "physical_cpu_core_count": 96,
            "logical_cpu_count": 192,
            "cpu_threads_per_core": 2,
            "numa_node_count": 2,
            "process_cpu_affinity_count": 192,
            "hypervisor_vendor": None,
        },
        "software": {
            "os": "Linux",
            "os_release": "5.15.0-78-generic",
            "machine": "x86_64",
            "python": "3.12.3",
            "packages": {
                "torch": "2.8.0a0+5228986c39.nv25.5",
                "triton": "3.6.0",
                "vllm": None,
                "sglang": None,
            },
            "torch_cuda_version": "12.9",
            "nvcc_version": "nvcc: NVIDIA (R) Cuda compiler driver\n"
            "Copyright (c) 2005-2025 NVIDIA Corporation\n"
            "Built on Wed_Apr__9_19:24:57_PDT_2025\n"
            "Cuda compilation tools, release 12.9, V12.9.41\n"
            "Build cuda_12.9.r12.9/compiler.35813241_0",
            "ncu_version": "NVIDIA (R) Nsight Compute Command Line Profiler\n"
            "Copyright (c) 2018-2025 NVIDIA Corporation\n"
            "Version 2025.2.0.0 (build 35613519) "
            "(public-release)",
        },
        "source": {
            "git_commit": "ccc0508d81b3b10283a62dab73fc05e256093848",
            "git_branch": "feat/multi-gpu-collection",
            "git_dirty": False,
        },
        "run": {
            "repo": "pytorch",
            "device": None,
            "ncu_enabled": True,
            "report_dir": (
                "/data/tj/workspace/ops-baseline-multigpu-20261016/runs/pytor"
                "ch_h800.json.work/coalesce/ncu_reports"
            ),
            "output_path": (
                "/data/tj/workspace/ops-baseline-multigpu-20261016/runs/pytor"
                "ch_h800.json.work/coalesce/result.json"
            ),
            "op_timeout_seconds": 3600.0,
            "reference_chip": "H800",
            "only": ["coalesce"],
            "whitelist": None,
            "blacklist": None,
            "selected_ops": ["coalesce"],
            "latency_warmup_ms": 25,
            "latency_rep_ms": 100,
            "ncu_warmup_calls": 3,
        },
        "environment_variables": {
            "CUDA_VISIBLE_DEVICES": "6",
            "CUDA_DEVICE_ORDER": None,
        },
    },
    "original_partial_sha256": "ede8327400a958a1c9dd30c6a7ff6788220f33b09e8cbd39f2f7dc86c92f94be",
}


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "legacy:pytorch-h800-primary": {
        "legacy_primary": True,
        "artifact": "op_pytorch_h800_perf_baseline.json",
        "sha256": "ed09649e4b5e64b94c6c610cc01422117f792ee8316524f3222711e4c5f3f4c3",
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
    if spec.get("legacy_primary"):
        if manifest["base_artifact_sha256"] != spec["sha256"]:
            raise ValueError("Unexpected historical native H800 snapshot")
        original = {
            key: value for key, value in data.items() if key != "_aten_consolidation"
        }
        copied = set()
        for source in manifest["sources"].values():
            for operator, entry in source["operators"].items():
                if operator not in copied:
                    original[operator] = dict(original[operator])
                    original[operator]["shapes"] = dict(original[operator]["shapes"])
                    copied.add(operator)
                for new_key in entry["shape_keys"].values():
                    del original[operator]["shapes"][new_key]
        for operator in copied:
            if not original[operator]["shapes"]:
                del original[operator]
        raw = (
            json.dumps(
                original, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("Historical native snapshot could not be restored exactly")
        return raw, None
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


def _cpu_oracle(inp):
    """CPU-native ref-as-candidate bootstrap for the original no-grad cases."""
    assert inp.layout == torch.sparse_coo and not inp.requires_grad
    if _OP_NAME == "coalesce" and inp.is_coalesced():
        return inp
    cpu = torch.sparse_coo_tensor(
        inp._indices().cpu(),
        inp._values().cpu(),
        inp.shape,
        dtype=inp.dtype,
        is_coalesced=inp.is_coalesced(),
    )
    result = getattr(torch.ops.aten, _OP_NAME)(cpu)
    if inp.device.type == "cpu":
        return result
    return torch.sparse_coo_tensor(
        result._indices().to(inp.device),
        result._values().to(inp.device),
        result.shape,
        dtype=result.dtype,
        is_coalesced=result.is_coalesced(),
    )


@dataclass
class _CrossMachineMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


def _case_fn(shape, dtype):
    # Two-phase GenericBenchmark: the shape entries are the (shape, nnz)
    # workload descriptors; emitting one BenchmarkCasePlan per descriptor keeps
    # nnz in the case id/params and defers tensor construction.
    del dtype
    sparse_shape, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": sparse_shape},
        params={"nnz": nnz},
        builder_args=(sparse_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    # Draw coordinates with replacement so the input is uncoalesced (the CUDA
    # reference asserts on that) and coalescing has to sort/merge.
    sparse_shape, nnz = plan.builder_args
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, device=device)
            for dim in sparse_shape
        ]
    )
    values = torch.randn(nnz, dtype=dtype, device=device)
    inp = torch.sparse_coo_tensor(indices, values, sparse_shape, device=device)
    return inp, {}


class CoalesceBenchmark(OperatorBenchmark):
    # coalesce is a sparse op, so there are no meaningful dense shapes in
    # core_shapes.yaml; benchmark the dedicated (shape, nnz) descriptors above.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_COALESCE_SHAPES)

    @cached_property
    def _nvidia_baseline(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = _aten_read_bytes(path, ("legacy:pytorch-h800-primary"))
        if hashlib.sha256(payload).hexdigest() != _NV_ARTIFACT_SHA256:
            raise ValueError("Expected the reviewed ops-baseline H800 snapshot")

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate baseline JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"Non-finite baseline JSON constant: {value}")

        data = json.loads(
            payload, object_pairs_hook=unique_keys, parse_constant=reject_constant
        )
        task = data["_collection"]["tasks"][self.op_name]
        device_environment = data["_environments"][str(task["device"])]
        # Published per-device metadata may describe a later task. The retained
        # original per-operator environment is separately pinned below.
        original = _NV_ORIGINAL["environment"]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or data[self.op_name]["native_api"] != f"ops.{self.op_name}"
            or original["run"]["selected_ops"] != [self.op_name]
            or original["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or device_environment["hardware"]["gpu"]["uuid"]
            != original["hardware"]["gpu"]["uuid"]
            or device_environment["source"] != original["source"]
            or original["source"]["git_dirty"] is not False
        ):
            raise ValueError("Incomplete or inconsistent original H800 trace")
        shapes = data[self.op_name]["shapes"]
        if sum(len(dtypes) for dtypes in shapes.values()) != 18:
            raise ValueError("Expected the original eighteen H800 core records")
        return shapes, {
            "kind": "external_nvidia",
            "artifact": str(path),
            "artifact_sha256": _NV_ARTIFACT_SHA256,
            "published_revision": "7d43707757d04910585b0b97f9bdcb56faa1e56e",
            "original_partial_sha256": _NV_ORIGINAL["original_partial_sha256"],
            "environment": original,
            "ratio": "(H800 latency_ms + EPS) / (target latency_ms + EPS); no peak normalization",
            "target_timing_mode": base.Config.mode.value,
        }

    def _external_input_config(self, args):
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError("Expected the original single COO input")
        inp = args[0]
        if inp.layout != torch.sparse_coo:
            raise ValueError("External baseline requires a COO input")
        expected_device = torch.device(flag_gems.device)
        if expected_device.type != "cpu" and expected_device.index is None:
            expected_device = torch.device(
                expected_device.type, base.torch_device_fn.current_device()
            )
        shape, nnz = tuple(inp.shape), inp._nnz()
        indices, values = inp._indices(), inp._values()
        if (
            inp.layout != torch.sparse_coo
            or inp.device != expected_device
            or inp.is_coalesced()
            or inp.requires_grad
            or (shape, nnz) not in _COALESCE_SHAPES
            or inp.dtype not in consts.FLOAT_DTYPES
            or inp.sparse_dim() != len(shape)
            or inp.dense_dim() != 0
            or inp._indices().dtype != torch.int64
            or tuple(inp._indices().shape) != (len(shape), nnz)
            or tuple(inp._values().shape) != (nnz,)
            or inp._indices().device != inp.device
            or inp._values().device != inp.device
            or inp._values().dtype != inp.dtype
            or tuple(indices.stride()) != (nnz, 1)
            or tuple(values.stride()) != (1,)
            or indices.storage_offset() != 0
            or values.storage_offset() != 0
            or indices.is_conj()
            or indices.is_neg()
            or values.is_conj()
            or values.is_neg()
        ):
            raise ValueError(
                "External baseline requires the original uncoalesced COO recipe"
            )
        # ops-baseline's sparse descriptor uses zero stride placeholders;
        # these are a serialization convention, not physical COO strides.
        return {
            "inputs": {
                "args": [
                    {
                        "shape": list(shape),
                        "dtype": str(inp.dtype),
                        "stride": [0] * len(shape),
                        "requires_grad": False,
                    }
                ],
                "kwargs": {},
            },
            "source_commit": "9d57a23b4386bd669152a935fe88a96db907da7c",
            "benchmark": f"benchmark/test_{self.op_name}.py",
            "descriptor": [list(shape), nnz],
            "scope": "core",
            "variant": 0,
        }

    def _import_nvidia_latency(self, args, kwargs):
        if kwargs or self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "External baseline supports these forward positional cases only"
            )
        config = self._external_input_config(args)
        shapes, provenance = self._nvidia_baseline
        matches = [
            record
            for dtypes in shapes.values()
            for dtype, record in dtypes.items()
            if dtype == str(args[0].dtype) and record.get("config") == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one NVIDIA record matching the actual inputs"
            )
        record = matches[0]
        latency_us = record.get("T_us")
        if (
            isinstance(latency_us, bool)
            or not isinstance(latency_us, (int, float))
            or not math.isfinite(latency_us)
            or latency_us <= 0
        ):
            raise ValueError("NVIDIA T_us must be finite and positive")
        latency_ms = latency_us / 1000
        recorded_ms = record.get("latency_ms", latency_ms)
        if (
            isinstance(recorded_ms, bool)
            or not isinstance(recorded_ms, (int, float))
            or not math.isfinite(recorded_ms)
            or not math.isclose(recorded_ms, latency_ms, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError("NVIDIA latency units disagree")
        self._imported_baseline = {
            **provenance,
            "config": config,
            "target_environment": {
                "vendor": flag_gems.vendor_name,
                "device": str(args[0].device),
                "device_name": base.torch_device_fn.get_device_name(),
                "torch": str(torch.__version__),
            },
        }
        if base.Config.reference_only:
            # The base runner creates all records before traversing inputs.
            active = [
                r
                for r in base.Config.reference_records
                if r.get("operator") == self.op_name
                and r.get("status") == "FAILED"
                and r.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise RuntimeError("Cannot identify the active external-reference case")
            active[0]["baseline_source"] = self._imported_baseline
        return latency_ms

    def get_latency(self, op, *args, **kwargs):
        importing = os.environ.get("FLAGGEMS_NV_BASELINE") and (
            base.Config.reference_only
            or getattr(self, "_native_measurement_pending", False)
        )
        if importing:
            if base.Config.skip_native or op is not self.torch_op:
                raise ValueError(
                    "External baseline requires the original reference measurement"
                )
            # Only the first, reference measurement imports. Even an identical
            # candidate callable must execute through the real timer next.
            self._native_measurement_pending = False
            return self._import_nvidia_latency(args, kwargs)
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        self._imported_baseline = None
        self._native_measurement_pending = False
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if external:
            if self.is_backward or "tflops" in self.to_bench_metrics:
                raise ValueError(
                    "External baseline supports forward latency measurements only"
                )
            if base.Config.skip_native:
                raise ValueError("Do not combine an external baseline with skip_native")
        self._native_measurement_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        try:
            metric = super()._measure_input(input, case_id)
            if external and metric.latency is not None:
                if (
                    isinstance(metric.latency, bool)
                    or not isinstance(metric.latency, (int, float))
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError("Target latency must be finite and positive")
            if self._imported_baseline is not None:
                return _CrossMachineMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._native_measurement_pending = False

    def _run_reference_cases(self, case_ids):
        first_record = len(base.Config.reference_records)
        if os.environ.get("FLAGGEMS_NV_BASELINE") and base.Config.skip_native:
            raise ValueError("Do not combine an external baseline with skip_native")
        try:
            return super()._run_reference_cases(case_ids)
        finally:
            for record in base.Config.reference_records[first_record:]:
                if record.get("baseline_source") and record.get("status") == "PASSED":
                    record["stage"] = "external_baseline"


@pytest.mark.coalesce
def test_coalesce():
    bench = CoalesceBenchmark(
        op_name="coalesce",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        # flag_gems has no public ``coalesce`` direct callable in this checkout;
        # the candidate is supplied by the KernelGen process-local override keyed
        # on the public operator name "coalesce" (resolved via
        # Benchmark._candidate_call inside the benchmark).
        torch_op=torch.ops.aten.coalesce,
        gems_op=getattr(flag_gems, "coalesce", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
