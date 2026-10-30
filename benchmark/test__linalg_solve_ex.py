# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# Measured with valid rank>=2 operands: float32, complex64, float64 and
# complex128 are solved natively. Only the 64-bit types follow the backend
# capability flag; complex64 does not need fp64 support.
SUPPORTED_DTYPES = [torch.float32, torch.complex64]
if flag_gems.runtime.device.support_fp64:
    SUPPORTED_DTYPES += [torch.float64, torch.complex128]

# Performance descriptors for the operator geometry (batch..., n, m): solvers
# are dominated by the n x n factorization, so the grid sweeps the matrix size,
# the right-hand-side width and the batch depth. No element or memory cap is
# applied.
PERFORMANCE_DESCRIPTORS = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 512),
    (16, 256, 128),
    (8, 4, 128, 64),
]

# Zero-extent operands are accepted natively, so the empty-system boundaries
# stay in the default grid (measured: A (0,0)/B (0,0), A (64,64)/B (64,0) and
# A (2,0,0)/B (2,0,0) return without error for both left modes).
ZERO_DESCRIPTORS = [(0, 0), (64, 0), (2, 0, 0)]


def _validate_descriptor(shape):
    """Reject anything that cannot describe an A of shape (batch..., n, n).

    Descriptors arrive from the default list or from a user shape file; both
    paths go through the case builder, so an invalid entry fails loudly instead
    of listing successfully and breaking later while building inputs.
    """
    if not isinstance(shape, (tuple, list)):
        raise ValueError(f"Benchmark descriptor must be a tuple or list, got {shape!r}")
    if len(shape) < 2:
        raise ValueError(f"aten::_linalg_solve_ex needs rank >= 2, got {shape!r}")
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise ValueError(f"Descriptor extents must be integers, got {shape!r}")
        if extent < 0:
            raise ValueError(f"Descriptor extents must be nonnegative, got {shape!r}")
    return tuple(shape)


def _operand_shapes(shape, left):
    """A/B shapes for one descriptor with every batch axis preserved.

    A is (batch..., n, n) with n = shape[-2]; B is (batch..., n, m) for
    AX = B and (batch..., m, n) for XA = B, with m = shape[-1].
    """
    batch = tuple(shape[:-2])
    n, m = shape[-2], shape[-1]
    return batch + (n, n), (batch + (n, m) if left else batch + (m, n))


def _case_fn(shape, dtype):
    del dtype
    descriptor = _validate_descriptor(shape)
    for left in (True, False):
        a_shape, b_shape = _operand_shapes(descriptor, left)
        yield base.BenchmarkCasePlan(
            shape={"A": a_shape, "B": b_shape},
            params={"left": left},
            builder_args=(descriptor, left),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, left = plan.builder_args
    a_shape, b_shape = _operand_shapes(shape, left)
    # generate_tensor_input only covers consts.FLOAT_DTYPES and returns None for
    # float64/complex128, so the operands are built directly; torch.randn
    # supports all four natively solved types.
    cpu_fixture = flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype.is_complex
    fixture_device = "cpu" if cpu_fixture else device
    a = torch.randn(a_shape, dtype=dtype, device=fixture_device)
    a.diagonal(dim1=-2, dim2=-1).add_(a_shape[-1])
    b = torch.randn(b_shape, dtype=dtype, device=fixture_device)
    if cpu_fixture:
        a, b = (
            value.contiguous()
            .view(torch.uint8)
            .to(device)
            .view(dtype)
            .reshape(value.shape)
            for value in (a, b)
        )
    for value, shape in ((a, a_shape), (b, b_shape)):
        assert value.dtype == dtype and tuple(value.shape) == shape
        target = torch.device(device)
        assert value.device == (
            target
            if target.type == "cpu"
            else torch.device(target.type, target.index or 0)
        )
    return a, b, {"left": left}


DEFAULT_DESCRIPTORS = [
    _validate_descriptor(shape) for shape in PERFORMANCE_DESCRIPTORS + ZERO_DESCRIPTORS
]


_H800_ARTIFACT_SHA256 = (
    "17929cff2a1c2cd2a45c01385d662e4dbd4f0cac9436f1bf0f6f81edf927d79a"
)
_H800_OPS_COMMIT = "eb2d2246e1a8ed99a85e2106363e844e021b1e60"


@dataclass
class _SolveMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _HiddenSolveH800Baseline:
    @cached_property
    def _h800_baseline(self):
        path = Path(os.environ["FLAGGEMS_HIDDEN_SOLVE_H800_BASELINE"]).resolve(
            strict=True
        )
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != _H800_ARTIFACT_SHA256:
            raise ValueError("Expected the reviewed hidden four-output H800 artifact")
        data = json.loads(payload)
        key = "_linalg_solve_ex"
        task = data["_collection"]["tasks"][key]
        environment = data["_environments"][str(task["device"])]
        if (
            data[key]["native_api"] != "ops._linalg_solve_ex"
            or task["status"] != "completed"
            or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"] != _H800_OPS_COMMIT
            or environment["source"]["git_dirty"] is not False
            or data["_collection"]["configuration"]["repo"] != "pytorch"
        ):
            raise ValueError("Expected a completed native hidden four-output H800 run")
        return data[key]["shapes"], {
            "artifact": str(path),
            "artifact_sha256": _H800_ARTIFACT_SHA256,
            "operator": key,
            "collection_task": task,
            "environment": environment,
            "input_identity": "Exact original geometry, flags, dtype and recipe; random payloads differ",
            "ratio": "raw H800 latency / locally timed candidate; no peak normalization",
            "environment_qualification": (
                "Functional latency collection with sharing allowed; "
                "no isolated performance claim"
            ),
        }

    def get_latency(self, op, *args, **kwargs):
        ascend_native = (
            flag_gems.vendor_name == "ascend"
            and getattr(self, "_hidden_asc_pending", False)
            and op is self.torch_op
        )
        if not ascend_native and (
            flag_gems.vendor_name != "thead" or op is not self.torch_op
        ):
            return super().get_latency(op, *args, **kwargs)
        if ascend_native:
            self._hidden_asc_pending = False
        if args[0].device.type != torch.device(self.device).type:
            raise ValueError("H800 comparison requires actual target-device inputs")
        if (
            self.is_backward
            or base.Config.skip_native
            or "tflops" in self.to_bench_metrics
        ):
            raise ValueError(
                "Hidden solve H800 baseline requires original forward core timing"
            )
        config = _h800_input_config(args, kwargs)
        shapes, provenance = self._h800_baseline
        matches = [
            record
            for by_dtype in shapes.values()
            for dtype, record in by_dtype.items()
            if dtype == str(args[0].dtype) and record.get("config") == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one four-output H800 record for the full core identity"
            )
        record = matches[0]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (float, int))
            or not math.isfinite(us)
            or us <= 0
        ):
            raise ValueError("H800 latency must be finite and positive")
        ms = us / 1000
        if not math.isclose(record["latency_ms"], ms, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("H800 latency units disagree")
        self._imported_baseline = {
            **provenance,
            "config": config,
            "target_timing_mode": base.Config.mode.value,
            "target_input_factory": (
                "CPU complex randn and diagonal conditioning, lossless bytes to actual target"
                if flag_gems.vendor_name in ("ascend", "kunlunxin")
                and args[0].is_complex()
                else "original requested-device randn and diagonal conditioning"
            ),
        }
        return ms

    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if flag_gems.vendor_name == "thead" and op is self.torch_op:
            raise ValueError("PPU hidden solve requires an explicit safe candidate")
        return op, dispatch, overridden

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name != "ascend":
            return self._measure_imported_input(input, case_id)
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        if not needs_native:
            return super()._measure_input(input, case_id)
        if not os.environ.get("FLAGGEMS_HIDDEN_SOLVE_H800_BASELINE"):
            raise ValueError(
                "Ascend hidden solve uses CPU fallback; set FLAGGEMS_HIDDEN_SOLVE_H800_BASELINE to the reviewed native four-output H800 artifact"
            )
        if self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "Static hidden solve baseline requires original forward latency"
            )
        self._hidden_asc_pending = True
        try:
            return self._measure_imported_input(input, case_id)
        finally:
            self._hidden_asc_pending = False

    def _measure_imported_input(self, input, case_id=None):
        self._imported_baseline = None
        metric = super()._measure_input(input, case_id)
        if self._imported_baseline is None:
            return metric
        if metric.latency is not None and (
            not math.isfinite(metric.latency) or metric.latency <= 0
        ):
            raise ValueError("Target candidate latency must be finite and positive")
        if "speedup" in self.to_bench_metrics:
            metric.speedup = metric.latency_base / metric.latency
        return _SolveMetrics(**asdict(metric), baseline_source=self._imported_baseline)


def _h800_input_config(args, kwargs):
    if len(args) != 2 or set(kwargs) != {"left"} or type(kwargs["left"]) is not bool:
        raise ValueError("Hidden solve baseline requires the original A/B/left ABI")
    a, b = args
    left = kwargs["left"]
    if a.ndim < 2 or b.ndim != a.ndim or a.shape[-1] != a.shape[-2]:
        raise ValueError("Hidden solve core rank differs")
    descriptor = tuple(a.shape[:-2]) + (
        a.shape[-1],
        b.shape[-1] if left else b.shape[-2],
    )
    a_shape, b_shape = _operand_shapes(descriptor, left)
    if descriptor not in DEFAULT_DESCRIPTORS:
        raise ValueError("Descriptor differs from the nine original core descriptors")
    for tensor, shape in ((a, a_shape), (b, b_shape)):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype
            not in (torch.float32, torch.complex64, torch.float64, torch.complex128)
            or tensor.dtype != a.dtype
            or tensor.requires_grad
            or tensor.storage_offset() != 0
            or not tensor.is_contiguous()
            or tensor.device != a.device
        ):
            raise ValueError("Hidden solve baseline inputs differ from original recipe")

    def tensor(x):
        return {
            "shape": list(x.shape),
            "stride": list(x.stride()),
            "storage_offset": 0,
            "dtype": str(x.dtype),
            "requires_grad": False,
        }

    return {
        "native_expression": "torch.ops.aten._linalg_solve_ex.default",
        "native_schema": (
            "aten::_linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, "
            "bool check_errors=False) -> (Tensor result, Tensor LU, Tensor pivots, Tensor info)"
        ),
        "source_commit": "2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
        "source_benchmark_sha256": "290a441e2eecd795d1d5212d74d5174746968e16a92b284f72c4c71f29233ce1",
        "source_builder_ast_sha256": "674a644e5239cca08107c1e588ab60f1777a3432c4e4a27581a563ca473b1118",
        "benchmark": "benchmark/test__linalg_solve_ex.py",
        "builder": "_build_inputs_fn",
        "scope": "core",
        "descriptor": list(descriptor),
        "variant": 0 if left else 1,
        "recipe": {
            "A": "torch.randn; diagonal add n",
            "B": "torch.randn",
            "on_requested_device": True,
            "seed": "original unseeded recipe",
        },
        "inputs": {"args": [tensor(a), tensor(b)], "kwargs": {"left": left}},
        "outputs": [
            {"name": "result", "shape": list(b.shape), "dtype": str(a.dtype)},
            {"name": "LU", "shape": list(a.shape), "dtype": str(a.dtype)},
            {"name": "pivots", "shape": list(a.shape[:-1]), "dtype": "torch.int32"},
            {"name": "info", "shape": list(a.shape[:-2]), "dtype": "torch.int32"},
        ],
        "effects": {"mutates": [], "returns_alias_of": {}},
    }


class SolveExBenchmark(_HiddenSolveH800Baseline, OperatorBenchmark):
    """Two-phase benchmark using the operator's own operand geometry.

    GenericBenchmark's default shapes contain 0-dim and 1-dim tensors, which
    aten::_linalg_solve_ex rejects (A must be at least 2-D), so the default
    descriptor list is replaced while an explicitly configured --shape-file
    still takes precedence for this operator name and class key.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=DEFAULT_DESCRIPTORS)


@pytest.mark.underscore_linalg_solve_ex
def test__linalg_solve_ex():
    bench = SolveExBenchmark(
        op_name="_linalg_solve_ex",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._linalg_solve_ex,
        dtypes=SUPPORTED_DTYPES,
    )
    bench.run()
