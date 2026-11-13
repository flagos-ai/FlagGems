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

import statistics
import time
from contextlib import nullcontext
from functools import partial

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# The ``indices_size`` and ``indices`` overloads build a sparse COO tensor from
# indices and values, with or without an explicit logical size. Construction
# work scales with the stored nnz and the number of sparse dims, not with the
# dense logical size, so each case pairs a large logical tensor shape with a
# bounded nnz. Every shape keeps a 2-D sparse grid (sparse_dim == 2) with the
# remaining dims dense, matching the dominant COO usage; the last case carries
# one dense dim. Each pair is a (tensor_shape, nnz) case.
_BENCH_SHAPES = [
    ((4096, 4096), 10000),
    ((4096, 4096), 100000),
    ((1024, 8192), 50000),
    ((8192, 1024), 50000),
    ((2048, 2048, 64), 20000),
]


def _make_coo_inputs(shape, nnz, dtype, device, seed=0):
    # Generate bounded coordinates for both size overloads. The first two
    # dimensions are sparse and any trailing dimensions are dense.
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:2]
    dense_shape = shape[2:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    ).to(device)
    values = utils.generate_tensor_input((nnz,) + tuple(dense_shape), dtype, device)
    return indices, values


def _case_fn(shape, dtype, *, infer_size):
    del dtype
    tensor_shape, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": tensor_shape},
        params={"nnz": nnz, "infer_size": infer_size},
        builder_args=(tensor_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    tensor_shape, nnz = plan.builder_args
    indices, values = _make_coo_inputs(tensor_shape, nnz, dtype, device)
    # The trailing dict is unpacked into kwargs by the benchmark runner.
    # Only the explicit-size overload receives the logical size.
    options = {"dtype": dtype, "device": device}
    if plan.params["infer_size"]:
        return indices, values, options
    return indices, values, list(tensor_shape), options


class SparseCooTensorBenchmark(OperatorBenchmark):
    def _candidate_call(self):
        registry = base.Config.override_registry
        override = registry.get_override(self.op_name) if registry is not None else None
        candidate = (
            override
            if override is not None
            else getattr(flag_gems, self.op_name, None)
        )
        if not callable(candidate):
            raise RuntimeError(f"No callable candidate for {self.op_name}")
        return candidate, nullcontext(), override is not None

    def get_latency(self, op, *args, **kwargs):
        if getattr(base.Config, "reference_only", False):
            return super().get_latency(op, *args, **kwargs)
        # Sparse-container construction is host work. Match master's
        # synchronized wall-clock samples, including inferred-size transfers.
        for _ in range(10):
            op(*args, **kwargs)
        samples = []
        for _ in range(5):
            torch_device_fn.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                op(*args, **kwargs)
            torch_device_fn.synchronize()
            samples.append((time.perf_counter() - start) * 1000 / 100)
        return statistics.median(samples)

    # sparse_coo_tensor is a sparse construction op; there are no meaningful
    # dense shapes in core_shapes.yaml, so benchmark dedicated
    # (tensor_shape, nnz) pairs instead.
    def set_shapes(self, shape_file_path=None):
        self.shapes = _BENCH_SHAPES
        self.shape_desc = "logical shape, nnz, dense shape; synchronized wall time"

    def set_more_shapes(self):
        return []


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("infer_size", [False, True], ids=["explicit", "inferred"])
def test_sparse_coo_tensor(infer_size, monkeypatch):
    monkeypatch.setattr(base.Config, "mode", consts.BenchMode.OPERATOR)
    bench = SparseCooTensorBenchmark(
        op_name="sparse_coo_tensor",
        case_fn=partial(_case_fn, infer_size=infer_size),
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_coo_tensor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
