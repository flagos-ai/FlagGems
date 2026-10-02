# Copyright 2026, The FlagGems Authors.
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

import pytest
import torch

import flag_gems

from . import base

# Extra int4 GEMM weight geometries, added through set_more_shapes; every entry
# is 2-D with size(0) divisible by 16 and even size(1).
_EXTRA_SHAPES = [
    (128, 4096),
    (1024, 1024),
    (2048, 2048),
    (4096, 11008),
]

# innerKTiles belongs to the call form; the native kernel ignores its value.
_INNER_K_TILES = (2, 4, 8)


def _case_fn(shape, dtype):
    """Tensor-free plans: --list-cases allocates nothing and calls no operator."""
    del dtype
    for inner_k_tiles in _INNER_K_TILES:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"innerKTiles": inner_k_tiles},
            builder_args=(shape, inner_k_tiles),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, inner_k_tiles = plan.builder_args
    # This CPU-only operator rejects accelerator tensors, so the argument is
    # built on CPU and the runner-provided device is intentionally unused. The
    # weight values do not affect the measured packing cost.
    del device
    inp = torch.randint(0, 16, shape, dtype=dtype, device="cpu")
    # Flat positional arguments, no trailing kwargs dict: op(inp, inner_k_tiles),
    # the same call form the correctness tests use.
    return inp, inner_k_tiles


class Int4PackCpuBenchmark(base.GenericBenchmark):
    """Two-phase case benchmark restricted to the native weight contract.

    set_shapes keeps the shared loader (a caller --shape_file still wins) and
    then drops shapes this fixed-rank CPU operator cannot accept: the generic
    shape sets are dominated by 1-D/3-D entries that the native validator
    rejects with "expect weight to be 2D tensor.". set_more_shapes adds the
    native-valid int4 weight geometries.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                tuple(shape)
                for shape in list(self.shapes) + _EXTRA_SHAPES
                if len(shape) == 2 and shape[0] % 16 == 0 and shape[1] % 2 == 0
            )
        )

    def set_more_shapes(self):
        return list(super().set_more_shapes()) + _EXTRA_SHAPES


@pytest.mark.convert_weight_to_int4pack_for_cpu
def test__convert_weight_to_int4pack_for_cpu():
    # torch_op is the perf comparison reference and gems_op the candidate; both
    # are called with the same (CPU int32 tensor, int) signature. The candidate
    # name starts with an underscore, so it is fetched with getattr to stay
    # importable before a kernel is registered, while executing the benchmark
    # still fails when no candidate is available.
    bench = Int4PackCpuBenchmark(
        op_name="_convert_weight_to_int4pack_for_cpu",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._convert_weight_to_int4pack_for_cpu,
        gems_op=getattr(flag_gems, "_convert_weight_to_int4pack_for_cpu", None),
        dtypes=[torch.int32],
    )
    bench.run()
