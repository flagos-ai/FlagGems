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

"""Benchmark for aten::_scaled_dot_product_flash_attention_for_cpu.

The kernel is CPU-only, so the input builders allocate CPU tensors; the shared
OperatorBenchmark shape loader still honours a caller-supplied shape file.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# The original square workloads keep the causal/non-causal pair. The extra rows
# add the rectangular cross-attention form, a 2-D mask and a non-default scale.
# A row is (query_shape, key/value_shape, is_causal, scale, mask_shape).
_BASELINE_SHAPES = [
    (1, 1, 1024, 64),
    (2, 4, 320, 15),
    (16, 8, 128, 60),
    (16, 128, 64, 60),
]

ATTENTION_ROWS = (
    [(shape, shape, False, None, None) for shape in _BASELINE_SHAPES]
    + [(shape, shape, True, None, None) for shape in _BASELINE_SHAPES]
    + [
        ((2, 4, 32, 60), (2, 4, 512, 60), False, None, None),
        ((2, 4, 32, 60), (2, 4, 512, 60), True, None, None),
        ((2, 4, 512, 60), (2, 4, 32, 60), False, None, (512, 32)),
        ((1, 1, 1024, 64), (1, 1, 1024, 64), False, 0.5, None),
    ]
)


def _rows_for(shape):
    """Annotated row, or the original causal pair for a plain 4-D shape."""
    if isinstance(shape[0], (tuple, list)):
        return [shape]
    return [(shape, shape, False, None, None), (shape, shape, True, None, None)]


def _case_fn(shape, dtype):
    # The dtype travels with the plan, so it is not part of the metadata.
    del dtype
    for q_shape, kv_shape, is_causal, scale, mask_shape in _rows_for(shape):
        yield base.BenchmarkCasePlan(
            shape={
                "query": list(q_shape),
                "key": list(kv_shape),
                "value": list(kv_shape),
            },
            params={
                "is_causal": is_causal,
                "scale": scale,
                "mask_shape": list(mask_shape) if mask_shape else None,
            },
            builder_args=(q_shape, kv_shape, is_causal, scale, mask_shape),
        )


def _build_inputs_fn(plan, dtype, device):
    # The kernel is CPU-only, so the builders ignore the benchmark device.
    del device
    q_shape, kv_shape, is_causal, scale, mask_shape = plan.builder_args
    cpu = torch.device("cpu")
    q = torch.randn(q_shape, dtype=dtype, device=cpu)
    k = torch.randn(kv_shape, dtype=dtype, device=cpu)
    v = torch.randn(kv_shape, dtype=dtype, device=cpu)
    kwargs = {}
    if mask_shape is not None:
        kwargs["attn_mask"] = torch.randn(tuple(mask_shape), dtype=dtype, device=cpu)
    if scale is not None:
        kwargs["scale"] = scale
    # unpack_to_args_kwargs takes flat positional arguments plus a trailing
    # kwargs dict; a nested (args, kwargs) tuple is not unpacked.
    return q, k, v, 0.0, is_causal, kwargs


class ScaledDotProductFlashAttentionForCpuBenchmark(OperatorBenchmark):
    # The shared loader resolves this operator (then the class name) from the
    # requested shape file and falls back to these rows when it has no entry.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # This kernel requires rank-4 query/key/value tensors. Preserve native-
        # valid shared geometries and union the original attention descriptors.
        loaded = [tuple(shape) for shape in self.shapes if len(shape) == 4]
        self.shapes = loaded + ATTENTION_ROWS


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
def test__scaled_dot_product_flash_attention_for_cpu():
    bench = ScaledDotProductFlashAttentionForCpuBenchmark(
        op_name="_scaled_dot_product_flash_attention_for_cpu",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
        gems_op=getattr(flag_gems, "_scaled_dot_product_flash_attention_for_cpu", None),
        dtypes=consts.FLOAT_DTYPES + [torch.float64],
    )
    bench.run()
