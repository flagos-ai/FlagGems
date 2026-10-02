# Copyright 2026, The FlagOS Contributors.
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

"""Benchmark for ``aten::_scaled_dot_product_flash_attention_for_cpu_backward``.

The native kernel is CPU-only and its signature is (grad_out, query, key,
value, out, logsumexp, dropout_p, is_causal, attn_mask=None, scale=None), so
``torch_op`` and the injected ``gems_op`` receive the same CPU tensors.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# Four-dimensional (batch, head, seq, head_dim) attention workloads.
_SHAPES = [
    (1, 2, 64, 32),
    (2, 4, 128, 64),
    (1, 8, 256, 64),
    (2, 8, 512, 64),
    (1, 16, 1024, 64),
    (4, 4, 128, 128),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={
            "grad_out": list(shape),
            "query": list(shape),
            "key": list(shape),
            "value": list(shape),
        },
        params={"dropout_p": 0.0, "is_causal": False},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    del device
    shape = plan.builder_args[0]
    cpu = torch.device("cpu")
    query = torch.randn(shape, dtype=dtype, device=cpu)
    key = torch.randn(shape, dtype=dtype, device=cpu)
    value = torch.randn(shape, dtype=dtype, device=cpu)
    out, logsumexp = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        query, key, value
    )
    grad_out = torch.randn(shape, dtype=dtype, device=cpu)
    # Flat positional arguments plus the trailing kwargs dict, which is what
    # unpack_to_args_kwargs expects.
    return (
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        plan.params["dropout_p"],
        plan.params["is_causal"],
        {},
    )


class ScaledDotProductFlashAttentionForCpuBackwardBenchmark(OperatorBenchmark):
    """Two-phase benchmark over 4-D attention shapes; honours --shape-file."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # This native interface requires rank 4. Keep all shared inputs of
        # that rank and every original operator workload; unsupported ranks are
        # tested as errors in the correctness suite, not reshaped into new semantics.
        self.shapes = list(
            dict.fromkeys(
                [tuple(shape) for shape in self.shapes if len(shape) == 4] + _SHAPES
            )
        )


@pytest.mark.scaled_dot_product_flash_attention_for_cpu_backward
def test_scaled_dot_product_flash_attention_for_cpu_backward():
    bench = ScaledDotProductFlashAttentionForCpuBackwardBenchmark(
        op_name="_scaled_dot_product_flash_attention_for_cpu_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward,
        gems_op=getattr(
            flag_gems, "_scaled_dot_product_flash_attention_for_cpu_backward", None
        ),
        dtypes=consts.FLOAT_DTYPES + [torch.float64],
    )
    bench.run()
