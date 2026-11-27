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

"""NVIDIA native baseline for `thnn_fused_gru_cell_backward_out`.

Mirrors benchmark/test_thnn_fused_gru_cell_backward.py
(FusedGRUCellBackwardOutBenchmark, a base.Benchmark subclass):
    op_name  = "thnn_fused_gru_cell_backward_out"
    torch_op = _fused_gru_cell_backward_out, which calls
               torch.ops.aten._thnn_fused_gru_cell_backward.out
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with the (batch, hidden) pairs of CORE_SHAPES
and only adds COMPREHENSIVE_SHAPES at --level comprehensive, so
core_shapes.yaml is not consulted.

The workspace comes from a real torch.ops.aten._thnn_fused_gru_cell forward
(its second output), which build_inputs performs outside the timed region as
`_make_workspace` does. has_bias is True and the five out buffers are
[batch, 3H], [batch, 3H], [batch, H], [3H], [3H], all passed as keywords.

This calls the ATen backward kernel directly; it is not an autograd benchmark.

float64 is in the dtype list because the benchmark appends it when
flag_gems.runtime.device.support_fp64 is true, and the NVIDIA backend
reports fp64_enabled=True.
"""

import torch

OP_NAME = "thnn_fused_gru_cell_backward_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.float64]
IS_INPLACE = False
MUTATED_INPUTS = (3, 4, 5, 6, 7)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._thnn_fused_gru_cell_backward.out is unavailable on this build"
)

# test_thnn_fused_gru_cell_backward.py::CORE_SHAPES, read as (batch, hidden)
_SHAPES = [
    (1, 32),
    (8, 64),
    (32, 128),
    (64, 256),
    (128, 512),
    (256, 1024),
]

_HAS_BIAS = True


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent.

    torch.ops.aten.<name>.<overload> raises for an operator this build does not
    carry, so the lookup is guarded.
    """
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def _make_workspace(batch_size, hidden_size, dtype, device):
    """benchmark::_make_workspace -- the forward's second output."""
    input_gates = torch.randn(batch_size, 3 * hidden_size, dtype=dtype, device=device)
    hidden_gates = torch.randn_like(input_gates)
    hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    input_bias = torch.randn(3 * hidden_size, dtype=dtype, device=device)
    hidden_bias = torch.randn_like(input_bias)
    return torch.ops.aten._thnn_fused_gru_cell(
        input_gates, hidden_gates, hx, input_bias, hidden_bias
    )[1]


def _fused_gru_cell_backward_out(
    grad_hy, workspace, has_bias, out0, out1, out2, out3, out4
):
    """benchmark::_fused_gru_cell_backward_out -- outs forwarded as keywords."""
    return torch.ops.aten._thnn_fused_gru_cell_backward.out(
        grad_hy,
        workspace,
        has_bias,
        out0=out0,
        out1=out1,
        out2=out2,
        out3=out3,
        out4=out4,
    )


def native():
    if _resolve("torch.ops.aten._thnn_fused_gru_cell_backward.out") is None:
        return None
    return _fused_gru_cell_backward_out


def grid():
    return [{"batch": batch, "hidden": hidden} for batch, hidden in _SHAPES]


def build_inputs(binding, dtype, device):
    batch_size, hidden_size = binding["batch"], binding["hidden"]
    workspace = _make_workspace(batch_size, hidden_size, dtype, device)
    grad_hy = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    out0 = torch.empty(batch_size, 3 * hidden_size, dtype=dtype, device=device)
    out1 = torch.empty_like(out0)
    out2 = torch.empty_like(grad_hy)
    out3 = torch.empty(3 * hidden_size, dtype=dtype, device=device)
    out4 = torch.empty_like(out3)
    return (
        grad_hy,
        workspace,
        _HAS_BIAS,
        out0,
        out1,
        out2,
        out3,
        out4,
    ), {}


def key_shape(binding):
    return [binding["batch"], binding["hidden"]]


def config(binding, dtype):
    batch, hidden = binding["batch"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [batch, hidden], "dtype": dt, "note": "grad_hy"},
                {
                    "shape": [batch, 5 * hidden],
                    "dtype": dt,
                    "note": "workspace from the forward",
                },
                {"scalar": _HAS_BIAS},
                {"shape": [batch, 3 * hidden], "dtype": dt, "note": "out0"},
                {"shape": [batch, 3 * hidden], "dtype": dt, "note": "out1"},
                {"shape": [batch, hidden], "dtype": dt, "note": "out2"},
                {"shape": [3 * hidden], "dtype": dt, "note": "out3"},
                {"shape": [3 * hidden], "dtype": dt, "note": "out4"},
            ],
            "kwargs": {},
        },
        "outputs": {
            "grad_input_gates": {"shape": [batch, 3 * hidden], "dtype": dt},
            "grad_hidden_gates": {"shape": [batch, 3 * hidden], "dtype": dt},
            "grad_hx": {"shape": [batch, hidden], "dtype": dt},
            "grad_input_bias": {"shape": [3 * hidden], "dtype": dt},
            "grad_hidden_bias": {"shape": [3 * hidden], "dtype": dt},
        },
        "dims": {"batch": batch, "hidden": hidden},
    }
