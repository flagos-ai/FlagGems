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

"""NVIDIA native baseline for `conv1d_padding`.

Mirrors benchmark/test_conv1d.py (Conv1DPaddingBenchmark, a GenericBenchmark
subclass):
    op_name  = "conv1d_padding"
    torch_op = torch.nn.functional.conv1d
    dtypes   = [torch.float16, torch.float32]

The class overrides get_input_iter with its own thirteen-entry shape list, so
core_shapes.yaml is not consulted. Each entry is
(batch, in_c, in_l, out_c, kernel, stride, padding, groups) and the padding
field covers integers as well as the "valid"/"same" string modes, which is the
point of this benchmark.

bfloat16 is excluded because of cuDNN limitations, and the benchmark sets
torch.backends.cudnn.allow_tf32 = False before running; the same flag is set
here so the baseline measures the same cuDNN configuration.

All arguments are passed as a single keyword dict, exactly as the input_fn
yields them (bias=None).
"""

import torch

OP_NAME = "conv1d_padding"
DTYPES = [torch.float16, torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.nn.functional.conv1d is unavailable on this torch build"
)

# Conv1DPaddingBenchmark.get_input_iter::shapes
# (batch, in_c, in_l, out_c, kernel, stride, padding, groups)
_SHAPES = [
    (32, 64, 512, 64, 3, 1, 0, 1),
    (32, 64, 512, 64, 3, 1, 1, 1),
    (32, 64, 512, 64, 3, 1, 2, 1),
    (64, 48, 1024, 128, 5, 2, 0, 1),
    (64, 48, 1024, 128, 5, 2, 2, 1),
    (64, 48, 1024, 128, 5, 2, 4, 1),
    (16, 24, 2048, 96, 7, 1, 0, 2),
    (16, 24, 2048, 96, 7, 1, 3, 2),
    (16, 24, 2048, 96, 7, 1, 6, 2),
    (32, 64, 512, 64, 3, 1, "valid", 1),
    (64, 48, 1024, 128, 5, 2, "valid", 1),
    (16, 24, 2048, 96, 7, 1, "same", 2),
    (8, 8, 8192, 16, 11, 1, "same", 1),
]


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


def native():
    op = _resolve("torch.nn.functional.conv1d")
    if not callable(op):
        return None
    # The benchmark pins this before running; cuDNN TF32 would otherwise change
    # which kernels the float32 rows measure.
    torch.backends.cudnn.allow_tf32 = False
    return op


def grid():
    return [
        {
            "batch": batch,
            "input_c": input_c,
            "input_l": input_l,
            "out_c": out_c,
            "kernel": kernel,
            "stride": stride,
            "padding": padding,
            "groups": groups,
        }
        for batch, input_c, input_l, out_c, kernel, stride, padding, groups in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    input_shape = (binding["batch"], binding["input_c"], binding["input_l"])
    weight_shape = (
        binding["out_c"],
        binding["input_c"] // binding["groups"],
        binding["kernel"],
    )
    inp = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)
    return (), {
        "input": inp,
        "weight": weight,
        "bias": None,
        "groups": binding["groups"],
        "stride": binding["stride"],
        "padding": binding["padding"],
    }


def key_shape(binding):
    return (
        f"b{binding['batch']}_ic{binding['input_c']}_l{binding['input_l']}"
        f"_oc{binding['out_c']}_k{binding['kernel']}_s{binding['stride']}"
        f"_p{binding['padding']}_g{binding['groups']}"
    )


def config(binding, dtype):
    dt = str(dtype)
    input_shape = [binding["batch"], binding["input_c"], binding["input_l"]]
    weight_shape = [
        binding["out_c"],
        binding["input_c"] // binding["groups"],
        binding["kernel"],
    ]
    return {
        "inputs": {
            "args": [],
            "kwargs": {
                "input": {"shape": input_shape, "dtype": dt},
                "weight": {"shape": weight_shape, "dtype": dt},
                "bias": None,
                "groups": {"scalar": binding["groups"]},
                "stride": {"scalar": binding["stride"]},
                "padding": {"scalar": binding["padding"]},
            },
        },
        "outputs": {"out": {"dtype": dt}},
        "dims": {
            "N": binding["batch"],
            "C_in": binding["input_c"],
            "L_in": binding["input_l"],
            "C_out": binding["out_c"],
            "kernel": binding["kernel"],
            "stride": binding["stride"],
            "padding": binding["padding"],
            "groups": binding["groups"],
        },
    }
