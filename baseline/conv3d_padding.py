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

"""NVIDIA native baseline for `conv3d_padding`.

Mirrors benchmark/test_conv3d.py (Conv3DPaddingBenchmark, a GenericBenchmark
subclass):
    op_name  = "conv3d_padding"
    torch_op = torch.nn.functional.conv3d
    dtypes   = consts.FLOAT_DTYPES

The class carries CONV3D_PADDING_SHAPES, overrides get_input_iter to walk it
and returns [] from set_more_shapes, so core_shapes.yaml is not consulted.
Each entry is (batch, in_c, in_d, in_h, in_w, out_c, k_d, k_h, k_w, stride,
padding, groups) and padding is the "valid"/"same" string mode, which is the
point of this benchmark.

The benchmark sets torch.backends.cudnn.allow_tf32 = False before running; the
same flag is set here. All arguments are passed as one keyword dict with
bias=None, as the input_fn yields them.
"""

import torch

OP_NAME = "conv3d_padding"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.nn.functional.conv3d is unavailable on this torch build"
)

# Conv3DPaddingBenchmark.CONV3D_PADDING_SHAPES
# (batch, in_c, in_d, in_h, in_w, out_c, k_d, k_h, k_w, stride, padding, groups)
_SHAPES = [
    (2, 3, 9, 9, 9, 3, 3, 3, 3, 1, "valid", 1),
    (9, 16, 4, 4, 4, 128, 2, 2, 2, 1, "valid", 4),
    (32, 8, 8, 8, 8, 32, 3, 3, 3, 1, "valid", 1),
    (2, 3, 9, 9, 9, 3, 3, 3, 3, 1, "same", 1),
    (9, 16, 4, 4, 4, 128, 2, 2, 2, 1, "same", 4),
    (32, 8, 8, 8, 8, 32, 3, 3, 3, 1, "same", 1),
]

_FIELDS = (
    "batch",
    "input_c",
    "input_d",
    "input_h",
    "input_w",
    "out_c",
    "kernel_d",
    "kernel_h",
    "kernel_w",
    "stride",
    "padding",
    "groups",
)


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
    op = _resolve("torch.nn.functional.conv3d")
    if not callable(op):
        return None
    # The benchmark pins this before running.
    torch.backends.cudnn.allow_tf32 = False
    return op


def grid():
    return [dict(zip(_FIELDS, shape)) for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    input_shape = (
        binding["batch"],
        binding["input_c"],
        binding["input_d"],
        binding["input_h"],
        binding["input_w"],
    )
    weight_shape = (
        binding["out_c"],
        binding["input_c"] // binding["groups"],
        binding["kernel_d"],
        binding["kernel_h"],
        binding["kernel_w"],
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
        f"b{binding['batch']}_ic{binding['input_c']}"
        f"_d{binding['input_d']}h{binding['input_h']}w{binding['input_w']}"
        f"_oc{binding['out_c']}"
        f"_k{binding['kernel_d']}x{binding['kernel_h']}x{binding['kernel_w']}"
        f"_s{binding['stride']}_p{binding['padding']}_g{binding['groups']}"
    )


def config(binding, dtype):
    dt = str(dtype)
    input_shape = [
        binding["batch"],
        binding["input_c"],
        binding["input_d"],
        binding["input_h"],
        binding["input_w"],
    ]
    weight_shape = [
        binding["out_c"],
        binding["input_c"] // binding["groups"],
        binding["kernel_d"],
        binding["kernel_h"],
        binding["kernel_w"],
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
        "dims": {key: binding[key] for key in _FIELDS},
    }
