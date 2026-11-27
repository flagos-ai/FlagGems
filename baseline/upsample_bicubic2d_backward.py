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

"""NVIDIA native baseline for `upsample_bicubic2d_backward`.

Mirrors benchmark/test_upsample_bicubic2d_backward.py
(UpsampleBicubic2dBackwardBenchmark, a base.Benchmark subclass):
    op_name  = "upsample_bicubic2d_backward"
    torch_op = torch.ops.aten.upsample_bicubic2d_backward.default
    dtypes   = [float32, float64, float16, bfloat16] (parametrized; every skip
               in that list is mthreads/ascend/iluvatar-only, so all four run
               on NVIDIA)

The class overrides set_shapes with the named CASES list -- each entry is
(name, input shape, output_size, scales_h, scales_w, layout) -- so
core_shapes.yaml is not consulted. get_input_iter crosses every case with
align_corners in (False, True), so each case yields two collection points.

The "noncontiguous" case allocates grad_output at twice the width and takes
[..., ::2], giving a strided input; that is the point of the case.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "upsample_bicubic2d_backward"
DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.upsample_bicubic2d_backward.default is unavailable"
)

# test_upsample_bicubic2d_backward.py::CASES
# (name, input shape, output_size, scales_h, scales_w, layout)
_CASES = [
    ("identity_small", (1, 3, 8, 9), (8, 9), None, None, "contiguous"),
    ("identity_large", (8, 16, 128, 128), (128, 128), None, None, "contiguous"),
    ("double_small", (1, 3, 8, 9), (16, 18), None, None, "contiguous"),
    ("double_large", (4, 16, 64, 64), (128, 128), None, None, "contiguous"),
    ("noninteger", (2, 8, 37, 53), (61, 89), None, None, "contiguous"),
    ("downsample", (2, 8, 64, 80), (23, 31), None, None, "contiguous"),
    ("strong_downsample", (2, 8, 128, 160), (3, 5), None, None, "contiguous"),
    ("out_one", (2, 3, 13, 17), (1, 1), None, None, "contiguous"),
    ("input_one", (1, 3, 1, 1), (31, 37), None, None, "contiguous"),
    ("explicit_scales", (2, 3, 13, 17), (23, 31), 1.7, 1.9, "contiguous"),
    ("small_scale", (1, 3, 13, 17), (23, 31), 0.6, 0.8, "contiguous"),
    ("noncontiguous", (2, 3, 13, 17), (23, 31), None, None, "strided"),
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
    op = _resolve("torch.ops.aten.upsample_bicubic2d_backward.default")
    return op if callable(op) else None


def grid():
    return [
        {
            "name": name,
            "shape": list(shape),
            "output_size": list(output_size),
            "scales_h": sh,
            "scales_w": sw,
            "layout": layout,
            "align_corners": align_corners,
        }
        for name, shape, output_size, sh, sw, layout in _CASES
        for align_corners in (False, True)
    ]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    oh, ow = binding["output_size"]
    n, c = shape[:2]
    if binding["layout"] == "strided":
        grad_output = torch.randn((n, c, oh, ow * 2), device=device, dtype=dtype)[
            ..., ::2
        ]
    else:
        grad_output = torch.randn((n, c, oh, ow), device=device, dtype=dtype)
    return (
        grad_output,
        tuple(binding["output_size"]),
        shape,
        binding["align_corners"],
        binding["scales_h"],
        binding["scales_w"],
    ), {}


def key_shape(binding):
    return f"{binding['name']}_align{int(binding['align_corners'])}"


def config(binding, dtype):
    shape = list(binding["shape"])
    oh, ow = binding["output_size"]
    n, c = shape[:2]
    dt = str(dtype)
    grad = {"shape": [n, c, oh, ow], "dtype": dt}
    if binding["layout"] == "strided":
        grad["stride"] = [c * oh * ow * 2, oh * ow * 2, ow * 2, 2]
    return {
        "inputs": {
            "args": [
                grad,
                list(binding["output_size"]),
                shape,
                {"scalar": binding["align_corners"]},
                binding["scales_h"],
                binding["scales_w"],
            ],
            "kwargs": {},
        },
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {
            "case": binding["name"],
            "input_shape": shape,
            "output_size": list(binding["output_size"]),
            "align_corners": binding["align_corners"],
            "scales_h": binding["scales_h"],
            "scales_w": binding["scales_w"],
            "layout": binding["layout"],
        },
    }
