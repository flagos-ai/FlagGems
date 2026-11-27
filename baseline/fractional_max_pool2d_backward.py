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

"""NVIDIA native baseline for `fractional_max_pool2d_backward`.

Mirrors benchmark/test_fractional_max_pool2d.py (FractionalMaxPool2dBenchmark,
a GenericBenchmark subclass):
    op_name  = "fractional_max_pool2d_backward"
    torch_op = torch_fractional_max_pool2d_backward_wrapper, which calls
               torch.ops.aten.fractional_max_pool2d_backward
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:FractionalMaxPool2dBenchmark (benchmark/core_shapes.yaml),
               shape_desc "N, C, H, W"

output_size is (H // 2, W // 2) and kernel_size is 2.

DEVIATION: the benchmark's input_fn produces `indices` by calling
flag_gems.fractional_max_pool2d, i.e. the candidate kernel. A baseline module
must not import flag_gems (it runs in a plain torch/vLLM image), so the
indices here come from the ATen forward
torch.ops.aten.fractional_max_pool2d with the same kernel_size/output_size.
Both produce a valid argmax index tensor of the same shape and dtype, which is
all the backward kernel consumes. See Q&A.md.

The forward runs in build_inputs, outside the timed region. This calls the
ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "fractional_max_pool2d_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.fractional_max_pool2d_backward is unavailable on this build"
)

# yaml:FractionalMaxPool2dBenchmark -- shape_desc "N, C, H, W"
_SHAPES = [
    (4, 3, 224, 224),
    (16, 64, 56, 56),
    (32, 128, 28, 28),
    (64, 256, 14, 14),
    (128, 512, 7, 7),
]

_KERNEL_SIZE = 2
_SAMPLES_SEED = 0

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_INT_DTYPES = (torch.int16, torch.int32)
_BOOL_DTYPES = (torch.bool,)
_COMPLEX_DTYPES = (torch.complex64,)


def _generate_tensor_input(shape, dtype, device):
    """benchmark/utils.py::generate_tensor_input, verbatim semantics."""
    if dtype in _FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype in _INT_DTYPES:
        return torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cpu",
        ).to(device)
    if dtype in _BOOL_DTYPES:
        return torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(device)
    if dtype in _COMPLEX_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    raise TypeError(f"unsupported dtype: {dtype}")


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


def _torch_fractional_max_pool2d_backward(grad_output, inp, **kwargs):
    """benchmark::torch_fractional_max_pool2d_backward_wrapper."""
    return torch.ops.aten.fractional_max_pool2d_backward(
        grad_output,
        inp,
        kwargs["kernel_size"],
        kwargs["output_size"],
        kwargs["indices"],
    )


def native():
    if _resolve("torch.ops.aten.fractional_max_pool2d_backward") is None:
        return None
    return _torch_fractional_max_pool2d_backward


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = _generate_tensor_input(shape, dtype, device)
    inp.requires_grad_(True)
    output_size = (shape[2] // 2, shape[3] // 2)
    # The ATen forward stands in for flag_gems.fractional_max_pool2d; it needs
    # the random sample tensor the functional wrapper would otherwise draw.
    generator = torch.Generator(device="cpu").manual_seed(_SAMPLES_SEED)
    random_samples = torch.rand(
        shape[0], shape[1], 2, dtype=torch.float32, generator=generator
    ).to(device=device, dtype=dtype)
    _, indices = torch.ops.aten.fractional_max_pool2d(
        inp, [_KERNEL_SIZE, _KERNEL_SIZE], list(output_size), random_samples
    )
    grad_output = torch.randn_like(
        torch.empty((shape[0], shape[1], *output_size), dtype=dtype, device=device)
    )
    return (grad_output, inp), {
        "kernel_size": _KERNEL_SIZE,
        "output_size": output_size,
        "indices": indices,
    }


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    output_size = [shape[2] // 2, shape[3] // 2]
    out_shape = [shape[0], shape[1]] + output_size
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": out_shape, "dtype": dt, "note": "grad_output"},
                {"shape": shape, "dtype": dt, "requires_grad": True},
            ],
            "kwargs": {
                "kernel_size": {"scalar": _KERNEL_SIZE},
                "output_size": output_size,
                "indices": {"shape": out_shape, "dtype": "torch.int64"},
            },
        },
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {
            "N": shape[0],
            "C": shape[1],
            "H": shape[2],
            "W": shape[3],
            "kernel_size": _KERNEL_SIZE,
            "output_size": output_size,
        },
    }
