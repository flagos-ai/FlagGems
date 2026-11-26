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

"""NVIDIA native baseline for `miopen_batch_norm_backward`.

Mirrors benchmark/test_miopen_batch_norm_backward.py (NormBenchmark, a
GenericBenchmark subclass):
    op_name  = "miopen_batch_norm_backward"
    torch_op = miopen_batch_norm_backward_reference (defined inside the test)
    dtypes   = consts.FLOAT_DTYPES on NVIDIA
    shapes   = yaml:NormBenchmark (benchmark/core_shapes.yaml),
               shape_desc "N, C, *"

PyTorch exposes no `miopen_batch_norm_backward`, so the benchmark's reference
calls torch.ops.aten.native_batch_norm_backward instead, with save_var read as
the saved inverse standard deviation and output_mask [True, weight is not None,
True]. That composition is reproduced here.

At --level core `batchnorm_input_fn` yields one forward argument set with
running_mean/running_var None, which the wrapper then replaces with zeros/ones;
the variant with real running stats is COMPREHENSIVE-only. save_mean and
save_var are float32 regardless of the input dtype. train is True and eps is
1e-5.
"""

import torch

OP_NAME = "miopen_batch_norm_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.native_batch_norm_backward is unavailable on this build"
)

# yaml:NormBenchmark -- shape_desc "N, C, *"
_SHAPES = [
    (4, 16, 64, 4),
    (16, 16, 8, 48),
    (16, 16, 8, 88),
    (16, 16, 128),
    (20, 6, 65536),
]

_EPS = 1e-5
_TRAIN = True


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


def _miopen_batch_norm_backward_reference(
    inp,
    grad_output,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_var,
    epsilon,
):
    """benchmark::miopen_batch_norm_backward_reference.

    MIOpen names the argument save_var but stores the saved inverse std, so it
    is passed through as save_invstd.
    """
    save_invstd = save_var
    output_mask = [True, weight is not None, True]
    return torch.ops.aten.native_batch_norm_backward(
        grad_output,
        inp,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        _TRAIN,
        epsilon,
        output_mask,
    )


def native():
    if _resolve("torch.ops.aten.native_batch_norm_backward") is None:
        return None
    return _miopen_batch_norm_backward_reference


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    channels = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((channels,), dtype=dtype, device=device)
    grad_output = torch.randn_like(inp)
    # batchnorm_input_fn passes None at --level core; the wrapper substitutes
    # zeros/ones.
    running_mean = torch.zeros(channels, dtype=dtype, device=device)
    running_var = torch.ones(channels, dtype=dtype, device=device)
    save_mean = torch.randn(channels, dtype=torch.float32, device=device)
    save_var = (
        torch.abs(torch.randn(channels, dtype=torch.float32, device=device)) + 0.1
    )
    return (
        inp,
        grad_output,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        _EPS,
    ), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    c = shape[1]
    dt = str(dtype)
    f32 = "torch.float32"
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": shape, "dtype": dt, "note": "grad_output"},
                {"shape": [c], "dtype": dt},
                {"shape": [c], "dtype": dt},
                {"shape": [c], "dtype": dt},
                {"shape": [c], "dtype": f32},
                {"shape": [c], "dtype": f32, "note": "saved inverse std"},
                {"scalar": _EPS},
            ],
            "kwargs": {},
        },
        "outputs": {
            "grad_input": {"shape": shape, "dtype": dt},
            "grad_weight": {"shape": [c], "dtype": dt},
            "grad_bias": {"shape": [c], "dtype": dt},
        },
        "dims": {"N": shape[0], "C": c, "shape": shape},
    }
