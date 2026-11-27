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

"""NVIDIA native baseline for `batch_norm_no_update`.

Mirrors benchmark/test_batch_norm_no_update.py (NormBenchmark, a
GenericBenchmark subclass):
    op_name  = "batch_norm_no_update"
    torch_op = torch_batch_norm_no_update  (the module-local reference below)
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:NormBenchmark (benchmark/core_shapes.yaml)

The benchmark's native reference is not the ATen `_batch_norm_no_update`
operator but a torch.nn.functional.batch_norm call with training=False, which
is the inference-mode equivalent of "no update". That exact composition is
reproduced here so the baseline measures what the benchmark measures.
"""

import torch

OP_NAME = "batch_norm_no_update"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.nn.functional.batch_norm is unavailable on this torch build"
)

# yaml:NormBenchmark -- shape_desc "N, C, *"
_SHAPES = [
    (4, 16, 64, 4),
    (16, 16, 8, 48),
    (16, 16, 8, 88),
    (16, 16, 128),
    (20, 6, 65536),
]

_MOMENTUM = 0.1
_EPS = 1e-5


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


def _torch_batch_norm_no_update(
    inp, weight, bias, running_mean, running_var, momentum, eps
):
    """benchmark/test_batch_norm_no_update.py::torch_batch_norm_no_update.

        return F.batch_norm(
            inp, running_mean, running_var, weight, bias, training=False, eps=eps
        )

    `momentum` is accepted to keep the signature aligned with the benchmark's
    input_fn but is unused, exactly as in the benchmark reference.
    """
    return torch.nn.functional.batch_norm(
        inp, running_mean, running_var, weight, bias, training=False, eps=eps
    )


def native():
    if _resolve("torch.nn.functional.batch_norm") is None:
        return None
    return _torch_batch_norm_no_update


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    c = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((c,), dtype=dtype, device=device)
    bias = torch.randn((c,), dtype=dtype, device=device)
    running_mean = torch.randn((c,), dtype=dtype, device=device)
    running_var = torch.abs(torch.randn((c,), dtype=dtype, device=device)) + 0.1
    return (inp, weight, bias, running_mean, running_var, _MOMENTUM, _EPS), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    c = shape[1]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": [c], "dtype": dt},
                {"shape": [c], "dtype": dt},
                {"shape": [c], "dtype": dt},
                {"shape": [c], "dtype": dt},
                {"scalar": _MOMENTUM},
                {"scalar": _EPS},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"N": shape[0], "C": c, "shape": shape},
    }
