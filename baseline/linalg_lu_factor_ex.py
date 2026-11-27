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

"""NVIDIA native baseline for `linalg_lu_factor_ex`.

Mirrors benchmark/test_linalg_lu_factor_ex.py (LinalgLuFactorExBenchmark, a
base.Benchmark subclass):
    op_name  = "linalg_lu_factor_ex"
    torch_op = _torch_lu_factor_ex
    dtypes   = _TEST_DTYPES = [float32, float64] when the backend has FP64

`_torch_lu_factor_ex` is defined twice in that file: the Python LU composite
only when VENDOR == "ascend", and otherwise a thin wrapper over
torch.linalg.lu_factor_ex. This baseline takes the non-Ascend definition.

The class overrides set_shapes with LU_FACTOR_EX_SHAPES, so core_shapes.yaml
is not consulted. get_input_iter crosses each shape with pivot in
_PIVOT_VALUES -- [True, False] on CUDA, since pivot=False is CUDA-only -- and
with check_errors in [False, True] for pivot=True and [False] for pivot=False,
which is expanded into the grid below.
"""

import torch

OP_NAME = "linalg_lu_factor_ex"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.linalg.lu_factor_ex is unavailable on this torch build"
)

# test_linalg_lu_factor_ex.py::LU_FACTOR_EX_SHAPES
_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (1024, 512),
    (32, 16),
    (16, 32),
    (128, 64),
    (64, 128),
    (4, 32, 32),
    (128, 16, 16),
    (1024, 512, 512),
]

# _PIVOT_VALUES on CUDA; pivot=False is only supported there.
_PIVOT_VALUES = (True, False)
_CHECK_ERRORS_VALUES = (False, True)


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


def _torch_lu_factor_ex(inp, *, pivot=True, check_errors=False, out=None):
    """benchmark::_torch_lu_factor_ex, the non-Ascend definition."""
    if out is not None:
        return torch.linalg.lu_factor_ex(
            inp, pivot=pivot, check_errors=check_errors, out=out
        )
    return torch.linalg.lu_factor_ex(inp, pivot=pivot, check_errors=check_errors)


def native():
    if _resolve("torch.linalg.lu_factor_ex") is None:
        return None
    return _torch_lu_factor_ex


def grid():
    bindings = []
    for shape in _SHAPES:
        for pivot in _PIVOT_VALUES:
            # check_errors only varies for pivot=True, to bound the grid size.
            check_errors_list = _CHECK_ERRORS_VALUES if pivot else (False,)
            for check_errors in check_errors_list:
                bindings.append(
                    {
                        "shape": list(shape),
                        "pivot": pivot,
                        "check_errors": check_errors,
                    }
                )
    return bindings


def build_inputs(binding, dtype, device):
    inp = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (inp,), {
        "pivot": binding["pivot"],
        "check_errors": binding["check_errors"],
    }


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return (
        f"{shape}_pivot{int(binding['pivot'])}" f"_check{int(binding['check_errors'])}"
    )


def config(binding, dtype):
    shape = list(binding["shape"])
    m, n = shape[-2], shape[-1]
    batch = shape[:-2]
    k = min(m, n)
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {
                "pivot": {"scalar": binding["pivot"]},
                "check_errors": {"scalar": binding["check_errors"]},
            },
        },
        "outputs": {
            "LU": {"shape": shape, "dtype": dt},
            "pivots": {"shape": batch + [k], "dtype": "torch.int32"},
            "info": {"shape": batch, "dtype": "torch.int32"},
        },
        "dims": {
            "shape": shape,
            "M": m,
            "N": n,
            "pivot": binding["pivot"],
            "check_errors": binding["check_errors"],
        },
    }
