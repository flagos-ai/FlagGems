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

"""NVIDIA native baseline for `cast_Double`.

Mirrors benchmark/test_cast_Double.py (CastDoubleBenchmark, a base.Benchmark
subclass):
    op_name  = "cast_Double"
    torch_op = torch.ops.aten._cast_Double.default
    dtypes   = [torch.float16, torch.bfloat16, torch.float32]

The class overrides get_input_iter and ignores self.shapes entirely, so
core_shapes.yaml is not consulted. It emits four contiguous 1-D sizes and then
two non-contiguous views of one 1024x1024 tensor: the transpose and an
every-other-column slice. Those two layouts are the point of the benchmark --
a cast kernel that only handles contiguous input would regress there -- so the
grid carries them as explicit named cases rather than as plain shapes.

_cast_Double casts to torch.float64.
"""

import torch

OP_NAME = "cast_Double"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._cast_Double.default is unavailable on this torch build"
)

# CastDoubleBenchmark.get_input_iter: four contiguous 1-D sizes ...
_SHAPES = [
    (1024,),
    (65536,),
    (1048576,),
    (16777216,),
]
# ... then two non-contiguous views of a 1024x1024 tensor.
_VIEW_BASE = (1024, 1024)


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
    op = _resolve("torch.ops.aten._cast_Double.default")
    return op if callable(op) else None


def grid():
    bindings = [{"shape": list(s), "layout": "contiguous"} for s in _SHAPES]
    bindings.append({"shape": list(_VIEW_BASE), "layout": "transpose"})
    bindings.append({"shape": list(_VIEW_BASE), "layout": "stride2_columns"})
    return bindings


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    layout = binding["layout"]
    if layout == "contiguous":
        return (torch.randn(shape, device=device, dtype=dtype),), {}
    x = torch.randn(_VIEW_BASE, device=device, dtype=dtype)
    if layout == "transpose":
        return (x.t(),), {}
    return (x[:, ::2],), {}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    if binding["layout"] == "contiguous":
        return list(binding["shape"])
    return f"{shape}_{binding['layout']}"


def config(binding, dtype):
    shape = list(binding["shape"])
    layout = binding["layout"]
    dt = str(dtype)
    if layout == "contiguous":
        in_shape, stride = shape, [1]
    elif layout == "transpose":
        in_shape, stride = [shape[1], shape[0]], [1, shape[1]]
    else:
        in_shape, stride = [shape[0], shape[1] // 2], [shape[1], 2]
    return {
        "inputs": {
            "args": [{"shape": in_shape, "dtype": dt, "stride": stride}],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": in_shape, "dtype": "torch.float64"}},
        "dims": {"shape": in_shape, "layout": layout},
    }
