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

"""NVIDIA native baseline for `copy_from_and_resize_out`.

Mirrors benchmark/test_copy_from_and_resize.py (CopyFromAndResizeOutBenchmark,
a base.Benchmark subclass):
    op_name  = "copy_from_and_resize_out"
    torch_op = _aten_copy_from_and_resize_out (the module-local reference)
    dtypes   = consts.FLOAT_DTYPES (one parametrized run each)

The parent class overrides set_shapes with RESIZE_SHAPE_PAIRS -- paired
(src, dst) shapes rather than single shapes -- and returns [] from
set_more_shapes, so core_shapes.yaml is not consulted.

CAUTION: there is no native CUDA kernel for this operator. The benchmark's
reference is the eager recipe the Lazy backends use -- resize dst to src's
shape, copy src into it, resize out, copy dst into out -- and that composition
is reproduced here verbatim. The measured latency is therefore three
resize/copy operations, not one fused kernel. See Q&A.md.
"""

import torch

OP_NAME = "copy_from_and_resize_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (1, "out")
NATIVE_UNAVAILABLE_REASON = "torch.Tensor.resize_ is unavailable on this torch build"

# test_copy_from_and_resize.py::RESIZE_SHAPE_PAIRS -- (SRC_SHAPE, DST_SHAPE)
_SHAPES = [
    ((4096, 4096), (1024, 1024)),
    ((1024, 1024), (4096, 4096)),
    ((4096, 4096), (4096, 4096)),
    ((2048, 2048), (8 * 1024 * 1024,)),
    ((8 * 1024 * 1024,), (2048, 2048)),
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


def _aten_copy_from_and_resize_out(src, dst, *, out):
    """benchmark::_aten_copy_from_and_resize_out.

    The native CUDA kernel is absent; this is the eager recipe the Lazy native
    implementations use -- dst resized in place, then the result mirrored into
    the resized out.
    """
    dst.resize_(src.shape)
    dst.copy_(src)
    out.resize_(src.shape)
    return out.copy_(dst)


def native():
    if _resolve("torch.Tensor.resize_") is None:
        return None
    return _aten_copy_from_and_resize_out


def grid():
    return [{"src_shape": list(src), "dst_shape": list(dst)} for src, dst in _SHAPES]


def build_inputs(binding, dtype, device):
    src_shape = tuple(binding["src_shape"])
    dst_shape = tuple(binding["dst_shape"])
    src = torch.randn(src_shape, dtype=dtype, device=device)
    dst = torch.randn(dst_shape, dtype=dtype, device=device)
    out = torch.empty(dst_shape, dtype=dtype, device=device)
    return (src, dst), {"out": out}


def key_shape(binding):
    src = "x".join(str(d) for d in binding["src_shape"])
    dst = "x".join(str(d) for d in binding["dst_shape"])
    return f"src{src}_dst{dst}"


def config(binding, dtype):
    src_shape = list(binding["src_shape"])
    dst_shape = list(binding["dst_shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": src_shape, "dtype": dt},
                {"shape": dst_shape, "dtype": dt, "note": "resized in place"},
            ],
            "kwargs": {
                "out": {"shape": dst_shape, "dtype": dt, "note": "resized to src"}
            },
        },
        "outputs": {"out": {"shape": src_shape, "dtype": dt}},
        "dims": {"src_shape": src_shape, "dst_shape": dst_shape},
    }
