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

"""Storage-only float64 compatibility helpers for Kunlunxin.

Kunlunxin's native allocator rewrites requested float64 allocations to
float32.  An int64 allocation has the required eight-byte element width and
can be viewed as float64 without invoking that rewrite.  Kernels using these
helpers deliberately compute in float32; they preserve dtype/storage semantics
but do not claim native float64 arithmetic support.
"""

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic


def _shape_tuple(size):
    if isinstance(size, int):
        return (size,)
    return tuple(size)


def empty_fp64(size, *, device):
    """Allocate contiguous float64 metadata over an int64 device allocation."""
    raw = torch.empty(_shape_tuple(size), dtype=torch.int64, device=device)
    return raw.view(torch.float64)


def empty_complex128(size, *, device):
    """Allocate contiguous complex128 metadata over an int64 allocation."""
    real = empty_fp64((*_shape_tuple(size), 2), device=device)
    return torch.view_as_complex(real)


def is_fp64_cuda_tensor(value):
    return (
        isinstance(value, torch.Tensor)
        and value.device.type == "cuda"
        and value.dtype == torch.float64
    )


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _mul_tensor_fp32(x, y):
    return x.to(tl.float32) * y.to(tl.float32)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _mul_scalar_fp32(x, y):
    return x.to(tl.float32) * y.to(tl.float32)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _sub_tensor_fp32(x, y, alpha):
    return x.to(tl.float32) - y.to(tl.float32) * alpha.to(tl.float32)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def _sub_scalar_fp32(x, y, alpha):
    return x.to(tl.float32) - y.to(tl.float32) * alpha.to(tl.float32)


def mul_fp64(self, other):
    """Multiply into float64 storage while evaluating in float32."""
    if not is_fp64_cuda_tensor(self):
        raise ValueError("mul_fp64 expects a CUDA float64 tensor as self")

    if isinstance(other, torch.Tensor):
        shape = torch.broadcast_shapes(self.shape, other.shape)
        out = empty_fp64(shape, device=self.device)
        _mul_tensor_fp32(self, other, out0=out)
    else:
        out = empty_fp64(self.shape, device=self.device)
        _mul_scalar_fp32(self, other, out0=out)
    return out


def sub_fp64(self, other, *, alpha=1):
    """Subtract into float64 storage while evaluating in float32."""
    if not is_fp64_cuda_tensor(self):
        raise ValueError("sub_fp64 expects a CUDA float64 tensor as self")

    if isinstance(other, torch.Tensor):
        shape = torch.broadcast_shapes(self.shape, other.shape)
        out = empty_fp64(shape, device=self.device)
        _sub_tensor_fp32(self, other, alpha, out0=out)
    else:
        out = empty_fp64(self.shape, device=self.device)
        _sub_scalar_fp32(self, other, alpha, out0=out)
    return out
