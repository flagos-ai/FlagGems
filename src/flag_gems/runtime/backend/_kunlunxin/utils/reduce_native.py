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

"""Native-redispatch helpers for dim reductions.

flag_gems.utils.dim_compress() ends with ``inp.permute(order).contiguous()``.
For any reduction whose reduced dims are not already the trailing dims, the
permuted view is non-contiguous, so ``.contiguous()`` materializes it through
torch ``copy_`` -- which is broken on this backend for strided sources
("CUDA error: invalid device function").  When that materialization would
happen, callers redispatch to the vendor/native ATen kernels instead
(same strategy as the amin/amax multi-dim fix).

Accuracy notes (probed on P800):
  * aten::sum.dim_IntList / mean.dim / amax / amin are accurate for every
    probed shape/dtype (sum rel ~1e-7 even over 8M-element reductions).
  * aten::var.correction / var_mean.correction / std / norm.ScalarOpt_dim on
    this vendor build are NOT: var over an 8.2M-element multi-dim reduction
    is off by ~1e-2 absolute and norm p=2 by ~0.5% relative.  var/std/
    var_mean/norm are therefore COMPOSED from the accurate sum/amax/amin
    primitives instead (native_var_mean_parts / norm composition in ops).

Kernel handles are captured at IMPORT time and must be: ``use_gems()``
stacks flag_gems Python kernels on the CUDA key, and a get_kernel() made
afterwards returns that wrapper -- for some ops (e.g. the wrong-overload
aten::std.dim stub) redispatching through it re-enters the vendor op and
blows the stack (644-frame RecursionError, verified in test_std).  The
handles captured here are only the composed-on primitives, and import of
this module happens before any registration.

float16/bfloat16 inputs are upcast to float32 around native calls: several
vendor kernels accumulate in the input dtype (e.g. std on f16 returns nan
once the reduction exceeds ~1M elements), while the CPU reference
accumulates in float32.  The upcast cast itself goes through the vendor
pointwise cast path, which is only known to be safe for contiguous inputs,
so strided low-precision tensors are passed to the native kernel unchanged
(native kernels handle strided layouts correctly by themselves).
"""

import torch

_CUDA_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
_LOW_PRECISION = (torch.float16, torch.bfloat16)

_KERNELS = {}
for _op in (
    "aten::amax",
    "aten::amin",
    "aten::mean.dim",
    "aten::sum.dim_IntList",
):
    try:
        _KERNELS[_op] = torch.library.get_kernel(_op, "CUDA")
    except Exception:
        _KERNELS[_op] = None


def dim_compress_materializes(inp, dims):
    """True when dim_compress(inp, dims) would perform an actual device copy.

    Mirrors the layout logic of flag_gems.utils.dim_compress: if the permuted
    view is already contiguous, dim_compress is a no-op view chain and the
    (correct) Triton inner-dim kernels can keep handling it.
    """
    ndim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(ndim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return not inp.permute(order).is_contiguous()


def native_reduce(opname, x, *args, **kwargs):
    """Call the native ATen kernel `opname` on x, upcasting fp16/bf16 first.

    Results are cast back to the original dtype (single low-precision tensor
    or tuple of tensors).
    """
    kernel = _KERNELS.get(opname) or torch.library.get_kernel(opname, "CUDA")
    orig = x.dtype
    if orig in _LOW_PRECISION and x.is_contiguous():
        r = kernel.call_boxed(_CUDA_KEYSET, x.float(), *args, **kwargs)
        if isinstance(r, torch.Tensor):
            return r.to(orig)
        return tuple(t.to(orig) for t in r)
    return kernel.call_boxed(_CUDA_KEYSET, x, *args, **kwargs)


def native_sum_fp32(x, dim, keepdim=True):
    """float32 sum over `dim` via the accurate native sum kernel."""
    y = x.float() if x.dtype in _LOW_PRECISION else x
    return native_reduce("aten::sum.dim_IntList", y, dim=dim, keepdim=keepdim)


def native_var_mean_parts(x, dim, correction):
    """Accurate (var, mean) over `dim` composed from two exact sums.

    The vendor aten::var.correction / var_mean.correction kernels are
    inaccurate for large multi-dim reductions (~1e-2 absolute error probed
    at (200,40999,3) x dim=[0,1]), while sum is accurate to ~1e-7 relative.
    Compose instead:  var = (sum(x^2) - sum(x)^2 / n) / (n - correction).

    Returns float32 tensors with keepdim=True shapes; NaN var when
    n - correction <= 0 (aten semantics).
    """
    y = x.float() if x.dtype in _LOW_PRECISION else x
    s = native_reduce("aten::sum.dim_IntList", y, dim=dim, keepdim=True)
    ss = native_reduce("aten::sum.dim_IntList", y * y, dim=dim, keepdim=True)
    n = x.numel() // s.numel()
    mean = s / n
    denom = n - correction
    if denom > 0:
        var = (ss - s * s / n) / denom
    else:
        var = torch.full_like(mean, float("nan"))
    return var, mean
