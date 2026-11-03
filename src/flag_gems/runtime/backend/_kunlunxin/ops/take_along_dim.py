# Copyright 2026, The FlagOS Contributors.
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

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def take_along_dim_kernel(
    inp_ptr,
    idx_ptr,
    out_ptr,
    R,
    D_in,
    D_out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # int32 offset arithmetic: XPU has no native int64 div/mul, so the generic
    # kernel's int64 address math is emulated and dominates the launch.
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    r = offs // D_out
    idx = tl.load(idx_ptr + offs, mask=mask, other=0).to(tl.int32)
    idx = tl.where(idx < 0, idx + D_in, idx)

    valid = (idx >= 0) & (idx < D_in)
    # The generic kernel's `tl.device_assert(valid | ~mask)` compiles to a
    # device-side assert that forces a synchronous global-memory check on every
    # program on XPU, adding a ~2 second fixed cost per launch (measured
    # [64,64]: ~2.2s vs torch 0.008ms).  Drop it and clamp invalid indices to a
    # safe in-bounds offset; the masked store discards out-of-range results.
    in_off = r * D_in + tl.where(valid, idx, 0)
    vals = tl.load(inp_ptr + in_off, mask=mask & valid, other=0)
    tl.store(out_ptr + offs, vals, mask=mask)


def _launch(inp2d, idx2d, out2d):
    R, D_in = inp2d.shape
    _, D_out = idx2d.shape
    N = out2d.numel()
    if N == 0:
        return
    # NOTE (kunlunxin/XPU): a callable (lambda) grid makes triton re-specialize
    # the launch on every call here, adding ~190ms per launch (measured).  Use a
    # plain tuple grid instead; BLOCK_SIZE is fixed so it can be precomputed.
    grid = (triton.cdiv(N, 1024),)
    with torch_device_fn.device(inp2d.device):
        take_along_dim_kernel[grid](
            inp2d, idx2d, out2d, R, D_in, D_out, N, BLOCK_SIZE=1024
        )


def _patch_generic_wrapper():
    """Route ``flag_gems.ops.take_along_dim`` to this backend-local override.

    The generic op is a plain function (no SpecOpRegistrar / libentry), so tests
    and benchmarks import it directly from ``flag_gems.ops.take_along_dim``.
    Patching the module attributes keeps the change backend-local (the generic
    source is untouched and other vendors keep the assert-based kernel).
    """
    import sys

    try:
        _generic_module = sys.modules.get("flag_gems.ops.take_along_dim")
        if _generic_module is not None:
            _generic_module._launch = _launch
            _generic_module.take_along_dim_kernel = take_along_dim_kernel
    except ImportError:
        pass


_patch_generic_wrapper()


__all__ = ["take_along_dim_kernel", "_launch"]
