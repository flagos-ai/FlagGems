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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.erfinv_ import erfinv_ as default_erfinv_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _erfinv(xf):
    # Polynomial approximation of the inverse error function evaluated in fp32.
    # The MThreads llc backend cannot lower the libdevice erfinv intrinsic, so we
    # inline the same rational approximation the generic experimental kernel uses.
    one = 1.0
    absx = tl.abs(xf)
    w = -tl.log((one - xf) * (one + xf))

    use_low = w < 5.0

    wl = w - 2.5
    pl = 2.81022636e-08
    pl = 3.43273939e-07 + pl * wl
    pl = -3.5233877e-06 + pl * wl
    pl = -4.39150654e-06 + pl * wl
    pl = 2.1858087e-04 + pl * wl
    pl = -1.25372503e-03 + pl * wl
    pl = -4.17768164e-03 + pl * wl
    pl = 2.46640727e-01 + pl * wl
    pl = 1.50140941e00 + pl * wl

    wh = tl.sqrt(w) - 3.0
    ph = -2.00214257e-04
    ph = 1.00950558e-04 + ph * wh
    ph = 1.34934322e-03 + ph * wh
    ph = -3.67342844e-03 + ph * wh
    ph = 5.73950773e-03 + ph * wh
    ph = -7.62246130e-03 + ph * wh
    ph = 9.43887047e-03 + ph * wh
    ph = 1.00167406e00 + ph * wh
    ph = 2.83297682e00 + ph * wh

    p = tl.where(use_low, pl, ph)
    res = p * xf

    nan_val = float("nan")
    inf_val = float("inf")
    res = tl.where(xf != xf, nan_val, res)
    res = tl.where(absx > 1.0, nan_val, res)
    res = tl.where(xf == 1.0, inf_val, res)
    res = tl.where(xf == -1.0, -inf_val, res)
    return res


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256, "VEC": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 4}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 2}, num_warps=8, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
    # Inplace: autotune reruns the kernel on the same buffer, so restore the
    # input between trials to avoid applying erfinv repeatedly in place.
    restore_value=["x_ptr"],
)
@triton.jit
def erfinv_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dtype_size,  # used for autotune key
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(0)
    BLOCK_ELEMS: tl.constexpr = BLOCK_SIZE * VEC
    offsets = (pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)).to(tl.int64)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Moore Threads hardware does not support fp64; compute in fp32.
    out = _erfinv(x.to(tl.float32)).to(x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)


# NOTE: the libdevice erfinv intrinsic crashes the MThreads llc backend, so the
# polynomial helper above is used instead of tl_extra_shim.erfinv.


def _use_triton_kernel(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not x.is_contiguous() or x.numel() == 0:
        return False
    return True


def _launch_erfinv(x: torch.Tensor, out: torch.Tensor):
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    n_elements = out_flat.numel()
    dtype_size = out_flat.element_size()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"] * META["VEC"]),)
    with torch_device_fn.device(out.device):
        erfinv_kernel[grid](x_flat, out_flat, n_elements, dtype_size)
    return out


def erfinv_(x):
    logger.debug("GEMS_MTHREADS ERFINV_")
    if not _use_triton_kernel(x):
        return default_erfinv_(x)

    return _launch_erfinv(x, x)
