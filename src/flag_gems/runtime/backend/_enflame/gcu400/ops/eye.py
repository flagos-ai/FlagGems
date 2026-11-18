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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
device_ = device

_EYE_I64_DTYPES = (torch.int64, torch.uint64)


def _install_i64_safe_eye_shim():
    # torch_gcu P21 workaround: topsaten native eye aborts the process for
    # 64-bit integer dtypes (op_aten_eye_gcu400.h:209 Invalid Data Type ->
    # std::runtime_error terminate, rc=134, no Python exception to catch).
    # Exactly those calls are generated on CPU and copied back H2D (int64
    # H2D copies work); every other call shape passes through untouched,
    # chaining under/over the complex-dtype shim installed by
    # _linalg_solve_ex when both are present.
    current = torch.eye
    if getattr(current, "_flaggems_gcu_i64_safe", False):
        return

    def _i64_safe_eye(*args, **kwargs):
        dtype = kwargs.get("dtype")
        if dtype not in _EYE_I64_DTYPES:
            return current(*args, **kwargs)
        device = kwargs.get("device")
        if device is None:
            return current(*args, **kwargs)
        try:
            target = torch.device(device)
        except (TypeError, ValueError):
            return current(*args, **kwargs)
        if target.type != "gcu":
            return current(*args, **kwargs)
        kw = dict(kwargs)
        kw["device"] = "cpu"
        out = kwargs.get("out")
        if out is not None:
            kw.pop("out")
            out.copy_(current(*args, **kw))
            return out
        return current(*args, **kw).to(target)

    _i64_safe_eye._flaggems_gcu_i64_safe = True
    torch.eye = _i64_safe_eye


_install_i64_safe_eye_shim()


@libentry()
@triton.jit
def eye_kernel(
    out_ptr,
    N,
    M,
    BLOCK_i: tl.constexpr,
    BLOCK_j: tl.constexpr,
    MAX_GRID_DIM_x: tl.constexpr,
    MAX_GRID_DIM_y: tl.constexpr,
):
    pid_i = tl.program_id(0)  # block id
    grim_size_x = (N + MAX_GRID_DIM_x - 1) // MAX_GRID_DIM_x
    pid_j = tl.program_id(1)  # block id
    grim_size_y = (M + MAX_GRID_DIM_y - 1) // MAX_GRID_DIM_y
    for i in range(0, grim_size_x, BLOCK_i):
        block_start_x = pid_i * grim_size_x + i
        off_i = block_start_x + tl.arange(0, BLOCK_i)
        mask_i = (off_i < N) & (off_i < pid_i * grim_size_x + grim_size_x)
        for j in range(0, grim_size_y, BLOCK_j):
            block_start_y = pid_j * grim_size_y + j
            off_j = block_start_y + tl.arange(0, BLOCK_j)
            mask_j = (off_j < M) & (off_j < pid_j * grim_size_y + grim_size_y)

            val = tl.where(off_i[:, None] == off_j[None, :], 1.0, 0.0)
            mask = mask_i[:, None] & mask_j[None, :]
            off_ij = off_i[:, None] * M + off_j[None, :]

            tl.store(out_ptr + off_ij, val, mask=mask)


def eye(size, *, dtype=None, layout=torch.strided, device=None, pin_memory=None):
    """
    Triton-based implementation of torch.eye(n, n), using 2D tiles to split the matrix into blocks.
    """
    logger.debug("GEMS_ENFLAME EYE")

    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    if layout != torch.strided:
        raise ValueError("Currently only strided layout is supported for eye.")

    out = torch.empty(
        (size, size), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
    BLOCK_SIZE = 128 if dtype.itemsize >= 8 else 256
    MAX_GRID_DIM_x = 12
    MAX_GRID_DIM_y = 12
    grid = (MAX_GRID_DIM_x, MAX_GRID_DIM_y)

    with torch_device_fn.device(device):
        eye_kernel[grid](
            out,
            size,
            size,
            BLOCK_SIZE,
            BLOCK_SIZE,
            MAX_GRID_DIM_x,
            MAX_GRID_DIM_y,
        )
    return out
