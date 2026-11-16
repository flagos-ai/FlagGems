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

"""Kunlunxin/XPU wrapper around the generic ``svd``.

The generic ``flag_gems.ops.svd`` has a broad set of rank/Jacobi kernels and is
kept as the primary path (it is correct for most tested shapes and carries the
specialised rank-1/2 and tiny-matrix conventions).  On XPU it refuses some
medium shapes (``(16,8) (8,16) (64,32) (32,64)``: ``NotImplementedError``) and
has no low-precision (fp16/bf16) route.

Those cases are handled here by the backend's own one-sided Jacobi SVD
(``_kunlunxin/ops/linalg_svd.py``, pure Triton): compute in float32, return
``(U, S, V)`` with ``V = Vh^T`` and cast the factors back to the input dtype.
"""

import importlib
import logging
from collections import namedtuple

import torch

logger = logging.getLogger(__name__)

SvdResult = namedtuple("svd", ["U", "S", "V"])



def _generic_module():
    return importlib.import_module("flag_gems.ops.svd")


def _backend_linalg_svd_module():
    return importlib.import_module(".linalg_svd", __package__)


def _backend_svd(input, some=True, compute_uv=True):
    if input.is_complex():
        raise NotImplementedError(
            "FlagGems svd: complex inputs are not supported on this backend"
        )
    if input.numel() == 0:
        m, n = input.shape[-2:]
        k = min(m, n)
        batch = input.shape[:-2]
        dtype = input.dtype
        u_cols = m if not some else k
        v_cols = n if not some else k
        u = torch.empty((*batch, m, u_cols), dtype=dtype, device=input.device)
        s = torch.empty((*batch, k), dtype=dtype, device=input.device)
        v = torch.empty((*batch, n, v_cols), dtype=dtype, device=input.device)
        return SvdResult(u, s, v)

    orig_dtype = input.dtype
    need_cast = orig_dtype in (torch.float16, torch.bfloat16)
    work = input.to(torch.float32) if need_cast else input
    if not work.is_contiguous():
        work = work.contiguous()

    # The backend SVD consumes a single batch dimension.
    m, n = work.shape[-2:]
    batch_shape = work.shape[:-2]
    batch = 1
    for d in batch_shape:
        batch *= d
    flat = work.reshape(batch, m, n)

    u, s, vh = _backend_linalg_svd_module().linalg_svd(
        flat, full_matrices=not some
    )
    v = vh.transpose(-2, -1)

    if not compute_uv:
        s = s.reshape(*batch_shape, s.shape[-1])
        if need_cast:
            s = s.to(orig_dtype)
        return SvdResult(None, s, None)

    u = u.reshape(*batch_shape, u.shape[-2], u.shape[-1])
    s = s.reshape(*batch_shape, s.shape[-1])
    v = v.reshape(*batch_shape, n, v.shape[-1])
    if need_cast:
        u = u.to(orig_dtype)
        s = s.to(orig_dtype)
        v = v.to(orig_dtype)
    return SvdResult(u, s, v)


def svd(input, some=True, compute_uv=True):
    """Singular value decomposition; routes XPU-unsupported shapes to the backend."""
    logger.debug("GEMS_KUNLUNXIN SVD")
    use_backend = input.dtype in (torch.float16, torch.bfloat16)
    if not use_backend:
        try:
            return _generic_module().svd(input, some=some, compute_uv=compute_uv)
        except NotImplementedError:
            use_backend = True
    return _backend_svd(input, some=some, compute_uv=compute_uv)
