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

from flag_gems.ops.ldl_factor_ex import ldl_factor_ex as default_ldl_factor_ex

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float32}


@triton.jit
def _ldl_ex_kernel(
    A_ptr,
    LD_ptr,
    PIV_ptr,
    INFO_ptr,
    n,
    sa0,
    sa1,
    sld0,
    sld1,
    ba,
    bld,
    bpiv,
    NMAX: tl.constexpr,
):
    pid = tl.program_id(0)
    A_ptr += pid * ba
    LD_ptr += pid * bld
    PIV_ptr += pid * bpiv
    INFO_ptr += pid

    ALPHA = 0.6403882032022076

    jj = tl.arange(0, NMAX)

    # init: LD = tril(A), upper = 0 (single 2D load/store, column-major LD)
    ii = tl.arange(0, NMAX)
    m2 = (ii[:, None] < n) & (jj[None, :] < n)
    row = tl.load(A_ptr + ii[:, None] * sa0 + jj[None, :] * sa1, mask=m2, other=0.0)
    row = tl.where(jj[None, :] <= ii[:, None], row, 0.0)
    tl.store(LD_ptr + ii[:, None] * sld0 + jj[None, :] * sld1, row, mask=m2)

    k = 0
    info = 0
    while k < n:
        absakk = tl.abs(tl.load(LD_ptr + k * sld0 + k * sld1))
        offs = k + 1 + jj
        colmask = offs < n
        col = tl.load(LD_ptr + offs * sld0 + k * sld1, mask=colmask, other=0.0)
        colabs = tl.where(colmask, tl.abs(col), -1.0e30)
        colmax = tl.max(tl.where(colmask, tl.abs(col), 0.0))
        imax = k + 1 + tl.argmax(colabs, axis=0)

        kp = k
        kstep = 1
        skip = 0
        if tl.maximum(absakk, colmax) == 0:
            if info == 0:
                info = k + 1
            skip = 1
        else:
            if absakk >= ALPHA * colmax:
                kp = k
            else:
                rj = k + jj
                rmask = rj < imax
                rowmax = tl.max(
                    tl.abs(
                        tl.load(LD_ptr + imax * sld0 + rj * sld1, mask=rmask, other=0.0)
                    )
                )
                o2 = imax + 1 + jj
                cmask2 = o2 < n
                colseg = tl.abs(
                    tl.load(LD_ptr + o2 * sld0 + imax * sld1, mask=cmask2, other=0.0)
                )
                rowmax = tl.maximum(rowmax, tl.max(colseg))
                if absakk >= ALPHA * colmax * (colmax / rowmax):
                    kp = k
                else:
                    if (
                        tl.abs(tl.load(LD_ptr + imax * sld0 + imax * sld1))
                        >= ALPHA * rowmax
                    ):
                        kp = imax
                    else:
                        kp = imax
                        kstep = 2

        kk = k + kstep - 1
        if kp != kk:
            # swap rows/cols kk and kp within trailing submatrix (lower triangle)
            sw = kp + 1 + jj
            swm = sw < n
            c1 = tl.load(LD_ptr + sw * sld0 + kk * sld1, mask=swm, other=0.0)
            c2 = tl.load(LD_ptr + sw * sld0 + kp * sld1, mask=swm, other=0.0)
            tl.store(LD_ptr + sw * sld0 + kk * sld1, c2, mask=swm)
            tl.store(LD_ptr + sw * sld0 + kp * sld1, c1, mask=swm)
            mid = kk + 1 + jj
            midm = mid < kp
            s1 = tl.load(LD_ptr + mid * sld0 + kk * sld1, mask=midm, other=0.0)
            s2 = tl.load(LD_ptr + kp * sld0 + mid * sld1, mask=midm, other=0.0)
            tl.store(LD_ptr + mid * sld0 + kk * sld1, s2, mask=midm)
            tl.store(LD_ptr + kp * sld0 + mid * sld1, s1, mask=midm)
            dkk = tl.load(LD_ptr + kk * sld0 + kk * sld1)
            dkp = tl.load(LD_ptr + kp * sld0 + kp * sld1)
            tl.store(LD_ptr + kk * sld0 + kk * sld1, dkp)
            tl.store(LD_ptr + kp * sld0 + kp * sld1, dkk)
            if kstep == 2:
                e1 = tl.load(LD_ptr + (k + 1) * sld0 + k * sld1)
                e2 = tl.load(LD_ptr + kp * sld0 + k * sld1)
                tl.store(LD_ptr + (k + 1) * sld0 + k * sld1, e2)
                tl.store(LD_ptr + kp * sld0 + k * sld1, e1)

        if skip == 0:
            if kstep == 1:
                # rank-1 update of trailing block T(k+1:n, k+1:n) as 2D tile.
                x = tl.load(LD_ptr + offs * sld0 + k * sld1, mask=colmask, other=0.0)
                dval = tl.load(LD_ptr + k * sld0 + k * sld1)
                d11 = 1.0 / dval
                w = x * d11
                rm = (jj[None, :] <= ii[:, None]) & colmask[:, None] & colmask[None, :]
                blk = tl.load(
                    LD_ptr
                    + (k + 1 + ii)[:, None] * sld0
                    + (k + 1 + jj)[None, :] * sld1,
                    mask=rm,
                    other=0.0,
                )
                blk = blk - x[:, None] * (x[None, :] * d11)
                tl.store(
                    LD_ptr
                    + (k + 1 + ii)[:, None] * sld0
                    + (k + 1 + jj)[None, :] * sld1,
                    blk,
                    mask=rm,
                )
                tl.store(LD_ptr + offs * sld0 + k * sld1, w, mask=colmask)
            else:
                # rank-2: sequential column update exactly as LAPACK DSYTF2
                o3 = k + 2 + jj
                m3 = o3 < n
                u = tl.load(LD_ptr + o3 * sld0 + k * sld1, mask=m3, other=0.0)
                v = tl.load(LD_ptr + o3 * sld0 + (k + 1) * sld1, mask=m3, other=0.0)
                d21 = tl.load(LD_ptr + (k + 1) * sld0 + k * sld1)
                d11b = tl.load(LD_ptr + (k + 1) * sld0 + (k + 1) * sld1) / d21
                d22b = tl.load(LD_ptr + k * sld0 + k * sld1) / d21
                t = 1.0 / (d11b * d22b - 1.0)
                d21s = t / d21
                w = d21s * (d11b * u - v)
                # sequential: for jj_ in k+2..n-1: update column jj_ (rows jj_..n-1)
                for j in range(0, NMAX):
                    jj_ = k + 2 + j
                    jm = jj_ < n
                    if jm:
                        wj = tl.load(LD_ptr + jj_ * sld0 + k * sld1)
                        zj = tl.load(LD_ptr + jj_ * sld0 + (k + 1) * sld1)
                        wk = d21s * (d11b * wj - zj)
                        wkp1 = d21s * (d22b * zj - wj)
                        rr = jj_ + jj
                        rrm = rr < n
                        colv = tl.load(
                            LD_ptr + rr * sld0 + jj_ * sld1, mask=rrm, other=0.0
                        )
                        ck = tl.load(LD_ptr + rr * sld0 + k * sld1, mask=rrm, other=0.0)
                        ck1 = tl.load(
                            LD_ptr + rr * sld0 + (k + 1) * sld1, mask=rrm, other=0.0
                        )
                        colv = colv - ck * wk - ck1 * wkp1
                        tl.store(LD_ptr + rr * sld0 + jj_ * sld1, colv, mask=rrm)
                        tl.store(LD_ptr + jj_ * sld0 + k * sld1, wk)
                        tl.store(LD_ptr + jj_ * sld0 + (k + 1) * sld1, wkp1)

        if kstep == 1:
            tl.store(PIV_ptr + k, kp + 1)
        else:
            tl.store(PIV_ptr + k, -(kp + 1))
            tl.store(PIV_ptr + k + 1, -(kp + 1))
        k = k + kstep

    tl.store(INFO_ptr, info)


def _specialized_ldl_factor_ex(self, *, hermitian=False, check_errors=False):
    ndim = self.dim()
    n = self.shape[-1]
    batch_shape = self.shape[:-2]
    batch = 1
    for s in batch_shape:
        batch *= s
    dev = self.device
    dtype = self.dtype

    if self.is_complex():
        raise NotImplementedError(
            "torch.linalg.ldl_factor_ex: complex inputs are not supported by this kernel yet"
        )

    if ndim == 2:
        ld_strides = (1, n)
    else:
        ld_strides = (n * n,) * (ndim - 2) + (1, n)
    LD = torch.empty_strided(self.shape, ld_strides, dtype=dtype, device=dev)
    pivots = torch.empty((*batch_shape, n), dtype=torch.int32, device=dev)
    info = torch.zeros(batch_shape, dtype=torch.int32, device=dev)

    if self.numel() == 0 or n == 0:
        return LD, pivots, info

    NMAX = triton.next_power_of_2(n)
    sa0 = self.stride(-2)
    sa1 = self.stride(-1)
    ba = self.stride(-3) if ndim > 2 else 0
    num_warps = 4 if NMAX <= 128 else 8

    _ldl_ex_kernel[(batch,)](
        self,
        LD,
        pivots,
        info,
        n,
        sa0,
        sa1,
        1,
        n,
        ba,
        n * n,
        n,
        NMAX=NMAX,
        num_warps=num_warps,
    )

    if check_errors and int(info.flatten().max().item()) != 0:
        raise RuntimeError(
            "torch.linalg.ldl_factor_ex: The factorization could not be completed "
            "because the input is singular or has a zero pivot (info != 0)."
        )
    return LD, pivots, info


def ldl_factor_ex(self, *, hermitian=False, check_errors=False):
    logger.debug("GEMS_MTHREADS LDL_FACTOR_EX")
    if (
        isinstance(self, torch.Tensor)
        and self.device.type == "musa"
        and self.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_ldl_factor_ex(
            self, hermitian=hermitian, check_errors=check_errors
        )
    return default_ldl_factor_ex(self, hermitian=hermitian, check_errors=check_errors)
