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
#
# Triton implementation of linalg_ldl_solve.
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

LDL_SOLVE_RHS_BLOCK = 32
LDL_SOLVE_CPLX_RHS_BLOCK = 16
LDL_SOLVE_REG_MAX_N = 128
REAL_DTYPES = (torch.float32, torch.float64)
COMPLEX_DTYPES = (torch.complex64, torch.complex128)
WIDE_DTYPES = (torch.float64, torch.complex128)
NARROW_OF = {torch.float64: torch.float32, torch.complex128: torch.complex64}


@triton.jit
def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def _cconj_mul(ar, ai, br, bi):
    return ar * br + ai * bi, ar * bi - ai * br


@triton.jit
def _cdiv(nr, ni, dr, di):
    denom = dr * dr + di * di
    return (nr * dr + ni * di) / denom, (ni * dr - nr * di) / denom


@triton.jit
def _row_at(w, index):
    # Row `index` of the register tile w as a 1D vector. GCU400 marks tt.gather
    # illegal, so use a one-hot weighted reduction (tl.gather semantics).
    row_indices = tl.arange(0, w.shape[0])
    return tl.sum(tl.where(row_indices[:, None] == index, w, 0.0), 0)


@triton.jit
def _scalar_at(v, index):
    idx = tl.arange(0, v.shape[0])
    return tl.sum(tl.where(idx == index, v, 0), 0)


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "nrhs",
        "ld_batch_stride",
        "ld_row_stride",
        "ld_col_stride",
        "piv_batch_stride",
        "piv_row_stride",
        "x_batch_stride",
        "x_row_stride",
        "x_col_stride",
    ]
)
def linalg_ldl_solve_real_kernel(
    LD,
    pivots,
    X,
    n,
    nrhs,
    ld_batch_stride,
    ld_row_stride,
    ld_col_stride,
    piv_batch_stride,
    piv_row_stride,
    x_batch_stride,
    x_row_stride,
    x_col_stride,
    BLOCK_N: tl.constexpr,
    BLOCK_NRH: tl.constexpr,
):
    # Register-resident Bunch-Kaufman solve: the whole [BLOCK_N, BLOCK_NRH]
    # block of X stays in registers for both substitution sweeps, so every
    # serial step is an intra-warp one-hot row extraction plus a rank-1 tile
    # update instead of a global-memory row round trip. num_warps=1 keeps all
    # cross-lane reductions warp-local (linalg_lu_solve gather precedent).
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    cols = block_idx * BLOCK_NRH + tl.arange(0, BLOCK_NRH)
    col_mask = cols < nrhs
    rows = tl.arange(0, BLOCK_N)
    row_mask = rows < n

    ld_base = LD + batch_idx * ld_batch_stride
    piv_base = pivots + batch_idx * piv_batch_stride
    x_base = X + batch_idx * x_batch_stride

    piv_vec = tl.load(piv_base + rows * piv_row_stride, mask=row_mask, other=1).to(
        tl.int32
    )
    w = tl.load(
        x_base + rows[:, None] * x_row_stride + cols[None, :] * x_col_stride,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )

    k = 0
    while k < n:
        ip = _scalar_at(piv_vec, k)
        if ip > 0:
            kp = ip - 1
            row_k = _row_at(w, k)
            row_kp = _row_at(w, kp)
            if kp != k:
                w = tl.where(rows[:, None] == k, row_kp[None, :], w)
                w = tl.where(rows[:, None] == kp, row_k[None, :], w)
            xk = row_kp
            l_col = tl.load(
                ld_base + rows * ld_row_stride + k * ld_col_stride,
                mask=(rows > k) & row_mask,
                other=0.0,
            )
            d = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            upd = w - l_col[:, None] * xk[None, :]
            w = tl.where(rows[:, None] == k, (xk / d)[None, :], upd)
            k += 1
        else:
            kp = -ip - 1
            row_k = _row_at(w, k)
            row_k1 = _row_at(w, k + 1)
            row_kp = _row_at(w, kp)
            if kp != k + 1:
                w = tl.where(rows[:, None] == k + 1, row_kp[None, :], w)
                w = tl.where(rows[:, None] == kp, row_k1[None, :], w)
            xk = row_k
            xk1 = row_kp
            sub_mask = (rows > k + 1) & row_mask
            l0 = tl.load(
                ld_base + rows * ld_row_stride + k * ld_col_stride,
                mask=sub_mask,
                other=0.0,
            )
            l1 = tl.load(
                ld_base + rows * ld_row_stride + (k + 1) * ld_col_stride,
                mask=sub_mask,
                other=0.0,
            )
            upd = w - l0[:, None] * xk[None, :] - l1[:, None] * xk1[None, :]
            b = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride)
            a = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            c = tl.load(ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride)
            akm1 = a / b
            ak = c / b
            denom = akm1 * ak - 1
            bkm1 = xk / b
            bk = xk1 / b
            new_k = (ak * bkm1 - bk) / denom
            new_k1 = (akm1 * bk - bkm1) / denom
            w = tl.where(rows[:, None] == k, new_k[None, :], upd)
            w = tl.where(rows[:, None] == k + 1, new_k1[None, :], w)
            k += 2

    k = n - 1
    while k >= 0:
        ip = _scalar_at(piv_vec, k)
        if ip > 0:
            kp = ip - 1
            row_k = _row_at(w, k)
            l_col = tl.load(
                ld_base + rows * ld_row_stride + k * ld_col_stride,
                mask=(rows > k) & row_mask,
                other=0.0,
            )
            acc = row_k - tl.sum(l_col[:, None] * w, 0)
            if kp != k:
                row_kp = _row_at(w, kp)
                w = tl.where(rows[:, None] == kp, acc[None, :], w)
                w = tl.where(rows[:, None] == k, row_kp[None, :], w)
            else:
                w = tl.where(rows[:, None] == k, acc[None, :], w)
            k -= 1
        else:
            row_km1 = _row_at(w, k - 1)
            row_k = _row_at(w, k)
            sub_mask = (rows > k) & row_mask
            l0 = tl.load(
                ld_base + rows * ld_row_stride + (k - 1) * ld_col_stride,
                mask=sub_mask,
                other=0.0,
            )
            l1 = tl.load(
                ld_base + rows * ld_row_stride + k * ld_col_stride,
                mask=sub_mask,
                other=0.0,
            )
            red0 = tl.sum(l0[:, None] * w, 0)
            red1 = tl.sum(l1[:, None] * w, 0)
            acc_km1 = row_km1 - red0
            acc_k = row_k - red1
            kp = -ip - 1
            if kp != k:
                row_kp = _row_at(w, kp)
                w = tl.where(rows[:, None] == kp, acc_k[None, :], w)
                w = tl.where(rows[:, None] == k, row_kp[None, :], w)
            else:
                w = tl.where(rows[:, None] == k, acc_k[None, :], w)
            w = tl.where(rows[:, None] == k - 1, acc_km1[None, :], w)
            k -= 2

    tl.store(
        x_base + rows[:, None] * x_row_stride + cols[None, :] * x_col_stride,
        w,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "nrhs",
        "ld_batch_stride",
        "ld_row_stride",
        "ld_col_stride",
        "piv_batch_stride",
        "piv_row_stride",
        "x_batch_stride",
        "x_row_stride",
        "x_col_stride",
    ]
)
def linalg_ldl_solve_real_kernel_global(
    LD,
    pivots,
    X,
    n,
    nrhs,
    ld_batch_stride,
    ld_row_stride,
    ld_col_stride,
    piv_batch_stride,
    piv_row_stride,
    x_batch_stride,
    x_row_stride,
    x_col_stride,
    BLOCK_NRH: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    col_start = block_idx * BLOCK_NRH
    cols = col_start + tl.arange(0, BLOCK_NRH)
    col_mask = cols < nrhs

    ld_base = LD + batch_idx * ld_batch_stride
    piv_base = pivots + batch_idx * piv_batch_stride
    x_base = X + batch_idx * x_batch_stride
    col_offsets = cols * x_col_stride

    k = 0
    while k < n:
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            kp = ip - 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                xk = tl.load(row_k_ptr, mask=col_mask, other=0.0)
                xkp = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp, mask=col_mask)
                tl.store(row_kp_ptr, xk, mask=col_mask)
            xk = tl.load(row_k_ptr, mask=col_mask, other=0.0)

            i = k + 1
            while i < n:
                lij = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                xi = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xi -= lij * xk
                tl.store(row_i_ptr, xi, mask=col_mask)
                i += 1

            d = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            xk = xk / d
            tl.store(row_k_ptr, xk, mask=col_mask)
            k += 1
        else:
            kp = -ip - 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k1_ptr = x_base + (k + 1) * x_row_stride + col_offsets
            if kp != k + 1:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                xk1 = tl.load(row_k1_ptr, mask=col_mask, other=0.0)
                xkp = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                tl.store(row_k1_ptr, xkp, mask=col_mask)
                tl.store(row_kp_ptr, xk1, mask=col_mask)
            xk = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            xk1 = tl.load(row_k1_ptr, mask=col_mask, other=0.0)

            i = k + 2
            while i < n:
                l0 = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l1 = tl.load(ld_base + i * ld_row_stride + (k + 1) * ld_col_stride)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                xi = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xi -= l0 * xk + l1 * xk1
                tl.store(row_i_ptr, xi, mask=col_mask)
                i += 1

            b = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride)
            a = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            c = tl.load(ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride)
            akm1 = a / b
            ak = c / b
            denom = akm1 * ak - 1
            bkm1 = xk / b
            bk = xk1 / b
            xk = (ak * bkm1 - bk) / denom
            xk1 = (akm1 * bk - bkm1) / denom
            tl.store(row_k_ptr, xk, mask=col_mask)
            tl.store(row_k1_ptr, xk1, mask=col_mask)
            k += 2

    k = n - 1
    while k >= 0:
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            xk = tl.load(row_k_ptr, mask=col_mask, other=0.0)

            i = k + 1
            while i < n:
                lij = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                xi = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xk -= lij * xi
                i += 1

            kp = ip - 1
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                xkp = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp, mask=col_mask)
                tl.store(row_kp_ptr, xk, mask=col_mask)
            else:
                tl.store(row_k_ptr, xk, mask=col_mask)
            k -= 1
        else:
            row_km1_ptr = x_base + (k - 1) * x_row_stride + col_offsets
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            xkm1 = tl.load(row_km1_ptr, mask=col_mask, other=0.0)
            xk = tl.load(row_k_ptr, mask=col_mask, other=0.0)

            i = k + 1
            while i < n:
                l0 = tl.load(ld_base + i * ld_row_stride + (k - 1) * ld_col_stride)
                l1 = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                xi = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xkm1 -= l0 * xi
                xk -= l1 * xi
                i += 1

            kp = -ip - 1
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                xkp = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp, mask=col_mask)
                tl.store(row_kp_ptr, xk, mask=col_mask)
            else:
                tl.store(row_k_ptr, xk, mask=col_mask)
            tl.store(row_km1_ptr, xkm1, mask=col_mask)
            k -= 2


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "nrhs",
        "ld_batch_stride",
        "ld_row_stride",
        "ld_col_stride",
        "piv_batch_stride",
        "piv_row_stride",
        "x_batch_stride",
        "x_row_stride",
        "x_col_stride",
    ]
)
def linalg_ldl_solve_complex_kernel(
    LD,
    pivots,
    X,
    n,
    nrhs,
    ld_batch_stride,
    ld_row_stride,
    ld_col_stride,
    piv_batch_stride,
    piv_row_stride,
    x_batch_stride,
    x_row_stride,
    x_col_stride,
    HERM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NRH: tl.constexpr,
):
    # Register-resident complex solve: real/imag live as separate [BLOCK_N,
    # BLOCK_NRH] tiles; every serial step is intra-warp one-hot row extraction
    # plus rank-1 tile updates (linalg_lu_solve gather precedent).
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    cols = block_idx * BLOCK_NRH + tl.arange(0, BLOCK_NRH)
    col_mask = cols < nrhs
    rows = tl.arange(0, BLOCK_N)
    row_mask = rows < n

    ld_base = LD + batch_idx * ld_batch_stride
    piv_base = pivots + batch_idx * piv_batch_stride
    x_base = X + batch_idx * x_batch_stride
    x_ptrs = x_base + rows[:, None] * x_row_stride + cols[None, :] * x_col_stride
    tile_mask = row_mask[:, None] & col_mask[None, :]

    piv_vec = tl.load(piv_base + rows * piv_row_stride, mask=row_mask, other=1).to(
        tl.int32
    )
    w_r = tl.load(x_ptrs, mask=tile_mask, other=0.0)
    w_i = tl.load(x_ptrs + 1, mask=tile_mask, other=0.0)

    k = 0
    while k < n:
        ip = _scalar_at(piv_vec, k)
        if ip > 0:
            kp = ip - 1
            row_k_r = _row_at(w_r, k)
            row_k_i = _row_at(w_i, k)
            row_kp_r = _row_at(w_r, kp)
            row_kp_i = _row_at(w_i, kp)
            if kp != k:
                w_r = tl.where(rows[:, None] == k, row_kp_r[None, :], w_r)
                w_i = tl.where(rows[:, None] == k, row_kp_i[None, :], w_i)
                w_r = tl.where(rows[:, None] == kp, row_k_r[None, :], w_r)
                w_i = tl.where(rows[:, None] == kp, row_k_i[None, :], w_i)
            xk_r = row_kp_r
            xk_i = row_kp_i
            sub_mask = (rows > k) & row_mask
            l_r = tl.load(
                ld_base + rows * ld_row_stride + k * ld_col_stride,
                mask=sub_mask,
                other=0.0,
            )
            l_i = tl.load(
                ld_base + rows * ld_row_stride + k * ld_col_stride + 1,
                mask=sub_mask,
                other=0.0,
            )
            prod_r = l_r[:, None] * xk_r[None, :] - l_i[:, None] * xk_i[None, :]
            prod_i = l_r[:, None] * xk_i[None, :] + l_i[:, None] * xk_r[None, :]
            upd_r = w_r - prod_r
            upd_i = w_i - prod_i
            d_r = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            if HERM:
                inv_d = 1.0 / d_r
                xk_r = xk_r * inv_d
                xk_i = xk_i * inv_d
            else:
                d_i = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride + 1)
                xk_r, xk_i = _cdiv(xk_r, xk_i, d_r, d_i)
            w_r = tl.where(rows[:, None] == k, xk_r[None, :], upd_r)
            w_i = tl.where(rows[:, None] == k, xk_i[None, :], upd_i)
            k += 1
        else:
            kp = -ip - 1
            row_k_r = _row_at(w_r, k)
            row_k_i = _row_at(w_i, k)
            row_k1_r = _row_at(w_r, k + 1)
            row_k1_i = _row_at(w_i, k + 1)
            row_kp_r = _row_at(w_r, kp)
            row_kp_i = _row_at(w_i, kp)
            if kp != k + 1:
                w_r = tl.where(rows[:, None] == k + 1, row_kp_r[None, :], w_r)
                w_i = tl.where(rows[:, None] == k + 1, row_kp_i[None, :], w_i)
                w_r = tl.where(rows[:, None] == kp, row_k1_r[None, :], w_r)
                w_i = tl.where(rows[:, None] == kp, row_k1_i[None, :], w_i)
            xk_r = row_k_r
            xk_i = row_k_i
            xk1_r = row_kp_r
            xk1_i = row_kp_i
            sub_mask = (rows > k + 1) & row_mask
            l0_r = tl.load(
                ld_base + rows * ld_row_stride + k * ld_col_stride,
                mask=sub_mask,
                other=0.0,
            )
            l0_i = tl.load(
                ld_base + rows * ld_row_stride + k * ld_col_stride + 1,
                mask=sub_mask,
                other=0.0,
            )
            l1_r = tl.load(
                ld_base + rows * ld_row_stride + (k + 1) * ld_col_stride,
                mask=sub_mask,
                other=0.0,
            )
            l1_i = tl.load(
                ld_base + rows * ld_row_stride + (k + 1) * ld_col_stride + 1,
                mask=sub_mask,
                other=0.0,
            )
            prod0_r = l0_r[:, None] * xk_r[None, :] - l0_i[:, None] * xk_i[None, :]
            prod0_i = l0_r[:, None] * xk_i[None, :] + l0_i[:, None] * xk_r[None, :]
            prod1_r = l1_r[:, None] * xk1_r[None, :] - l1_i[:, None] * xk1_i[None, :]
            prod1_i = l1_r[:, None] * xk1_i[None, :] + l1_i[:, None] * xk1_r[None, :]
            upd_r = w_r - prod0_r - prod1_r
            upd_i = w_i - prod0_i - prod1_i
            b_r = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride)
            b_i = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride + 1)
            a_r = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            a_i = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride + 1)
            c_r = tl.load(ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride)
            c_i = tl.load(
                ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride + 1
            )
            if HERM:
                akm1_r, akm1_i = _cdiv(a_r, a_i, b_r, -b_i)
                ak_r, ak_i = _cdiv(c_r, c_i, b_r, b_i)
                den_r, den_i = _cmul(akm1_r, akm1_i, ak_r, ak_i)
                den_r = den_r - 1
                bkm1_r, bkm1_i = _cdiv(xk_r, xk_i, b_r, -b_i)
                bk_r, bk_i = _cdiv(xk1_r, xk1_i, b_r, b_i)
                tmp_r, tmp_i = _cmul(ak_r, ak_i, bkm1_r, bkm1_i)
                xk_r, xk_i = _cdiv(tmp_r - bk_r, tmp_i - bk_i, den_r, den_i)
                tmp_r, tmp_i = _cmul(akm1_r, akm1_i, bk_r, bk_i)
                xk1_r, xk1_i = _cdiv(tmp_r - bkm1_r, tmp_i - bkm1_i, den_r, den_i)
            else:
                akm1_r, akm1_i = _cdiv(a_r, a_i, b_r, b_i)
                ak_r, ak_i = _cdiv(c_r, c_i, b_r, b_i)
                den_r, den_i = _cmul(akm1_r, akm1_i, ak_r, ak_i)
                den_r = den_r - 1
                bkm1_r, bkm1_i = _cdiv(xk_r, xk_i, b_r, b_i)
                bk_r, bk_i = _cdiv(xk1_r, xk1_i, b_r, b_i)
                tmp_r, tmp_i = _cmul(ak_r, ak_i, bkm1_r, bkm1_i)
                xk_r, xk_i = _cdiv(tmp_r - bk_r, tmp_i - bk_i, den_r, den_i)
                tmp_r, tmp_i = _cmul(akm1_r, akm1_i, bk_r, bk_i)
                xk1_r, xk1_i = _cdiv(tmp_r - bkm1_r, tmp_i - bkm1_i, den_r, den_i)
            w_r = tl.where(rows[:, None] == k, xk_r[None, :], upd_r)
            w_i = tl.where(rows[:, None] == k, xk_i[None, :], upd_i)
            w_r = tl.where(rows[:, None] == k + 1, xk1_r[None, :], w_r)
            w_i = tl.where(rows[:, None] == k + 1, xk1_i[None, :], w_i)
            k += 2

    k = n - 1
    while k >= 0:
        ip = _scalar_at(piv_vec, k)
        if ip > 0:
            kp = ip - 1
            row_k_r = _row_at(w_r, k)
            row_k_i = _row_at(w_i, k)
            sub_mask = (rows > k) & row_mask
            l_r = tl.load(
                ld_base + rows * ld_row_stride + k * ld_col_stride,
                mask=sub_mask,
                other=0.0,
            )
            l_i = tl.load(
                ld_base + rows * ld_row_stride + k * ld_col_stride + 1,
                mask=sub_mask,
                other=0.0,
            )
            if HERM:
                prod_r = l_r[:, None] * w_r - (-l_i[:, None]) * w_i
                prod_i = l_r[:, None] * w_i + (-l_i[:, None]) * w_r
                red_r = tl.sum(prod_r, 0)
                red_i = tl.sum(prod_i, 0)
            else:
                prod_r = l_r[:, None] * w_r - l_i[:, None] * w_i
                prod_i = l_r[:, None] * w_i + l_i[:, None] * w_r
                red_r = tl.sum(prod_r, 0)
                red_i = tl.sum(prod_i, 0)
            acc_r = row_k_r - red_r
            acc_i = row_k_i - red_i
            if kp != k:
                row_kp_r = _row_at(w_r, kp)
                row_kp_i = _row_at(w_i, kp)
                w_r = tl.where(rows[:, None] == kp, acc_r[None, :], w_r)
                w_i = tl.where(rows[:, None] == kp, acc_i[None, :], w_i)
                w_r = tl.where(rows[:, None] == k, row_kp_r[None, :], w_r)
                w_i = tl.where(rows[:, None] == k, row_kp_i[None, :], w_i)
            else:
                w_r = tl.where(rows[:, None] == k, acc_r[None, :], w_r)
                w_i = tl.where(rows[:, None] == k, acc_i[None, :], w_i)
            k -= 1
        else:
            row_km1_r = _row_at(w_r, k - 1)
            row_km1_i = _row_at(w_i, k - 1)
            row_k_r = _row_at(w_r, k)
            row_k_i = _row_at(w_i, k)
            sub_mask = (rows > k) & row_mask
            l0_r = tl.load(
                ld_base + rows * ld_row_stride + (k - 1) * ld_col_stride,
                mask=sub_mask,
                other=0.0,
            )
            l0_i = tl.load(
                ld_base + rows * ld_row_stride + (k - 1) * ld_col_stride + 1,
                mask=sub_mask,
                other=0.0,
            )
            l1_r = tl.load(
                ld_base + rows * ld_row_stride + k * ld_col_stride,
                mask=sub_mask,
                other=0.0,
            )
            l1_i = tl.load(
                ld_base + rows * ld_row_stride + k * ld_col_stride + 1,
                mask=sub_mask,
                other=0.0,
            )
            if HERM:
                prod0_r = l0_r[:, None] * w_r - (-l0_i[:, None]) * w_i
                prod0_i = l0_r[:, None] * w_i + (-l0_i[:, None]) * w_r
                prod1_r = l1_r[:, None] * w_r - (-l1_i[:, None]) * w_i
                prod1_i = l1_r[:, None] * w_i + (-l1_i[:, None]) * w_r
            else:
                prod0_r = l0_r[:, None] * w_r - l0_i[:, None] * w_i
                prod0_i = l0_r[:, None] * w_i + l0_i[:, None] * w_r
                prod1_r = l1_r[:, None] * w_r - l1_i[:, None] * w_i
                prod1_i = l1_r[:, None] * w_i + l1_i[:, None] * w_r
            red0_r = tl.sum(prod0_r, 0)
            red0_i = tl.sum(prod0_i, 0)
            red1_r = tl.sum(prod1_r, 0)
            red1_i = tl.sum(prod1_i, 0)
            acc_km1_r = row_km1_r - red0_r
            acc_km1_i = row_km1_i - red0_i
            acc_k_r = row_k_r - red1_r
            acc_k_i = row_k_i - red1_i
            kp = -ip - 1
            if kp != k:
                row_kp_r = _row_at(w_r, kp)
                row_kp_i = _row_at(w_i, kp)
                w_r = tl.where(rows[:, None] == kp, acc_k_r[None, :], w_r)
                w_i = tl.where(rows[:, None] == kp, acc_k_i[None, :], w_i)
                w_r = tl.where(rows[:, None] == k, row_kp_r[None, :], w_r)
                w_i = tl.where(rows[:, None] == k, row_kp_i[None, :], w_i)
            else:
                w_r = tl.where(rows[:, None] == k, acc_k_r[None, :], w_r)
                w_i = tl.where(rows[:, None] == k, acc_k_i[None, :], w_i)
            w_r = tl.where(rows[:, None] == k - 1, acc_km1_r[None, :], w_r)
            w_i = tl.where(rows[:, None] == k - 1, acc_km1_i[None, :], w_i)
            k -= 2

    tl.store(x_ptrs, w_r, mask=tile_mask)
    tl.store(x_ptrs + 1, w_i, mask=tile_mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "nrhs",
        "ld_batch_stride",
        "ld_row_stride",
        "ld_col_stride",
        "piv_batch_stride",
        "piv_row_stride",
        "x_batch_stride",
        "x_row_stride",
        "x_col_stride",
    ]
)
def linalg_ldl_solve_complex_kernel_global(
    LD,
    pivots,
    X,
    n,
    nrhs,
    ld_batch_stride,
    ld_row_stride,
    ld_col_stride,
    piv_batch_stride,
    piv_row_stride,
    x_batch_stride,
    x_row_stride,
    x_col_stride,
    HERM: tl.constexpr,
    BLOCK_NRH: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    col_start = block_idx * BLOCK_NRH
    cols = col_start + tl.arange(0, BLOCK_NRH)
    col_mask = cols < nrhs

    ld_base = LD + batch_idx * ld_batch_stride
    piv_base = pivots + batch_idx * piv_batch_stride
    x_base = X + batch_idx * x_batch_stride
    col_offsets = cols * x_col_stride

    k = 0
    while k < n:
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            kp = ip - 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k_ptr_i = row_k_ptr + 1
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                row_kp_ptr_i = row_kp_ptr + 1
                xk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
                xk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)
                xkp_r = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                xkp_i = tl.load(row_kp_ptr_i, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp_r, mask=col_mask)
                tl.store(row_k_ptr_i, xkp_i, mask=col_mask)
                tl.store(row_kp_ptr, xk_r, mask=col_mask)
                tl.store(row_kp_ptr_i, xk_i, mask=col_mask)
            xk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            xk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)

            i = k + 1
            while i < n:
                l_r = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l_i = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride + 1)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                row_i_ptr_i = row_i_ptr + 1
                xi_r = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xi_i = tl.load(row_i_ptr_i, mask=col_mask, other=0.0)
                prod_r, prod_i = _cmul(l_r, l_i, xk_r, xk_i)
                xi_r -= prod_r
                xi_i -= prod_i
                tl.store(row_i_ptr, xi_r, mask=col_mask)
                tl.store(row_i_ptr_i, xi_i, mask=col_mask)
                i += 1

            d_r = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            if HERM:
                inv_d = 1.0 / d_r
                xk_r = xk_r * inv_d
                xk_i = xk_i * inv_d
            else:
                d_i = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride + 1)
                xk_r, xk_i = _cdiv(xk_r, xk_i, d_r, d_i)
            tl.store(row_k_ptr, xk_r, mask=col_mask)
            tl.store(row_k_ptr_i, xk_i, mask=col_mask)
            k += 1
        else:
            kp = -ip - 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k_ptr_i = row_k_ptr + 1
            row_k1_ptr = x_base + (k + 1) * x_row_stride + col_offsets
            row_k1_ptr_i = row_k1_ptr + 1
            if kp != k + 1:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                row_kp_ptr_i = row_kp_ptr + 1
                xk1_r = tl.load(row_k1_ptr, mask=col_mask, other=0.0)
                xk1_i = tl.load(row_k1_ptr_i, mask=col_mask, other=0.0)
                xkp_r = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                xkp_i = tl.load(row_kp_ptr_i, mask=col_mask, other=0.0)
                tl.store(row_k1_ptr, xkp_r, mask=col_mask)
                tl.store(row_k1_ptr_i, xkp_i, mask=col_mask)
                tl.store(row_kp_ptr, xk1_r, mask=col_mask)
                tl.store(row_kp_ptr_i, xk1_i, mask=col_mask)
            xk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            xk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)
            xk1_r = tl.load(row_k1_ptr, mask=col_mask, other=0.0)
            xk1_i = tl.load(row_k1_ptr_i, mask=col_mask, other=0.0)

            i = k + 2
            while i < n:
                l0_r = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l0_i = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride + 1)
                l1_r = tl.load(ld_base + i * ld_row_stride + (k + 1) * ld_col_stride)
                l1_i = tl.load(
                    ld_base + i * ld_row_stride + (k + 1) * ld_col_stride + 1
                )
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                row_i_ptr_i = row_i_ptr + 1
                xi_r = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xi_i = tl.load(row_i_ptr_i, mask=col_mask, other=0.0)
                prod0_r, prod0_i = _cmul(l0_r, l0_i, xk_r, xk_i)
                prod1_r, prod1_i = _cmul(l1_r, l1_i, xk1_r, xk1_i)
                xi_r -= prod0_r + prod1_r
                xi_i -= prod0_i + prod1_i
                tl.store(row_i_ptr, xi_r, mask=col_mask)
                tl.store(row_i_ptr_i, xi_i, mask=col_mask)
                i += 1

            b_r = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride)
            b_i = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride + 1)
            a_r = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            a_i = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride + 1)
            c_r = tl.load(ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride)
            c_i = tl.load(
                ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride + 1
            )
            if HERM:
                akm1_r, akm1_i = _cdiv(a_r, a_i, b_r, -b_i)
                ak_r, ak_i = _cdiv(c_r, c_i, b_r, b_i)
                denom_r, denom_i = _cmul(akm1_r, akm1_i, ak_r, ak_i)
                denom_r -= 1
                bkm1_r, bkm1_i = _cdiv(xk_r, xk_i, b_r, -b_i)
                bk_r, bk_i = _cdiv(xk1_r, xk1_i, b_r, b_i)
                tmp_r, tmp_i = _cmul(ak_r, ak_i, bkm1_r, bkm1_i)
                xk_r, xk_i = _cdiv(tmp_r - bk_r, tmp_i - bk_i, denom_r, denom_i)
                tmp_r, tmp_i = _cmul(akm1_r, akm1_i, bk_r, bk_i)
                xk1_r, xk1_i = _cdiv(tmp_r - bkm1_r, tmp_i - bkm1_i, denom_r, denom_i)
            else:
                akm1_r, akm1_i = _cdiv(a_r, a_i, b_r, b_i)
                ak_r, ak_i = _cdiv(c_r, c_i, b_r, b_i)
                denom_r, denom_i = _cmul(akm1_r, akm1_i, ak_r, ak_i)
                denom_r -= 1
                bkm1_r, bkm1_i = _cdiv(xk_r, xk_i, b_r, b_i)
                bk_r, bk_i = _cdiv(xk1_r, xk1_i, b_r, b_i)
                tmp_r, tmp_i = _cmul(ak_r, ak_i, bkm1_r, bkm1_i)
                xk_r, xk_i = _cdiv(tmp_r - bk_r, tmp_i - bk_i, denom_r, denom_i)
                tmp_r, tmp_i = _cmul(akm1_r, akm1_i, bk_r, bk_i)
                xk1_r, xk1_i = _cdiv(tmp_r - bkm1_r, tmp_i - bkm1_i, denom_r, denom_i)
            tl.store(row_k_ptr, xk_r, mask=col_mask)
            tl.store(row_k_ptr_i, xk_i, mask=col_mask)
            tl.store(row_k1_ptr, xk1_r, mask=col_mask)
            tl.store(row_k1_ptr_i, xk1_i, mask=col_mask)
            k += 2

    k = n - 1
    while k >= 0:
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k_ptr_i = row_k_ptr + 1
            xk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            xk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)

            i = k + 1
            while i < n:
                l_r = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l_i = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride + 1)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                row_i_ptr_i = row_i_ptr + 1
                xi_r = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xi_i = tl.load(row_i_ptr_i, mask=col_mask, other=0.0)
                if HERM:
                    prod_r, prod_i = _cconj_mul(l_r, l_i, xi_r, xi_i)
                else:
                    prod_r, prod_i = _cmul(l_r, l_i, xi_r, xi_i)
                xk_r -= prod_r
                xk_i -= prod_i
                i += 1

            kp = ip - 1
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                row_kp_ptr_i = row_kp_ptr + 1
                xkp_r = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                xkp_i = tl.load(row_kp_ptr_i, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp_r, mask=col_mask)
                tl.store(row_k_ptr_i, xkp_i, mask=col_mask)
                tl.store(row_kp_ptr, xk_r, mask=col_mask)
                tl.store(row_kp_ptr_i, xk_i, mask=col_mask)
            else:
                tl.store(row_k_ptr, xk_r, mask=col_mask)
                tl.store(row_k_ptr_i, xk_i, mask=col_mask)
            k -= 1
        else:
            row_km1_ptr = x_base + (k - 1) * x_row_stride + col_offsets
            row_km1_ptr_i = row_km1_ptr + 1
            row_k_ptr = x_base + k * x_row_stride + col_offsets
            row_k_ptr_i = row_k_ptr + 1
            xkm1_r = tl.load(row_km1_ptr, mask=col_mask, other=0.0)
            xkm1_i = tl.load(row_km1_ptr_i, mask=col_mask, other=0.0)
            xk_r = tl.load(row_k_ptr, mask=col_mask, other=0.0)
            xk_i = tl.load(row_k_ptr_i, mask=col_mask, other=0.0)

            i = k + 1
            while i < n:
                l0_r = tl.load(ld_base + i * ld_row_stride + (k - 1) * ld_col_stride)
                l0_i = tl.load(
                    ld_base + i * ld_row_stride + (k - 1) * ld_col_stride + 1
                )
                l1_r = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l1_i = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride + 1)
                row_i_ptr = x_base + i * x_row_stride + col_offsets
                row_i_ptr_i = row_i_ptr + 1
                xi_r = tl.load(row_i_ptr, mask=col_mask, other=0.0)
                xi_i = tl.load(row_i_ptr_i, mask=col_mask, other=0.0)
                if HERM:
                    prod0_r, prod0_i = _cconj_mul(l0_r, l0_i, xi_r, xi_i)
                    prod1_r, prod1_i = _cconj_mul(l1_r, l1_i, xi_r, xi_i)
                else:
                    prod0_r, prod0_i = _cmul(l0_r, l0_i, xi_r, xi_i)
                    prod1_r, prod1_i = _cmul(l1_r, l1_i, xi_r, xi_i)
                xkm1_r -= prod0_r
                xkm1_i -= prod0_i
                xk_r -= prod1_r
                xk_i -= prod1_i
                i += 1

            kp = -ip - 1
            if kp != k:
                row_kp_ptr = x_base + kp * x_row_stride + col_offsets
                row_kp_ptr_i = row_kp_ptr + 1
                xkp_r = tl.load(row_kp_ptr, mask=col_mask, other=0.0)
                xkp_i = tl.load(row_kp_ptr_i, mask=col_mask, other=0.0)
                tl.store(row_k_ptr, xkp_r, mask=col_mask)
                tl.store(row_k_ptr_i, xkp_i, mask=col_mask)
                tl.store(row_kp_ptr, xk_r, mask=col_mask)
                tl.store(row_kp_ptr_i, xk_i, mask=col_mask)
            else:
                tl.store(row_k_ptr, xk_r, mask=col_mask)
                tl.store(row_k_ptr_i, xk_i, mask=col_mask)
            tl.store(row_km1_ptr, xkm1_r, mask=col_mask)
            tl.store(row_km1_ptr_i, xkm1_i, mask=col_mask)
            k -= 2


def _validate_inputs(LD, pivots, B):
    if LD.device != B.device or LD.device != pivots.device:
        raise ValueError("LD, pivots, and B must be on the same device")
    if LD.dtype != B.dtype:
        raise TypeError("LD and B must have the same dtype")
    if LD.ndim < 2 or B.ndim < 2:
        raise ValueError("LD and B must be at least 2D")
    if LD.shape[-1] != LD.shape[-2]:
        raise ValueError("LD must be a square matrix or a batch of square matrices")
    if B.shape[-2] != LD.shape[-1]:
        raise ValueError("B must have shape (*, n, k) with the same n as LD")
    if pivots.shape != LD.shape[:-1]:
        raise ValueError("pivots must have shape (*, n) matching LD")
    if LD.shape[:-2] != B.shape[:-2]:
        raise ValueError("LD, pivots, and B must share the same batch dimensions")


def linalg_ldl_solve(LD, pivots, B, *, hermitian=False):
    """
    Solve a linear system using the compact LDL factorization produced by
    torch.linalg.ldl_factor_ex.
    """
    logger.debug("GEMS_ENFLAME LINALG_LDL_SOLVE")
    _validate_inputs(LD, pivots, B)

    if LD.dtype not in REAL_DTYPES + COMPLEX_DTYPES:
        raise TypeError(
            "linalg_ldl_solve supports only float32, float64, complex64, and complex128 inputs"
        )
    if LD.dtype in WIDE_DTYPES and not runtime_device.support_fp64:
        # No f64 arithmetic on this device (support_fp64 == False): compiling the
        # kernels with f64 pointers crashes the GCU400 Triton compiler in-process
        # (make_gcuir SIGSEGV, compiler_problems P04 f64 family). Native aten
        # already computes Double as FP32 on this backend ("Double use Float
        # replace"), so mirror that: solve value-level in the narrow dtype and
        # widen the outputs back (linalg_lu_solve precedent).
        narrow_dtype = NARROW_OF[LD.dtype]
        X_narrow = linalg_ldl_solve(
            LD.to(narrow_dtype),
            pivots,
            B.to(narrow_dtype),
            hermitian=hermitian,
        )
        return X_narrow.to(LD.dtype)
    if LD.numel() == 0 or B.numel() == 0:
        return B.clone()

    batch = math.prod(LD.shape[:-2])
    n = LD.shape[-1]
    nrhs = B.shape[-1]

    with torch.no_grad():
        if LD.dtype in REAL_DTYPES:
            LD_work = LD.reshape(batch, n, n).contiguous()
            piv_work = pivots.reshape(batch, n).contiguous()
            X = B.reshape(batch, n, nrhs).clone()
            if n <= LDL_SOLVE_REG_MAX_N:
                block_n = max(triton.next_power_of_2(n), 2)
                block_rhs = max(
                    min(triton.next_power_of_2(nrhs), LDL_SOLVE_RHS_BLOCK), 1
                )
                grid = (batch, triton.cdiv(nrhs, block_rhs))
                linalg_ldl_solve_real_kernel[grid](
                    LD_work,
                    piv_work,
                    X,
                    n,
                    nrhs,
                    LD_work.stride(0),
                    LD_work.stride(1),
                    LD_work.stride(2),
                    piv_work.stride(0),
                    piv_work.stride(1),
                    X.stride(0),
                    X.stride(1),
                    X.stride(2),
                    BLOCK_N=block_n,
                    BLOCK_NRH=block_rhs,
                    num_warps=1,
                    num_stages=2,
                )
            else:
                grid = (batch, triton.cdiv(nrhs, LDL_SOLVE_RHS_BLOCK))
                linalg_ldl_solve_real_kernel_global[grid](
                    LD_work,
                    piv_work,
                    X,
                    n,
                    nrhs,
                    LD_work.stride(0),
                    LD_work.stride(1),
                    LD_work.stride(2),
                    piv_work.stride(0),
                    piv_work.stride(1),
                    X.stride(0),
                    X.stride(1),
                    X.stride(2),
                    BLOCK_NRH=LDL_SOLVE_RHS_BLOCK,
                    num_warps=4,
                )
            return X.reshape(B.shape)

        LD_work = torch.view_as_real(LD.reshape(batch, n, n).contiguous())
        piv_work = pivots.reshape(batch, n).contiguous()
        X = B.reshape(batch, n, nrhs).clone()
        X_work = torch.view_as_real(X)
        if n <= LDL_SOLVE_REG_MAX_N:
            block_n = max(triton.next_power_of_2(n), 2)
            rhs_cap = (
                LDL_SOLVE_CPLX_RHS_BLOCK if block_n >= 128 else LDL_SOLVE_RHS_BLOCK
            )
            block_rhs = max(min(triton.next_power_of_2(nrhs), rhs_cap), 1)
            grid = (batch, triton.cdiv(nrhs, block_rhs))
            linalg_ldl_solve_complex_kernel[grid](
                LD_work,
                piv_work,
                X_work,
                n,
                nrhs,
                LD_work.stride(0),
                LD_work.stride(1),
                LD_work.stride(2),
                piv_work.stride(0),
                piv_work.stride(1),
                X_work.stride(0),
                X_work.stride(1),
                X_work.stride(2),
                HERM=hermitian,
                BLOCK_N=block_n,
                BLOCK_NRH=block_rhs,
                num_warps=1,
                num_stages=2,
            )
        else:
            grid = (batch, triton.cdiv(nrhs, LDL_SOLVE_RHS_BLOCK))
            linalg_ldl_solve_complex_kernel_global[grid](
                LD_work,
                piv_work,
                X_work,
                n,
                nrhs,
                LD_work.stride(0),
                LD_work.stride(1),
                LD_work.stride(2),
                piv_work.stride(0),
                piv_work.stride(1),
                X_work.stride(0),
                X_work.stride(1),
                X_work.stride(2),
                HERM=hermitian,
                BLOCK_NRH=LDL_SOLVE_RHS_BLOCK,
                num_warps=4,
            )
        return X.reshape(B.shape)
