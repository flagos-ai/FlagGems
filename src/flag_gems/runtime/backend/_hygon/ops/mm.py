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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    pid = ext.program_id(0)

    # --------------------------
    # match naming: num_pid_m, num_pid_n
    # --------------------------
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # reorder for L2
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    group_size_m = min(num_pid_m - group_id * GROUP_M, GROUP_M)

    pid_m = group_id * GROUP_M + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --------------------------
    # match naming: offs_am, offs_bn, offs_k
    # --------------------------
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # contiguous aligned offsets (ram/rbn → offs_am/offs_bn)
    offs_am_cont = tl.max_contiguous(tl.multiple_of(offs_am % M, BLOCK_M), BLOCK_M)
    offs_bn_cont = tl.max_contiguous(tl.multiple_of(offs_bn % N, BLOCK_N), BLOCK_N)

    # previous K multiple
    # prev_k_mult = prev_multiple_of(K, BLOCK_K)
    prev_k_mult = tl.cdiv(K, BLOCK_K) * BLOCK_K - BLOCK_K

    # accumulator
    if IS_FP64:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # --------------------------
    # main K loop
    # --------------------------
    for start_k in range(0, prev_k_mult, BLOCK_K):
        rk = start_k + offs_k

        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak)
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn)
        )

        if a.dtype != b.dtype:
            a = a.to(c_ptr.dtype.element_ty)
            b = b.to(c_ptr.dtype.element_ty)

        if IS_FP64:
            accumulator += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)
        else:
            accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # --------------------------
    # loop peel
    # --------------------------
    rk = prev_k_mult + offs_k
    mask_k = rk < K

    if IS_FP64:
        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
            mask=mask_k[:, None],
            other=0.0,
        )
    else:
        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
            mask=mask_k[:, None],
        )

    if a.dtype != b.dtype:
        a = a.to(c_ptr.dtype.element_ty)
        b = b.to(c_ptr.dtype.element_ty)

    if IS_FP64:
        accumulator += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)
    else:
        accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # cast to output dtype
    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # --------------------------
    # rematerialize offsets for store
    # (match naming: offs_cm, offs_cn)
    # --------------------------
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    c_ptr = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask_store = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]

    tl.store(c_ptr, accumulator, mask=mask_store)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def _mm_default(a, b):
    logger.debug("GEMS_HYGON MM")
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            GROUP_M=8,
            num_stages=1,
            num_warps=4,
            num_ldmatrixes=0,
            IS_FP64=a.dtype == torch.float64,
        )
    return c


def mm(a, b):
    """Hygon GEMM with native-stride kernels and selected BF16 launches."""
    assert a.ndim == 2 and b.ndim == 2, "mm expects matrices"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    if a.dtype != b.dtype or a.dtype not in (
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ):
        return _mm_default(a, b)
    m, k = a.shape
    n = b.shape[1]
    if m == 0 or n == 0:
        return torch.empty((m, n), device=a.device, dtype=a.dtype)
    from .mm_config import select_config
    from .mm_kernels import (
        reduction,
        simt,
        small_k,
        split_reduce,
        tiled,
        tiled_fast,
        tiled_unrolled,
        transpose,
    )

    cfg = select_config(a, b)
    if cfg is None:
        return _mm_default(a, b)
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    kind = cfg["kind"]
    with torch_device_fn.device(a.device):
        common = (a, b, c, m, n, k, *a.stride(), *b.stride())
        if kind == "outer":
            if k != 1:
                return _mm_default(a, b)
            from .mm_outer_kernels import outer_product

            rows, cols = cfg["rows"], cfg["cols"]
            outer_product[(triton.cdiv(m, rows), triton.cdiv(n, cols))](
                a,
                b,
                c,
                m,
                n,
                a.stride(0),
                b.stride(1),
                ROWS=rows,
                COLS=cols,
                STORE=cfg.get("store", ""),
                num_warps=cfg["warps"],
            )
        elif kind == "small":
            block = cfg["block"]
            small_k[(triton.cdiv(m * n, block),)](
                *common, BLOCK=block, num_warps=cfg["warps"], num_stages=1
            )
        elif kind == "reduce":
            bo = cfg["bo"]
            reduction[(triton.cdiv(m * n, bo),)](
                *common,
                BO=bo,
                RK=triton.next_power_of_2(k),
                num_warps=cfg["warps"],
                num_stages=1,
            )
        elif kind == "cache3":
            from .mm_cache_kernels import tiled_cached

            bm, bn, bk = cfg["bm"], cfg["bn"], cfg["bk"]
            swap = cfg.get("swap", False)
            if swap:
                aa, bb = b, a
                gm, gn = n, m
                strides = (b.stride(1), b.stride(0), a.stride(1), a.stride(0))
            else:
                aa, bb = a, b
                gm, gn = m, n
                strides = (*a.stride(), *b.stride())
            tiled_cached[(triton.cdiv(gm, bm), triton.cdiv(gn, bn))](
                aa,
                bb,
                c,
                gm,
                gn,
                k,
                *strides,
                BM=bm,
                BN=bn,
                BK=bk,
                UNROLL=cfg["unroll"],
                TAIL=cfg["tail"],
                TRANS=swap,
                num_warps=cfg["warps"],
                num_stages=cfg["stages"],
                async_copy_use_single_buffer=False,
                LOAD=cfg.get("load", ""),
                STORE=cfg.get("store", ""),
            )
        elif kind in ("fast3", "swap32"):
            bm, bn, bk = cfg["bm"], cfg["bn"], cfg["bk"]
            swap = cfg.get("swap", False)
            if swap:
                aa, bb = b, a
                gm, gn = n, m
                strides = (b.stride(1), b.stride(0), a.stride(1), a.stride(0))
            else:
                aa, bb = a, b
                gm, gn = m, n
                strides = (*a.stride(), *b.stride())
            tiled_unrolled[(triton.cdiv(gm, bm), triton.cdiv(gn, bn))](
                aa,
                bb,
                c,
                gm,
                gn,
                k,
                *strides,
                BM=bm,
                BN=bn,
                BK=bk,
                UNROLL=cfg["unroll"],
                TAIL=cfg["tail"],
                TRANS=swap,
                num_warps=cfg["warps"],
                num_stages=cfg["stages"],
                async_copy_use_single_buffer=False,
                **cfg.get("launch_options", {}),
            )
        elif kind == "fasttile":
            bm, bn, bk = cfg["bm"], cfg["bn"], cfg["bk"]
            tiled_fast[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
                *common,
                BM=bm,
                BN=bn,
                BK=bk,
                GROUP=cfg["group"],
                num_warps=cfg["warps"],
                num_stages=cfg["stages"],
                async_copy_use_single_buffer=cfg.get("single_buffer", True),
                kpack=cfg.get("kpack", 1),
                waves_per_eu=cfg.get("waves_per_eu", 0),
            )
        elif kind == "simt":
            bm, bn = cfg["bm"], cfg["bn"]
            simt[(triton.cdiv(m, bm), triton.cdiv(n, bn), 1)](
                *common,
                BM=bm,
                BN=bn,
                SPLIT=1,
                UNROLL=cfg["unroll"],
                num_warps=cfg["warps"],
                num_stages=1,
            )
        elif kind == "fast_split":
            from .mm_fast_split_kernels import tiled_fast_split

            bm, bn, bk = cfg["bm"], cfg["bn"], cfg["bk"]
            split = cfg["split"]
            swap = cfg.get("swap", False)
            if swap:
                aa, bb = b, a
                gm, gn = n, m
                strides = (b.stride(1), b.stride(0), a.stride(1), a.stride(0))
            else:
                aa, bb = a, b
                gm, gn = m, n
                strides = (*a.stride(), *b.stride())
            partial = torch.empty((split, m, n), device=a.device, dtype=a.dtype)
            tiled_fast_split[(triton.cdiv(gm, bm) * triton.cdiv(gn, bn), split)](
                aa,
                bb,
                partial,
                gm,
                gn,
                k,
                *strides,
                BM=bm,
                BN=bn,
                BK=bk,
                GROUP=cfg.get("group", 8),
                SPLIT=split,
                TRANS=swap,
                num_warps=cfg["warps"],
                num_stages=cfg["stages"],
                async_copy_use_single_buffer=False,
                **cfg.get("launch_options", {}),
            )
            split_reduce[(triton.cdiv(m * n, 1024),)](
                partial, c, m * n, split, 1024, num_warps=4
            )
        elif kind == "swizzle_split":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return _mm_default(a, b)
            from .mm_split_kernels import swizzle_split

            tm, tn, wm, warps = cfg["tm"], cfg["tn"], cfg["wm"], cfg["warps"]
            split = cfg["split"]
            partial = torch.empty((split, m, n), device=a.device, dtype=a.dtype)
            swizzle_split[
                (
                    triton.cdiv(m, 16 * tm * wm),
                    triton.cdiv(n, 16 * tn * (warps // wm)),
                    split,
                )
            ](
                a,
                b,
                partial,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                WARPS=warps,
                WM=wm,
                TM=tm,
                TN=tn,
                PREFETCH=cfg["prefetch"],
                SPLIT=split,
                num_warps=warps,
                num_stages=1,
            )
            split_reduce[(triton.cdiv(m * n, 256),)](
                partial, c, m * n, split, 256, num_warps=4
            )
        elif kind == "bank_tail":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return _mm_default(a, b)
            from .mm_bank_tail_kernels import mmac_banked_tail

            tm, tn, wm, warps = cfg["tm"], cfg["tn"], cfg["wm"], cfg["warps"]
            swap = cfg.get("swap", False)
            if swap:
                aa, bb = b, a
                gm, gn = n, m
                strides = (b.stride(1), b.stride(0), a.stride(1), a.stride(0))
            else:
                aa, bb = a, b
                gm, gn = m, n
                strides = (*a.stride(), *b.stride())
            mmac_banked_tail[
                (
                    triton.cdiv(gm, 16 * tm * wm),
                    triton.cdiv(gn, 16 * tn * (warps // wm)),
                )
            ](
                aa,
                bb,
                c,
                gm,
                gn,
                k,
                *strides,
                WARPS=warps,
                WM=wm,
                TM=tm,
                TN=tn,
                PREFETCH=cfg["prefetch"],
                GROUP=cfg["group"],
                PIPE=cfg["pipe"],
                SWIZZLE=True,
                TRANS=swap,
                SA_VEC=cfg["sa_vec"],
                SA_MAX=cfg["sa_max"],
                SB_VEC=cfg["sb_vec"],
                SB_MAX=cfg["sb_max"],
                SA_PER=cfg["sa_per"],
                SB_PER=cfg["sb_per"],
                num_warps=warps,
                num_stages=1,
                **cfg.get("launch_options", {}),
            )
        elif kind == "banked":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return _mm_default(a, b)
            from .mm_bank_kernels import mmac_banked

            tm, tn, wm, warps = cfg["tm"], cfg["tn"], cfg["wm"], cfg["warps"]
            swap = cfg.get("swap", False)
            if swap:
                aa, bb = b, a
                gm, gn = n, m
                strides = (b.stride(1), b.stride(0), a.stride(1), a.stride(0))
            else:
                aa, bb = a, b
                gm, gn = m, n
                strides = (*a.stride(), *b.stride())
            mmac_banked[
                (
                    triton.cdiv(gm, 16 * tm * wm),
                    triton.cdiv(gn, 16 * tn * (warps // wm)),
                )
            ](
                aa,
                bb,
                c,
                gm,
                gn,
                k,
                *strides,
                WARPS=warps,
                WM=wm,
                TM=tm,
                TN=tn,
                PREFETCH=cfg["prefetch"],
                GROUP=cfg["group"],
                PIPE=cfg["pipe"],
                SWIZZLE=True,
                TRANS=swap,
                SA_VEC=cfg["sa_vec"],
                SA_MAX=cfg["sa_max"],
                SB_VEC=cfg["sb_vec"],
                SB_MAX=cfg["sb_max"],
                num_warps=warps,
                num_stages=1,
                **cfg.get("launch_options", {}),
            )
        elif kind == "native_fp32":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return _mm_default(a, b)
            from .mm_native_fp32_kernels import mmac_tiled

            mmac_tiled[(triton.cdiv(n, 128), triton.cdiv(m, 64))](
                b,
                a,
                c,
                n,
                m,
                k,
                b.stride(1),
                b.stride(0),
                a.stride(1),
                a.stride(0),
                WARPS=4,
                WM=4,
                TM=2,
                TN=4,
                PREFETCH=2,
                GROUP=cfg["group"],
                PIPE=True,
                SWIZZLE=False,
                TRANS=True,
                num_warps=4,
                num_stages=1,
            )
        elif kind == "swizzle":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return _mm_default(a, b)
            from .mm_mmac_swizzle import mmac_tiled

            tm, tn, wm, warps = cfg["tm"], cfg["tn"], cfg["wm"], cfg["warps"]
            swap = cfg.get("swap", False)
            if swap:
                aa, bb = b, a
                gm, gn = n, m
                strides = (b.stride(1), b.stride(0), a.stride(1), a.stride(0))
            else:
                aa, bb = a, b
                gm, gn = m, n
                strides = (*a.stride(), *b.stride())
            mmac_tiled[
                (
                    triton.cdiv(gm, 16 * tm * wm),
                    triton.cdiv(gn, 16 * tn * (warps // wm)),
                )
            ](
                aa,
                bb,
                c,
                gm,
                gn,
                k,
                *strides,
                WARPS=warps,
                WM=wm,
                TM=tm,
                TN=tn,
                PREFETCH=cfg["prefetch"],
                GROUP=cfg["group"],
                PIPE=cfg["pipe"],
                SWIZZLE=True,
                TRANS=swap,
                num_warps=warps,
                num_stages=1,
            )
        elif kind in (
            "peel",
            "mmac",
            "prefetch",
            "coalesced",
            "reshape",
            "pipeline",
            "coopfast",
            "rawpipe",
        ):
            if kind == "rawpipe":
                from .mm_mmac_rawpipe import mmac_tiled
            elif kind == "peel":
                from .mm_mmac_peel import mmac_tiled
            elif kind == "coopfast":
                from .mm_mmac_coopfast import mmac_tiled
            elif kind == "pipeline":
                from .mm_mmac_pipeline import mmac_tiled
            elif kind == "coalesced":
                from .mm_mmac_coalesced import mmac_tiled
            elif kind == "reshape":
                from .mm_mmac_reshape import mmac_tiled
            elif kind == "prefetch":
                from .mm_mmac_prefetch import mmac_tiled
            else:
                from .mm_mmac_operands import mmac_tiled

            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return _mm_default(a, b)
            tm, tn, wm, warps = cfg["tm"], cfg["tn"], cfg["wm"], cfg["warps"]
            split = cfg.get("split", 1)
            partial = (
                c
                if split == 1
                else torch.empty((split, m, n), device=a.device, dtype=a.dtype)
            )
            if cfg.get("trans", False):
                aa = torch.empty((k, m), device=a.device, dtype=a.dtype)
                transpose[(triton.cdiv(m, 32), triton.cdiv(k, 32))](
                    a, aa, m, k, *a.stride(), 32, num_warps=4
                )
                am, ak = 1, m
            else:
                aa = a
                am, ak = a.stride()
            mmac_tiled[
                (
                    triton.cdiv(m, 16 * tm * wm),
                    triton.cdiv(n, 16 * tn * (warps // wm)),
                    split,
                )
            ](
                aa,
                b,
                partial,
                m,
                n,
                k,
                am,
                ak,
                *b.stride(),
                WARPS=warps,
                WM=wm,
                TM=tm,
                TN=tn,
                SPLIT=split,
                **(
                    {"PREFETCH": cfg["prefetch"]}
                    if kind != "mmac"
                    else {"UNROLL": cfg["unroll"]}
                ),
                **(
                    {"GROUP": cfg["group"]}
                    if kind
                    in (
                        "coalesced",
                        "reshape",
                        "pipeline",
                        "coopfast",
                        "peel",
                        "rawpipe",
                    )
                    else {}
                ),
                num_warps=warps,
                num_stages=1,
            )
            if split > 1:
                split_reduce[(triton.cdiv(m * n, 256),)](
                    partial, c, m * n, split, 256, num_warps=4
                )
        else:
            bm, bn, bk = cfg["bm"], cfg["bn"], cfg["bk"]
            split = cfg.get("split", 1)
            partial = (
                c
                if split == 1
                else torch.empty((split, m, n), device=a.device, dtype=a.dtype)
            )
            tiled[(triton.cdiv(m, bm) * triton.cdiv(n, bn), split)](
                a,
                b,
                partial,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                BM=bm,
                BN=bn,
                BK=bk,
                GROUP=cfg.get("group", 8),
                SPLIT=split,
                PREC="ieee",
                num_warps=cfg["warps"],
                num_stages=cfg["stages"],
                async_copy_use_single_buffer=cfg.get("single_buffer", True),
            )
            if split > 1:
                split_reduce[(triton.cdiv(m * n, 256),)](
                    partial, c, m * n, split, 256, num_warps=4
                )
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS_HYGON MM_OUT")
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = out
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            GROUP_M=8,
            num_stages=1,
            num_warps=4,
            num_ldmatrixes=0,
            IS_FP64=a.dtype == torch.float64,
        )
    return c
