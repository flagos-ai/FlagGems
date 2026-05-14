import logging
import math
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.triton_version_utils import HAS_TLE

logger = logging.getLogger(__name__)

if HAS_TLE:
    import triton.experimental.tle.language as tle
else:
    tle = None

PI = math.pi
_FFT_REG_THRESHOLD = 256

_BITREV_CACHE: dict[Tuple[int, torch.device], torch.Tensor] = {}
_TWIDDLE_CACHE: dict[Tuple[int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _log2(n: int) -> int:
    return n.bit_length() - 1


def _bitrev_indices(n: int, device: torch.device) -> torch.Tensor:
    key = (n, device)
    cached = _BITREV_CACHE.get(key)
    if cached is not None:
        return cached
    log_n = _log2(n)
    idx = torch.arange(n, device=device, dtype=torch.int32)
    rev = torch.zeros_like(idx)
    tmp = idx.clone()
    for _ in range(log_n):
        rev = (rev << 1) | (tmp & 1)
        tmp = tmp >> 1
    _BITREV_CACHE[key] = rev
    return rev


def _twiddle_tables(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (n, device)
    cached = _TWIDDLE_CACHE.get(key)
    if cached is not None:
        return cached
    log_n = _log2(n)
    tw_real = torch.empty((n - 1,), device=device, dtype=torch.float32)
    tw_imag = torch.empty((n - 1,), device=device, dtype=torch.float32)
    offset = 0
    for stage in range(log_n):
        m = 1 << (stage + 1)
        half = m >> 1
        j = torch.arange(half, device=device, dtype=torch.float32)
        angle = (-2.0 * PI / m) * j
        tw_real[offset : offset + half] = torch.cos(angle)
        tw_imag[offset : offset + half] = torch.sin(angle)
        offset += half
    _TWIDDLE_CACHE[key] = (tw_real, tw_imag)
    return tw_real, tw_imag


def _prepare_input(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.is_complex():
        if x.dtype not in (torch.complex64, torch.complex128):
            raise ValueError(f"unsupported complex dtype: {x.dtype}")
        x = x.to(torch.complex64)
        real = x.real.contiguous()
        imag = x.imag.contiguous()
    else:
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError(f"unsupported dtype: {x.dtype}")
        x = x.to(torch.float32)
        real = x.contiguous()
        imag = torch.zeros_like(real)
    return real, imag


@triton.jit
def fft_kernel_triton(
    in_real,
    in_imag,
    bitrev,
    twiddle_real,
    twiddle_imag,
    buf0_real,
    buf0_imag,
    buf1_real,
    buf1_imag,
    stride_in,
    stride_buf,
    n_rows,
    N: tl.constexpr,
    LOG_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    offs = tl.arange(0, N)
    row_valid = row < n_rows
    mask = row_valid & (offs < N)

    rev = tl.load(bitrev + offs, mask=offs < N, other=0)
    in_real_ptrs = in_real + row * stride_in + rev
    in_imag_ptrs = in_imag + row * stride_in + rev
    vals_real = tl.load(in_real_ptrs, mask=mask, other=0.0)
    vals_imag = tl.load(in_imag_ptrs, mask=mask, other=0.0)

    buf0_real_ptrs = buf0_real + row * stride_buf + offs
    buf0_imag_ptrs = buf0_imag + row * stride_buf + offs
    tl.store(buf0_real_ptrs, vals_real, mask=mask)
    tl.store(buf0_imag_ptrs, vals_imag, mask=mask)

    buf_a_real = buf0_real
    buf_a_imag = buf0_imag
    buf_b_real = buf1_real
    buf_b_imag = buf1_imag

    if LOG_N % 2 == 1:
        m = 2
        half = 1
        idx = offs
        pos = idx & (m - 1)
        j = pos & (half - 1)
        base = idx - pos
        even_idx = base + j
        odd_idx = even_idx + half

        even_ptrs_real = buf_a_real + row * stride_buf + even_idx
        even_ptrs_imag = buf_a_imag + row * stride_buf + even_idx
        odd_ptrs_real = buf_a_real + row * stride_buf + odd_idx
        odd_ptrs_imag = buf_a_imag + row * stride_buf + odd_idx

        u_real = tl.load(even_ptrs_real, mask=mask, other=0.0)
        u_imag = tl.load(even_ptrs_imag, mask=mask, other=0.0)
        v_real = tl.load(odd_ptrs_real, mask=mask, other=0.0)
        v_imag = tl.load(odd_ptrs_imag, mask=mask, other=0.0)

        base_tw = 0
        tw_idx = base_tw + j
        tw_real = tl.load(twiddle_real + tw_idx, mask=mask, other=1.0)
        tw_imag = tl.load(twiddle_imag + tw_idx, mask=mask, other=0.0)

        v_tw_real = v_real * tw_real - v_imag * tw_imag
        v_tw_imag = v_real * tw_imag + v_imag * tw_real

        add_mask = pos < half
        out_real = tl.where(add_mask, u_real + v_tw_real, u_real - v_tw_real)
        out_imag = tl.where(add_mask, u_imag + v_tw_imag, u_imag - v_tw_imag)

        out_ptrs_real = buf_b_real + row * stride_buf + idx
        out_ptrs_imag = buf_b_imag + row * stride_buf + idx
        tl.store(out_ptrs_real, out_real, mask=mask)
        tl.store(out_ptrs_imag, out_imag, mask=mask)
        tl.debug_barrier()

        buf_a_real, buf_b_real = buf_b_real, buf_a_real
        buf_a_imag, buf_b_imag = buf_b_imag, buf_a_imag

    if LOG_N % 2 == 1:
        for r4 in tl.static_range((LOG_N - 1) // 2):
            stage_s = 2 + r4 * 2
            m = 1 << (stage_s + 1)
            quarter = m >> 2
            half = m >> 1
            three_quarter = quarter + half

            idx = offs
            pos = idx & (m - 1)
            j = pos & (quarter - 1)
            base = idx - pos
            i0 = base + j
            i1 = i0 + quarter
            i2 = i1 + quarter
            i3 = i2 + quarter

            ptr0_real = buf_a_real + row * stride_buf + i0
            ptr0_imag = buf_a_imag + row * stride_buf + i0
            ptr1_real = buf_a_real + row * stride_buf + i1
            ptr1_imag = buf_a_imag + row * stride_buf + i1
            ptr2_real = buf_a_real + row * stride_buf + i2
            ptr2_imag = buf_a_imag + row * stride_buf + i2
            ptr3_real = buf_a_real + row * stride_buf + i3
            ptr3_imag = buf_a_imag + row * stride_buf + i3

            x0_real = tl.load(ptr0_real, mask=mask, other=0.0)
            x0_imag = tl.load(ptr0_imag, mask=mask, other=0.0)
            x1_real = tl.load(ptr1_real, mask=mask, other=0.0)
            x1_imag = tl.load(ptr1_imag, mask=mask, other=0.0)
            x2_real = tl.load(ptr2_real, mask=mask, other=0.0)
            x2_imag = tl.load(ptr2_imag, mask=mask, other=0.0)
            x3_real = tl.load(ptr3_real, mask=mask, other=0.0)
            x3_imag = tl.load(ptr3_imag, mask=mask, other=0.0)

            base_tw1 = (1 << (stage_s - 1)) - 1
            base_tw2 = (1 << stage_s) - 1
            tw1_idx = base_tw1 + j
            tw2_idx = base_tw2 + j
            tw1_real = tl.load(twiddle_real + tw1_idx, mask=mask, other=1.0)
            tw1_imag = tl.load(twiddle_imag + tw1_idx, mask=mask, other=0.0)
            tw2_real = tl.load(twiddle_real + tw2_idx, mask=mask, other=1.0)
            tw2_imag = tl.load(twiddle_imag + tw2_idx, mask=mask, other=0.0)

            t1_real = x1_real * tw1_real - x1_imag * tw1_imag
            t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
            t3_real = x3_real * tw1_real - x3_imag * tw1_imag
            t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

            u0_real = x0_real + t1_real
            u0_imag = x0_imag + t1_imag
            u1_real = x0_real - t1_real
            u1_imag = x0_imag - t1_imag
            v0_real = x2_real + t3_real
            v0_imag = x2_imag + t3_imag
            v1_real = x2_real - t3_real
            v1_imag = x2_imag - t3_imag

            v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
            v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
            w3_real = tw2_imag
            w3_imag = -tw2_real
            v1_tw_real = v1_real * w3_real - v1_imag * w3_imag
            v1_tw_imag = v1_real * w3_imag + v1_imag * w3_real

            o0_real = u0_real + v0_tw_real
            o0_imag = u0_imag + v0_tw_imag
            o2_real = u0_real - v0_tw_real
            o2_imag = u0_imag - v0_tw_imag
            o1_real = u1_real + v1_tw_real
            o1_imag = u1_imag + v1_tw_imag
            o3_real = u1_real - v1_tw_real
            o3_imag = u1_imag - v1_tw_imag

            m0 = pos < quarter
            m1 = (pos >= quarter) & (pos < half)
            m2 = (pos >= half) & (pos < three_quarter)
            out_real = tl.where(
                m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real))
            )
            out_imag = tl.where(
                m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag))
            )

            out_ptrs_real = buf_b_real + row * stride_buf + idx
            out_ptrs_imag = buf_b_imag + row * stride_buf + idx
            tl.store(out_ptrs_real, out_real, mask=mask)
            tl.store(out_ptrs_imag, out_imag, mask=mask)
            tl.debug_barrier()

            buf_a_real, buf_b_real = buf_b_real, buf_a_real
            buf_a_imag, buf_b_imag = buf_b_imag, buf_a_imag
    else:
        for r4 in tl.static_range(LOG_N // 2):
            stage_s = 1 + r4 * 2
            m = 1 << (stage_s + 1)
            quarter = m >> 2
            half = m >> 1
            three_quarter = quarter + half

            idx = offs
            pos = idx & (m - 1)
            j = pos & (quarter - 1)
            base = idx - pos
            i0 = base + j
            i1 = i0 + quarter
            i2 = i1 + quarter
            i3 = i2 + quarter

            ptr0_real = buf_a_real + row * stride_buf + i0
            ptr0_imag = buf_a_imag + row * stride_buf + i0
            ptr1_real = buf_a_real + row * stride_buf + i1
            ptr1_imag = buf_a_imag + row * stride_buf + i1
            ptr2_real = buf_a_real + row * stride_buf + i2
            ptr2_imag = buf_a_imag + row * stride_buf + i2
            ptr3_real = buf_a_real + row * stride_buf + i3
            ptr3_imag = buf_a_imag + row * stride_buf + i3

            x0_real = tl.load(ptr0_real, mask=mask, other=0.0)
            x0_imag = tl.load(ptr0_imag, mask=mask, other=0.0)
            x1_real = tl.load(ptr1_real, mask=mask, other=0.0)
            x1_imag = tl.load(ptr1_imag, mask=mask, other=0.0)
            x2_real = tl.load(ptr2_real, mask=mask, other=0.0)
            x2_imag = tl.load(ptr2_imag, mask=mask, other=0.0)
            x3_real = tl.load(ptr3_real, mask=mask, other=0.0)
            x3_imag = tl.load(ptr3_imag, mask=mask, other=0.0)

            base_tw1 = (1 << (stage_s - 1)) - 1
            base_tw2 = (1 << stage_s) - 1
            tw1_idx = base_tw1 + j
            tw2_idx = base_tw2 + j
            tw1_real = tl.load(twiddle_real + tw1_idx, mask=mask, other=1.0)
            tw1_imag = tl.load(twiddle_imag + tw1_idx, mask=mask, other=0.0)
            tw2_real = tl.load(twiddle_real + tw2_idx, mask=mask, other=1.0)
            tw2_imag = tl.load(twiddle_imag + tw2_idx, mask=mask, other=0.0)

            t1_real = x1_real * tw1_real - x1_imag * tw1_imag
            t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
            t3_real = x3_real * tw1_real - x3_imag * tw1_imag
            t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

            u0_real = x0_real + t1_real
            u0_imag = x0_imag + t1_imag
            u1_real = x0_real - t1_real
            u1_imag = x0_imag - t1_imag
            v0_real = x2_real + t3_real
            v0_imag = x2_imag + t3_imag
            v1_real = x2_real - t3_real
            v1_imag = x2_imag - t3_imag

            v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
            v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
            w3_real = tw2_imag
            w3_imag = -tw2_real
            v1_tw_real = v1_real * w3_real - v1_imag * w3_imag
            v1_tw_imag = v1_real * w3_imag + v1_imag * w3_real

            o0_real = u0_real + v0_tw_real
            o0_imag = u0_imag + v0_tw_imag
            o2_real = u0_real - v0_tw_real
            o2_imag = u0_imag - v0_tw_imag
            o1_real = u1_real + v1_tw_real
            o1_imag = u1_imag + v1_tw_imag
            o3_real = u1_real - v1_tw_real
            o3_imag = u1_imag - v1_tw_imag

            m0 = pos < quarter
            m1 = (pos >= quarter) & (pos < half)
            m2 = (pos >= half) & (pos < three_quarter)
            out_real = tl.where(
                m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real))
            )
            out_imag = tl.where(
                m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag))
            )

            out_ptrs_real = buf_b_real + row * stride_buf + idx
            out_ptrs_imag = buf_b_imag + row * stride_buf + idx
            tl.store(out_ptrs_real, out_real, mask=mask)
            tl.store(out_ptrs_imag, out_imag, mask=mask)
            tl.debug_barrier()

            buf_a_real, buf_b_real = buf_b_real, buf_a_real
            buf_a_imag, buf_b_imag = buf_b_imag, buf_a_imag


if HAS_TLE:

    @triton.jit
    def fft_kernel_tle(
        in_real,
        in_imag,
        bitrev,
        twiddle_real,
        twiddle_imag,
        out_real,
        out_imag,
        stride_in,
        stride_out,
        n_rows,
        N: tl.constexpr,
        LOG_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid
        offs = tl.arange(0, N)
        row_valid = row < n_rows
        mask = row_valid & (offs < N)

        smem_a_real = tle.gpu.alloc(
            [N],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_a_imag = tle.gpu.alloc(
            [N],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_b_real = tle.gpu.alloc(
            [N],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_b_imag = tle.gpu.alloc(
            [N],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )

        rev = tl.load(bitrev + offs, mask=offs < N, other=0)
        in_real_ptrs = in_real + row * stride_in + rev
        in_imag_ptrs = in_imag + row * stride_in + rev
        vals_real = tl.load(in_real_ptrs, mask=mask, other=0.0)
        vals_imag = tl.load(in_imag_ptrs, mask=mask, other=0.0)

        smem_a_real_ptrs = tle.gpu.local_ptr(smem_a_real, (offs,))
        smem_a_imag_ptrs = tle.gpu.local_ptr(smem_a_imag, (offs,))
        tl.store(smem_a_real_ptrs, vals_real, mask=mask)
        tl.store(smem_a_imag_ptrs, vals_imag, mask=mask)
        tl.debug_barrier()

        smem_in_real = smem_a_real
        smem_in_imag = smem_a_imag
        smem_out_real = smem_b_real
        smem_out_imag = smem_b_imag

        if LOG_N % 2 == 1:
            m = 2
            half = 1
            idx = offs
            pos = idx & (m - 1)
            j = pos & (half - 1)
            base = idx - pos
            even_idx = base + j
            odd_idx = even_idx + half

            even_ptrs_real = tle.gpu.local_ptr(smem_in_real, (even_idx,))
            even_ptrs_imag = tle.gpu.local_ptr(smem_in_imag, (even_idx,))
            odd_ptrs_real = tle.gpu.local_ptr(smem_in_real, (odd_idx,))
            odd_ptrs_imag = tle.gpu.local_ptr(smem_in_imag, (odd_idx,))

            u_real = tl.load(even_ptrs_real, mask=mask, other=0.0)
            u_imag = tl.load(even_ptrs_imag, mask=mask, other=0.0)
            v_real = tl.load(odd_ptrs_real, mask=mask, other=0.0)
            v_imag = tl.load(odd_ptrs_imag, mask=mask, other=0.0)

            base_tw = 0
            tw_idx = base_tw + j
            tw_real = tl.load(twiddle_real + tw_idx, mask=mask, other=1.0)
            tw_imag = tl.load(twiddle_imag + tw_idx, mask=mask, other=0.0)

            v_tw_real = v_real * tw_real - v_imag * tw_imag
            v_tw_imag = v_real * tw_imag + v_imag * tw_real

            add_mask = pos < half
            out_real_val = tl.where(add_mask, u_real + v_tw_real, u_real - v_tw_real)
            out_imag_val = tl.where(add_mask, u_imag + v_tw_imag, u_imag - v_tw_imag)

            out_ptrs_real = tle.gpu.local_ptr(smem_out_real, (idx,))
            out_ptrs_imag = tle.gpu.local_ptr(smem_out_imag, (idx,))
            tl.store(out_ptrs_real, out_real_val, mask=mask)
            tl.store(out_ptrs_imag, out_imag_val, mask=mask)
            tl.debug_barrier()

            smem_in_real, smem_out_real = smem_out_real, smem_in_real
            smem_in_imag, smem_out_imag = smem_out_imag, smem_in_imag

        if LOG_N % 2 == 1:
            for r4 in tl.static_range((LOG_N - 1) // 2):
                stage_s = 2 + r4 * 2
                m = 1 << (stage_s + 1)
                quarter = m >> 2
                half = m >> 1
                three_quarter = quarter + half

                idx = offs
                pos = idx & (m - 1)
                j = pos & (quarter - 1)
                base = idx - pos
                i0 = base + j
                i1 = i0 + quarter
                i2 = i1 + quarter
                i3 = i2 + quarter

                ptr0_real = tle.gpu.local_ptr(smem_in_real, (i0,))
                ptr0_imag = tle.gpu.local_ptr(smem_in_imag, (i0,))
                ptr1_real = tle.gpu.local_ptr(smem_in_real, (i1,))
                ptr1_imag = tle.gpu.local_ptr(smem_in_imag, (i1,))
                ptr2_real = tle.gpu.local_ptr(smem_in_real, (i2,))
                ptr2_imag = tle.gpu.local_ptr(smem_in_imag, (i2,))
                ptr3_real = tle.gpu.local_ptr(smem_in_real, (i3,))
                ptr3_imag = tle.gpu.local_ptr(smem_in_imag, (i3,))

                x0_real = tl.load(ptr0_real, mask=mask, other=0.0)
                x0_imag = tl.load(ptr0_imag, mask=mask, other=0.0)
                x1_real = tl.load(ptr1_real, mask=mask, other=0.0)
                x1_imag = tl.load(ptr1_imag, mask=mask, other=0.0)
                x2_real = tl.load(ptr2_real, mask=mask, other=0.0)
                x2_imag = tl.load(ptr2_imag, mask=mask, other=0.0)
                x3_real = tl.load(ptr3_real, mask=mask, other=0.0)
                x3_imag = tl.load(ptr3_imag, mask=mask, other=0.0)

                base_tw1 = (1 << (stage_s - 1)) - 1
                base_tw2 = (1 << stage_s) - 1
                tw1_idx = base_tw1 + j
                tw2_idx = base_tw2 + j
                tw1_real = tl.load(twiddle_real + tw1_idx, mask=mask, other=1.0)
                tw1_imag = tl.load(twiddle_imag + tw1_idx, mask=mask, other=0.0)
                tw2_real = tl.load(twiddle_real + tw2_idx, mask=mask, other=1.0)
                tw2_imag = tl.load(twiddle_imag + tw2_idx, mask=mask, other=0.0)

                t1_real = x1_real * tw1_real - x1_imag * tw1_imag
                t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
                t3_real = x3_real * tw1_real - x3_imag * tw1_imag
                t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

                u0_real = x0_real + t1_real
                u0_imag = x0_imag + t1_imag
                u1_real = x0_real - t1_real
                u1_imag = x0_imag - t1_imag
                v0_real = x2_real + t3_real
                v0_imag = x2_imag + t3_imag
                v1_real = x2_real - t3_real
                v1_imag = x2_imag - t3_imag

                v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
                v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
                w3_real = tw2_imag
                w3_imag = -tw2_real
                v1_tw_real = v1_real * w3_real - v1_imag * w3_imag
                v1_tw_imag = v1_real * w3_imag + v1_imag * w3_real

                o0_real = u0_real + v0_tw_real
                o0_imag = u0_imag + v0_tw_imag
                o2_real = u0_real - v0_tw_real
                o2_imag = u0_imag - v0_tw_imag
                o1_real = u1_real + v1_tw_real
                o1_imag = u1_imag + v1_tw_imag
                o3_real = u1_real - v1_tw_real
                o3_imag = u1_imag - v1_tw_imag

                m0 = pos < quarter
                m1 = (pos >= quarter) & (pos < half)
                m2 = (pos >= half) & (pos < three_quarter)
                out_real_val = tl.where(
                    m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real))
                )
                out_imag_val = tl.where(
                    m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag))
                )

                out_ptrs_real = tle.gpu.local_ptr(smem_out_real, (idx,))
                out_ptrs_imag = tle.gpu.local_ptr(smem_out_imag, (idx,))
                tl.store(out_ptrs_real, out_real_val, mask=mask)
                tl.store(out_ptrs_imag, out_imag_val, mask=mask)
                tl.debug_barrier()

                smem_in_real, smem_out_real = smem_out_real, smem_in_real
                smem_in_imag, smem_out_imag = smem_out_imag, smem_in_imag
        else:
            for r4 in tl.static_range(LOG_N // 2):
                stage_s = 1 + r4 * 2
                m = 1 << (stage_s + 1)
                quarter = m >> 2
                half = m >> 1
                three_quarter = quarter + half

                idx = offs
                pos = idx & (m - 1)
                j = pos & (quarter - 1)
                base = idx - pos
                i0 = base + j
                i1 = i0 + quarter
                i2 = i1 + quarter
                i3 = i2 + quarter

                ptr0_real = tle.gpu.local_ptr(smem_in_real, (i0,))
                ptr0_imag = tle.gpu.local_ptr(smem_in_imag, (i0,))
                ptr1_real = tle.gpu.local_ptr(smem_in_real, (i1,))
                ptr1_imag = tle.gpu.local_ptr(smem_in_imag, (i1,))
                ptr2_real = tle.gpu.local_ptr(smem_in_real, (i2,))
                ptr2_imag = tle.gpu.local_ptr(smem_in_imag, (i2,))
                ptr3_real = tle.gpu.local_ptr(smem_in_real, (i3,))
                ptr3_imag = tle.gpu.local_ptr(smem_in_imag, (i3,))

                x0_real = tl.load(ptr0_real, mask=mask, other=0.0)
                x0_imag = tl.load(ptr0_imag, mask=mask, other=0.0)
                x1_real = tl.load(ptr1_real, mask=mask, other=0.0)
                x1_imag = tl.load(ptr1_imag, mask=mask, other=0.0)
                x2_real = tl.load(ptr2_real, mask=mask, other=0.0)
                x2_imag = tl.load(ptr2_imag, mask=mask, other=0.0)
                x3_real = tl.load(ptr3_real, mask=mask, other=0.0)
                x3_imag = tl.load(ptr3_imag, mask=mask, other=0.0)

                base_tw1 = (1 << (stage_s - 1)) - 1
                base_tw2 = (1 << stage_s) - 1
                tw1_idx = base_tw1 + j
                tw2_idx = base_tw2 + j
                tw1_real = tl.load(twiddle_real + tw1_idx, mask=mask, other=1.0)
                tw1_imag = tl.load(twiddle_imag + tw1_idx, mask=mask, other=0.0)
                tw2_real = tl.load(twiddle_real + tw2_idx, mask=mask, other=1.0)
                tw2_imag = tl.load(twiddle_imag + tw2_idx, mask=mask, other=0.0)

                t1_real = x1_real * tw1_real - x1_imag * tw1_imag
                t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
                t3_real = x3_real * tw1_real - x3_imag * tw1_imag
                t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

                u0_real = x0_real + t1_real
                u0_imag = x0_imag + t1_imag
                u1_real = x0_real - t1_real
                u1_imag = x0_imag - t1_imag
                v0_real = x2_real + t3_real
                v0_imag = x2_imag + t3_imag
                v1_real = x2_real - t3_real
                v1_imag = x2_imag - t3_imag

                v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
                v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
                w3_real = tw2_imag
                w3_imag = -tw2_real
                v1_tw_real = v1_real * w3_real - v1_imag * w3_imag
                v1_tw_imag = v1_real * w3_imag + v1_imag * w3_real

                o0_real = u0_real + v0_tw_real
                o0_imag = u0_imag + v0_tw_imag
                o2_real = u0_real - v0_tw_real
                o2_imag = u0_imag - v0_tw_imag
                o1_real = u1_real + v1_tw_real
                o1_imag = u1_imag + v1_tw_imag
                o3_real = u1_real - v1_tw_real
                o3_imag = u1_imag - v1_tw_imag

                m0 = pos < quarter
                m1 = (pos >= quarter) & (pos < half)
                m2 = (pos >= half) & (pos < three_quarter)
                out_real_val = tl.where(
                    m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real))
                )
                out_imag_val = tl.where(
                    m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag))
                )

                out_ptrs_real = tle.gpu.local_ptr(smem_out_real, (idx,))
                out_ptrs_imag = tle.gpu.local_ptr(smem_out_imag, (idx,))
                tl.store(out_ptrs_real, out_real_val, mask=mask)
                tl.store(out_ptrs_imag, out_imag_val, mask=mask)
                tl.debug_barrier()

                smem_in_real, smem_out_real = smem_out_real, smem_in_real
                smem_in_imag, smem_out_imag = smem_out_imag, smem_in_imag

        out_real_ptrs = out_real + row * stride_out + offs
        out_imag_ptrs = out_imag + row * stride_out + offs
        smem_final_real_ptrs = tle.gpu.local_ptr(smem_in_real, (offs,))
        smem_final_imag_ptrs = tle.gpu.local_ptr(smem_in_imag, (offs,))
        out_vals_real = tl.load(smem_final_real_ptrs, mask=mask, other=0.0)
        out_vals_imag = tl.load(smem_final_imag_ptrs, mask=mask, other=0.0)
        tl.store(out_real_ptrs, out_vals_real, mask=mask)
        tl.store(out_imag_ptrs, out_vals_imag, mask=mask)

    @triton.jit
    def fft_kernel_tle_reg(
        in_real,
        in_imag,
        bitrev,
        twiddle_real,
        twiddle_imag,
        out_real,
        out_imag,
        stride_in,
        stride_out,
        n_rows,
        N: tl.constexpr,
        LOG_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid
        offs = tl.arange(0, N)
        row_valid = row < n_rows
        mask = row_valid & (offs < N)

        rev = tl.load(bitrev + offs, mask=offs < N, other=0)
        in_real_ptrs = in_real + row * stride_in + rev
        in_imag_ptrs = in_imag + row * stride_in + rev
        x_real = tl.load(in_real_ptrs, mask=mask, other=0.0)
        x_imag = tl.load(in_imag_ptrs, mask=mask, other=0.0)

        if LOG_N % 2 == 1:
            m = 2
            half = 1
            idx = offs
            pos = idx & (m - 1)
            j = pos & (half - 1)
            base = idx - pos
            even_idx = base + j
            odd_idx = even_idx + half

            u_real = tl.gather(x_real, even_idx, axis=0)
            u_imag = tl.gather(x_imag, even_idx, axis=0)
            v_real = tl.gather(x_real, odd_idx, axis=0)
            v_imag = tl.gather(x_imag, odd_idx, axis=0)

            tw_real = tl.load(twiddle_real + j, mask=mask, other=1.0)
            tw_imag = tl.load(twiddle_imag + j, mask=mask, other=0.0)

            v_tw_real = v_real * tw_real - v_imag * tw_imag
            v_tw_imag = v_real * tw_imag + v_imag * tw_real

            add_mask = pos < half
            out_real_val = tl.where(add_mask, u_real + v_tw_real, u_real - v_tw_real)
            out_imag_val = tl.where(add_mask, u_imag + v_tw_imag, u_imag - v_tw_imag)
            x_real = out_real_val
            x_imag = out_imag_val

        if LOG_N % 2 == 1:
            for r4 in tl.static_range((LOG_N - 1) // 2):
                stage_s = 2 + r4 * 2
                m = 1 << (stage_s + 1)
                quarter = m >> 2
                half = m >> 1
                three_quarter = quarter + half

                idx = offs
                pos = idx & (m - 1)
                j = pos & (quarter - 1)
                base = idx - pos
                i0 = base + j
                i1 = i0 + quarter
                i2 = i1 + quarter
                i3 = i2 + quarter

                x0_real = tl.gather(x_real, i0, axis=0)
                x0_imag = tl.gather(x_imag, i0, axis=0)
                x1_real = tl.gather(x_real, i1, axis=0)
                x1_imag = tl.gather(x_imag, i1, axis=0)
                x2_real = tl.gather(x_real, i2, axis=0)
                x2_imag = tl.gather(x_imag, i2, axis=0)
                x3_real = tl.gather(x_real, i3, axis=0)
                x3_imag = tl.gather(x_imag, i3, axis=0)

                base_tw1 = (1 << (stage_s - 1)) - 1
                base_tw2 = (1 << stage_s) - 1
                tw1_idx = base_tw1 + j
                tw2_idx = base_tw2 + j
                tw1_real = tl.load(twiddle_real + tw1_idx, mask=mask, other=1.0)
                tw1_imag = tl.load(twiddle_imag + tw1_idx, mask=mask, other=0.0)
                tw2_real = tl.load(twiddle_real + tw2_idx, mask=mask, other=1.0)
                tw2_imag = tl.load(twiddle_imag + tw2_idx, mask=mask, other=0.0)

                t1_real = x1_real * tw1_real - x1_imag * tw1_imag
                t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
                t3_real = x3_real * tw1_real - x3_imag * tw1_imag
                t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

                u0_real = x0_real + t1_real
                u0_imag = x0_imag + t1_imag
                u1_real = x0_real - t1_real
                u1_imag = x0_imag - t1_imag
                v0_real = x2_real + t3_real
                v0_imag = x2_imag + t3_imag
                v1_real = x2_real - t3_real
                v1_imag = x2_imag - t3_imag

                v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
                v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
                w3_real = tw2_imag
                w3_imag = -tw2_real
                v1_tw_real = v1_real * w3_real - v1_imag * w3_imag
                v1_tw_imag = v1_real * w3_imag + v1_imag * w3_real

                o0_real = u0_real + v0_tw_real
                o0_imag = u0_imag + v0_tw_imag
                o2_real = u0_real - v0_tw_real
                o2_imag = u0_imag - v0_tw_imag
                o1_real = u1_real + v1_tw_real
                o1_imag = u1_imag + v1_tw_imag
                o3_real = u1_real - v1_tw_real
                o3_imag = u1_imag - v1_tw_imag

                m0 = pos < quarter
                m1 = (pos >= quarter) & (pos < half)
                m2 = (pos >= half) & (pos < three_quarter)
                out_real_val = tl.where(
                    m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real))
                )
                out_imag_val = tl.where(
                    m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag))
                )
                x_real = out_real_val
                x_imag = out_imag_val
        else:
            for r4 in tl.static_range(LOG_N // 2):
                stage_s = 1 + r4 * 2
                m = 1 << (stage_s + 1)
                quarter = m >> 2
                half = m >> 1
                three_quarter = quarter + half

                idx = offs
                pos = idx & (m - 1)
                j = pos & (quarter - 1)
                base = idx - pos
                i0 = base + j
                i1 = i0 + quarter
                i2 = i1 + quarter
                i3 = i2 + quarter

                x0_real = tl.gather(x_real, i0, axis=0)
                x0_imag = tl.gather(x_imag, i0, axis=0)
                x1_real = tl.gather(x_real, i1, axis=0)
                x1_imag = tl.gather(x_imag, i1, axis=0)
                x2_real = tl.gather(x_real, i2, axis=0)
                x2_imag = tl.gather(x_imag, i2, axis=0)
                x3_real = tl.gather(x_real, i3, axis=0)
                x3_imag = tl.gather(x_imag, i3, axis=0)

                base_tw1 = (1 << (stage_s - 1)) - 1
                base_tw2 = (1 << stage_s) - 1
                tw1_idx = base_tw1 + j
                tw2_idx = base_tw2 + j
                tw1_real = tl.load(twiddle_real + tw1_idx, mask=mask, other=1.0)
                tw1_imag = tl.load(twiddle_imag + tw1_idx, mask=mask, other=0.0)
                tw2_real = tl.load(twiddle_real + tw2_idx, mask=mask, other=1.0)
                tw2_imag = tl.load(twiddle_imag + tw2_idx, mask=mask, other=0.0)

                t1_real = x1_real * tw1_real - x1_imag * tw1_imag
                t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
                t3_real = x3_real * tw1_real - x3_imag * tw1_imag
                t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

                u0_real = x0_real + t1_real
                u0_imag = x0_imag + t1_imag
                u1_real = x0_real - t1_real
                u1_imag = x0_imag - t1_imag
                v0_real = x2_real + t3_real
                v0_imag = x2_imag + t3_imag
                v1_real = x2_real - t3_real
                v1_imag = x2_imag - t3_imag

                v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
                v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
                w3_real = tw2_imag
                w3_imag = -tw2_real
                v1_tw_real = v1_real * w3_real - v1_imag * w3_imag
                v1_tw_imag = v1_real * w3_imag + v1_imag * w3_real

                o0_real = u0_real + v0_tw_real
                o0_imag = u0_imag + v0_tw_imag
                o2_real = u0_real - v0_tw_real
                o2_imag = u0_imag - v0_tw_imag
                o1_real = u1_real + v1_tw_real
                o1_imag = u1_imag + v1_tw_imag
                o3_real = u1_real - v1_tw_real
                o3_imag = u1_imag - v1_tw_imag

                m0 = pos < quarter
                m1 = (pos >= quarter) & (pos < half)
                m2 = (pos >= half) & (pos < three_quarter)
                out_real_val = tl.where(
                    m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real))
                )
                out_imag_val = tl.where(
                    m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag))
                )
                x_real = out_real_val
                x_imag = out_imag_val

        out_real_ptrs = out_real + row * stride_out + offs
        out_imag_ptrs = out_imag + row * stride_out + offs
        tl.store(out_real_ptrs, x_real, mask=mask)
        tl.store(out_imag_ptrs, x_imag, mask=mask)


def fft(x: torch.Tensor) -> torch.Tensor:
    """
    1D FFT with Triton and TLE (TLE Tutorial)
    =======================================

    This tutorial implements a simple 1D complex FFT over the last dimension of an
    (M, N) tensor and compares Triton vs TLE kernels against torch.fft.fft. If
    `cuda.tile` is available, it also runs a cuTile FFT kernel adapted from NVIDIA's
    cutile-python tests.

    Notes
    -----
    - N must be a power-of-two (<= 1024) for this tutorial implementation.
    - Complex values are represented as two float32 arrays (real/imag).
    - The kernels implement iterative Cooley-Tukey DIT with a bit-reversal copy.
    - Twiddle factors are precomputed on the host and read from global memory.
    - TLE uses a register-only path for small N to reduce shared-memory traffic.
    - cuTile path is optional and requires `cuda.tile` + `cupy`; it uses a 3-factor
      decomposition with precomputed DFT/twiddle tables.
    """
    logger.debug("GEMS FFT")
    assert x.is_cuda, "input must be on CUDA"
    assert x.ndim == 2, "input must be 2D (M, N)"
    m, n = x.shape
    if not _is_power_of_two(n):
        raise ValueError(f"N={n} must be a power-of-two")
    if n > 1024:
        raise ValueError(f"N={n} too large for this kernel (max 1024)")

    in_real, in_imag = _prepare_input(x)
    bitrev = _bitrev_indices(n, x.device)
    tw_real, tw_imag = _twiddle_tables(n, x.device)
    log_n = _log2(n)

    with torch_device_fn.device(x.device):
        if HAS_TLE:
            out_real = torch.empty((m, n), device=x.device, dtype=torch.float32)
            out_imag = torch.empty((m, n), device=x.device, dtype=torch.float32)

            grid = (m,)
            if n == _FFT_REG_THRESHOLD:
                fft_kernel_tle_reg[grid](
                    in_real,
                    in_imag,
                    bitrev,
                    tw_real,
                    tw_imag,
                    out_real,
                    out_imag,
                    in_real.stride(0),
                    out_real.stride(0),
                    m,
                    N=n,
                    LOG_N=log_n,
                    num_warps=4,
                    num_stages=1,
                )
            else:
                fft_kernel_tle[grid](
                    in_real,
                    in_imag,
                    bitrev,
                    tw_real,
                    tw_imag,
                    out_real,
                    out_imag,
                    in_real.stride(0),
                    out_real.stride(0),
                    m,
                    N=n,
                    LOG_N=log_n,
                    num_warps=4,
                    num_stages=1,
                )
            return torch.complex(out_real, out_imag)
        else:
            buf0_real = torch.empty((m, n), device=x.device, dtype=torch.float32)
            buf0_imag = torch.empty((m, n), device=x.device, dtype=torch.float32)
            buf1_real = torch.empty((m, n), device=x.device, dtype=torch.float32)
            buf1_imag = torch.empty((m, n), device=x.device, dtype=torch.float32)

            grid = (m,)
            fft_kernel_triton[grid](
                in_real,
                in_imag,
                bitrev,
                tw_real,
                tw_imag,
                buf0_real,
                buf0_imag,
                buf1_real,
                buf1_imag,
                in_real.stride(0),
                buf0_real.stride(0),
                m,
                N=n,
                LOG_N=log_n,
                num_warps=4,
                num_stages=1,
            )

            # Kernel swaps buf_a/buf_b after each stage write.
            # Total swaps = (log_n + 1) // 2 (1 radix-2 if odd, then radix-4 pairs).
            # Result lands in buf0 when total_swaps is even, buf1 when odd.
            total_swaps = (log_n + 1) // 2
            if total_swaps % 2 == 0:
                out_real = buf0_real
                out_imag = buf0_imag
            else:
                out_real = buf1_real
                out_imag = buf1_imag

            return torch.complex(out_real, out_imag)
