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

from flag_gems.ops.scaled_grouped_mm import (
    _check_dims,
    _decode_e4m3_tensor,
    _default_out_dtype,
    _is_float8_dtype,
    _normalize_bias,
    _normalize_scale,
    _resolve_shapes,
    _scaled_grouped_mm_fallback,
    _supports_triton_dot,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _padded_group_mm(
    A,
    B,
    C,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)[:, None]
    rows = offsets // N
    cols = offsets % N
    ks = tl.arange(0, BLOCK)[None, :]
    a_ptrs = A + rows * K + ks
    b_ptrs = B + cols * K + ks
    acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    # Follow the vendor mv layout: retain lane-wise products across K tiles,
    # then reduce once outside the loop so CoreTiling sees the output chain.
    # Padding keeps every unmasked load inside the operands.
    for start in range(0, K, BLOCK):
        a = tl.load(a_ptrs).to(tl.float32)
        b = tl.load(b_ptrs).to(tl.float32)
        acc += a * b
        a_ptrs += BLOCK
        b_ptrs += BLOCK
    result = tl.sum(acc, axis=1)[:, None]
    tl.store(C + offsets, result)


@libentry()
@triton.jit
def _scaled_epilogue(
    Acc,
    ScaleA,
    ScaleB,
    Bias,
    Out,
    M,
    N,
    ACC_STRIDE,
    OUT_STRIDE_M,
    OUT_STRIDE_N,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Keep even masked addresses inside allocations on Triton XPU.
    valid_offsets = offsets % (M * N)
    rows = valid_offsets // N
    cols = valid_offsets % N
    acc = tl.load(Acc + rows * ACC_STRIDE + cols)
    scale_a = tl.load(ScaleA + rows)
    scale_b = tl.load(ScaleB + cols)
    result = acc * scale_a * scale_b
    if HAS_BIAS:
        result += tl.load(Bias + cols).to(tl.float32)
    tl.store(
        Out + rows * OUT_STRIDE_M + cols * OUT_STRIDE_N,
        result,
        mask=offsets < M * N,
    )


def _decode_operand(operand, fnuz):
    # Copy raw storage on device; do not depend on native FP8 cast support.
    count = operand.numel()
    padded_count = triton.cdiv(count, 256) * 256
    raw = torch.zeros((padded_count,), dtype=torch.uint8, device=operand.device)
    torch.ops.aten._copy_from(
        operand.view(torch.uint8), raw[:count].view(operand.shape), False
    )
    decoded = torch.empty_like(raw, dtype=torch.float16)
    if count:
        _decode_e4m3_tensor[(padded_count // 256,)](
            raw, decoded, padded_count, fnuz, 256
        )
    return decoded[:count].view(operand.shape)


def scaled_grouped_mm(
    self,
    mat2,
    scale_a,
    scale_b,
    offs=None,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    logger.debug("GEMS_KUNLUNXIN SCALED_GROUPED_MM")
    if scale_result is not None:
        raise RuntimeError("scale_result is not supported for scaled_grouped_mm")
    _check_dims(self, mat2)
    a_is_2d, b_is_2d, num_groups, M, N, K, out_shape, offs = _resolve_shapes(
        self, mat2, offs
    )
    output_dtype = out_dtype or _default_out_dtype(self.dtype)
    multiplier = num_groups if a_is_2d and b_is_2d else 1
    scale_a = _normalize_scale(
        scale_a,
        self,
        dim=0,
        num_groups=num_groups,
        scale_multiplier=multiplier,
        name="scale_a",
    )
    scale_b = _normalize_scale(
        scale_b,
        mat2,
        dim=1,
        num_groups=num_groups,
        scale_multiplier=multiplier,
        name="scale_b",
    )
    bias, _ = _normalize_bias(
        bias, a_is_2d=a_is_2d, b_is_2d=b_is_2d, num_groups=num_groups, N=N
    )
    if not _supports_triton_dot(self.dtype):
        return _scaled_grouped_mm_fallback(
            self,
            mat2,
            scale_a,
            scale_b,
            offs,
            bias,
            output_dtype,
            a_is_2d,
            b_is_2d,
            num_groups,
        )
    out = torch.empty(out_shape, dtype=output_dtype, device=self.device)
    if not out.numel():
        return out
    boundaries = [0] + (offs.detach().cpu().tolist() if offs is not None else [])
    with torch_device_fn.device(self.device):
        if self.dtype in tuple(
            getattr(torch, name)
            for name in ("float8_e4m3fn", "float8_e4m3fnuz")
            if hasattr(torch, name)
        ):
            fnuz = self.dtype == getattr(torch, "float8_e4m3fnuz", None)
            self = _decode_operand(self, fnuz)
            mat2 = _decode_operand(mat2, fnuz)
        elif _is_float8_dtype(self.dtype):
            self, mat2 = self.to(torch.float16), mat2.to(torch.float16)
        if a_is_2d and b_is_2d:
            scale_a = scale_a.view(num_groups, M)
            scale_b = scale_b.view(num_groups, N)
        # Split groups on the host so each device GEMM has fixed padded extents.
        # Preserve FP32 accumulation until the separate scaling epilogue.
        for group in range(num_groups):
            group_bias = bias if bias is None or bias.dim() == 1 else bias[group]
            if a_is_2d and not b_is_2d:
                start, end = boundaries[group : group + 2]
                a, b = self[start:end], mat2[group]
                sa, sb, dest = scale_a[start:end], scale_b[group], out[start:end]
            elif not a_is_2d and b_is_2d:
                start, end = boundaries[group : group + 2]
                a, b = self[group], mat2[:, start:end]
                sa, sb, dest = scale_a[group], scale_b[start:end], out[:, start:end]
                group_bias = bias[start:end] if bias is not None else None
            elif a_is_2d and b_is_2d:
                start, end = boundaries[group : group + 2]
                a, b = self[:, start:end], mat2[start:end]
                sa, sb, dest = scale_a[group], scale_b[group], out[group]
            else:
                a, b = self[group], mat2[group]
                sa, sb, dest = scale_a[group], scale_b[group], out[group]
            m, k = a.shape
            n = b.shape[1]
            if not m or not n:
                continue
            block = 32
            mp = triton.cdiv(m, block) * block
            np = triton.cdiv(n, block) * block
            kp = triton.cdiv(k, block) * block
            acc = torch.empty((mp, np), dtype=torch.float32, device=self.device)
            if k:
                padded_a = torch.zeros((mp, kp), dtype=a.dtype, device=self.device)
                # Both operands use a contiguous reduction dimension on XPU.
                padded_b = torch.zeros((np, kp), dtype=b.dtype, device=self.device)
                torch.ops.aten._copy_from(a, padded_a[:m, :k], False)
                torch.ops.aten._copy_from(b.T, padded_b[:n, :k], False)
                _padded_group_mm[(mp * np // block,)](
                    padded_a,
                    padded_b,
                    acc,
                    kp,
                    np,
                    block,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
            else:
                acc.zero_()
            _scaled_epilogue[(triton.cdiv(m * n, 256),)](
                acc,
                sa,
                sb,
                group_bias,
                dest,
                m,
                n,
                acc.stride(0),
                dest.stride(0),
                dest.stride(1),
                group_bias is not None,
                256,
                enable_fp_fusion=False,
            )
    return out
