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

from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.utils import CORE_NUM, compile_topk
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

al = None


@lru_cache(None)
def _prepare():
    global al
    from triton.language.extra.cann import extension as al

    cpp = Path(__file__).with_suffix(".cpp")
    bc, command, revision = compile_topk.build(cpp)

    class VectorOp:
        core = al.CORE.VECTOR
        pipe = al.PIPE.PIPE_V
        mode = al.MODE.SIMD
        bitcode = str(bc)
        source = str(cpp)
        compile = command
        extra_attr = "flaggems_pass_outputs=true"

    @al.register_custom_op
    class topk_sort_pairs(VectorOp):
        name = symbol = "topk_sort_pairs"

        def __init__(self, values, indices, scratch, n, out=None):
            self.arg_type["n"] = tl.int32

    @al.register_custom_op
    class topk_sort_prefix(VectorOp):
        name = symbol = "topk_sort_prefix"

        def __init__(self, values, indices, scratch, n, keep, out=None):
            self.arg_type["n"] = tl.int32
            self.arg_type["keep"] = tl.int32

    @al.register_custom_op
    class topk_select_codes(VectorOp):
        name = symbol = "topk_select_codes"

        def __init__(self, q, index, v, i, row, n, k, flip, scratch, out=None):
            for arg in ("row", "n", "k", "flip"):
                self.arg_type[arg] = tl.int32

    return revision


@lru_cache(None)
def _indices(n, device):
    return torch.arange(n, dtype=torch.int16).to(device)


@triton.jit
def _decode(q, E5: tl.constexpr):
    q = q.to(tl.uint16)
    if E5:
        # E5M2 and binary16 have the same exponent bias.
        bits = q << 8
        return bits.to(tl.float16, bitcast=True).to(tl.float32)
    else:
        # Reposition the sign/mantissa, then correct the exponent bias in
        # FP32. Casting before scaling preserves binary16 subnormals.
        bits = (q + (q & 128)) << 7
        return bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0


@triton.jit
def _sort(v, ids, B: tl.constexpr, K: tl.constexpr):
    scratch = tl.full((2 * B,), 0, tl.float32)
    dst = tl.full((2 * B,), 0, tl.float32)
    if B >= 128 and triton.next_power_of_2(K) <= 32:
        pairs = al.custom(
            "topk_sort_prefix",
            v,
            ids,
            scratch,
            B,
            max(8, triton.next_power_of_2(K)),
            out=dst,
        )
    else:
        pairs = al.custom("topk_sort_pairs", v, ids, scratch, B, out=dst)
    return pairs


@libentry()
@triton.jit
def _stage1(
    Q,
    S,
    V,
    Indices,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
    DESC: tl.constexpr,
    E5: tl.constexpr,
    TOTAL: tl.constexpr,
    CORES: tl.constexpr,
    REV: tl.constexpr,
):
    with al.scope(core_mode="vector"):
        pid = ext.program_id(0)
        for pid in range(pid, TOTAL, CORES):
            row = pid // P
            part = pid % P
            col = part * B + tl.arange(0, B)
            q = tl.load(Q + row * N + col, col < N, other=0)
            if G >= N:
                s = tl.load(S + row).to(tl.float32)
                qi = q.to(tl.int32)
                v = tl.where((qi & 128) != 0, -(qi & 127), qi & 127).to(tl.float32)
                v = tl.where(s < 0, -v, v)
            else:
                s = tl.load(S + row * NG + col // G, col < N, other=0).to(tl.float32)
                v = _decode(q, E5) * s
            if not DESC:
                v = -v
            v = tl.where(col < N, v, float("-inf"))
            pairs = _sort(v, col.to(tl.uint32), B, K)
            kk = tl.arange(0, triton.next_power_of_2(K))
            vals = tl.gather(pairs, 2 * kk, 0)
            ids = tl.gather(pairs, 2 * kk + 1, 0).to(tl.uint32, bitcast=True)
            if not DESC:
                vals = -vals
            if G >= N:
                code = tl.where(s < 0, -vals, vals).to(tl.int32)
                raw = tl.where(code < 0, 128 - code, code)
                vals = _decode(raw, E5) * s
            tl.store(V + pid * K + kk, vals, kk < K)
            tl.store(Indices + pid * K + kk, ids, kk < K)


@libentry()
@triton.jit
def _merge(
    V,
    Indices,
    Output,
    J,
    K: tl.constexpr,
    P: tl.constexpr,
    B: tl.constexpr,
    DESC: tl.constexpr,
    M: tl.constexpr,
    CORES: tl.constexpr,
    REV: tl.constexpr,
):
    with al.scope(core_mode="vector"):
        for row in range(ext.program_id(0), M, CORES):
            col = tl.arange(0, B)
            v = tl.load(V + row * P * K + col, col < P * K, other=0)
            ids = tl.load(Indices + row * P * K + col, col < P * K, other=0).to(
                tl.uint32
            )
            if not DESC:
                v = -v
            v = tl.where(col < P * K, v, float("-inf"))
            pairs = _sort(v, ids, B, K)
            kk = tl.arange(0, triton.next_power_of_2(K))
            vals = tl.gather(pairs, 2 * kk, 0)
            ids = tl.gather(pairs, 2 * kk + 1, 0).to(tl.uint32, bitcast=True)
            if not DESC:
                vals = -vals
            tl.store(Output + row * K + kk, vals, kk < K)
            tl.store(J + row * K + kk, ids, kk < K)


@libentry()
@triton.jit
def _row_finish(
    V,
    Indices,
    S,
    Output,
    J,
    K: tl.constexpr,
    B: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    DESC: tl.constexpr,
    E5: tl.constexpr,
    REV: tl.constexpr,
):
    with al.scope(core_mode="vector"):
        for row in range(ext.program_id(0), M, C):
            col = tl.arange(0, B)
            v = tl.load(V + row * K + col, col < K, other=float("-inf"))
            ids = tl.load(Indices + row * K + col, col < K, other=0).to(tl.uint32)
            pairs = _sort(v, ids, B, K)
            key = tl.gather(pairs, 2 * col, 0).to(tl.int32)
            idx = tl.gather(pairs, 2 * col + 1, 0).to(tl.uint32, bitcast=True)
            s = tl.load(S + row).to(tl.float32)
            if not DESC:
                key = 255 - key
            key = tl.where(s < 0, 255 - key, key)
            raw = tl.where(key >= 128, key - 128, 255 - key)
            values = _decode(raw, E5) * s
            tl.store(Output + row * K + col, values, col < K)
            tl.store(J + row * K + col, idx, col < K)


@libentry()
@triton.jit
def _row_select(
    Q,
    Index,
    S,
    V,
    Indices,
    N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    DESC: tl.constexpr,
    WORDS: tl.constexpr,
    REV: tl.constexpr,
):
    with al.scope(core_mode="vector"):
        for row in range(ext.program_id(0), M, C):
            scale = tl.load(S + row).to(tl.float32)
            flip = (scale < 0) ^ (not DESC)
            scratch = tl.full((WORDS,), 0, tl.int32)
            dummy = tl.full((16,), 0, tl.int32)
            al.custom(
                "topk_select_codes",
                Q,
                Index,
                V,
                Indices,
                row,
                N,
                K,
                flip.to(tl.int32),
                scratch,
                out=dummy,
            )


def topk_w8a16_fp8(x, x_scale, k, dim=-1, largest=True, sorted=True, group_size=128):
    """Last-dimension TopK of finite FP8 values and row/group scales.

    Values are selected in FP32 and returned as BF16 with int64 indices.
    """
    assert dim in (-1, x.ndim - 1)
    assert x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert x.is_contiguous() and x_scale.is_contiguous()
    assert x_scale.device == x.device and x_scale.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )
    n = x.shape[-1]
    m = x.numel() // n
    ng = triton.cdiv(n, group_size)
    assert 0 <= k <= n and x_scale.numel() == m * ng
    out = torch.empty(x.shape[:-1] + (k,), dtype=torch.bfloat16, device=x.device)
    indices = torch.empty_like(out, dtype=torch.int64)
    if k == 0 or m == 0:
        return out, indices
    with torch_device_fn.device(x.device):
        revision = _prepare()
        e5 = x.dtype == torch.float8_e5m2
        q = x.view(torch.uint8)
        cores = min(CORE_NUM, m)
        if (
            group_size >= n
            and n in (4096, 8192, 16384, 32768)
            and x.data_ptr() % 32 == 0
            and 8 <= k <= 512
            and k & (k - 1) == 0
        ):
            values = torch.empty((m * k,), dtype=torch.float32, device=x.device)
            ids = torch.empty((m * k,), dtype=torch.int32, device=x.device)
            _row_select[(cores,)](
                q,
                _indices(n, x.device),
                x_scale,
                values,
                ids,
                n,
                k,
                m,
                cores,
                largest,
                (5 * n + n // 4 + 8192) // 4,
                revision,
                multibuffer=False,
            )
            _row_finish[(cores,)](
                values,
                ids,
                x_scale,
                out,
                indices,
                k,
                max(32, triton.next_power_of_2(k)),
                m,
                cores,
                largest,
                e5,
                revision,
                multibuffer=False,
            )
        else:
            block = min(2048, max(32, triton.next_power_of_2(n)))
            parts = triton.cdiv(n, block)
            merged = triton.next_power_of_2(parts * k)
            assert k <= block and merged <= 4096
            values = (
                out
                if parts == 1
                else torch.empty((m * parts * k,), dtype=torch.float32, device=x.device)
            )
            ids = (
                indices
                if parts == 1
                else torch.empty((m * parts * k,), dtype=torch.int32, device=x.device)
            )
            grid = min(CORE_NUM, m * parts)
            _stage1[(grid,)](
                q,
                x_scale,
                values,
                ids,
                n,
                k,
                group_size,
                ng,
                block,
                parts,
                largest,
                e5,
                m * parts,
                grid,
                revision,
                multibuffer=False,
            )
            if parts > 1:
                _merge[(cores,)](
                    values,
                    ids,
                    out,
                    indices,
                    k,
                    parts,
                    max(32, merged),
                    largest,
                    m,
                    cores,
                    revision,
                    multibuffer=False,
                )
    return out, indices
