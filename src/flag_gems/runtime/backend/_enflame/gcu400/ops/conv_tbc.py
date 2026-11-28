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

logger = logging.getLogger(__name__)


def _conv_tbc_configs():
    """Candidate block configs on gcu400 (measured, see logs/conv_tbc_sweep*.log).

    Does not reuse `runtime.get_tuned_config("conv_tbc")`: that table is the full grid of
    32/64/128 block sizes x num_stages=2 (52 entries) generated for the masked-load kernel,
    whose blocks are small; with unmasked loads big blocks measured 2~3x faster, while the full
    grid doubles compile/autotune time and biases autotune toward suboptimal picks. Only the 5
    measured-best entries are kept, plus one small-block fallback for tiny shapes
    """
    return [
        # Large N/Cin: best for shapes like N=512, Cin=512
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256},
            num_warps=4,
            num_stages=2,
        ),
        # Medium N/Cin (N=256, Cin<=256): best for most medium shapes
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128},
            num_warps=8,
            num_stages=2,
        ),
        # Small N (N<=128): general-purpose
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
            num_warps=8,
            num_stages=3,
        ),
        # Small Cin (Cin<=64) and thin/tall shapes (such as the bench shape with kW=7, Cin=64): BM=64
        # raises the grid from (32,1) (67% single-wave occupancy) to (64,1), measured ~10% faster (64.3→58.5us
        # fp16 / 69.1→61.9us fp32; single-config A/B under the same methodology as the official bench);
        # BM=64 is not advantageous for keys with Cin>=128 because num_k doubles, so autotune will not select it by
        # mistake.
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=4,
            num_stages=2,
        ),
        # Very small Cin (such as the bench shape with kW=7, Cin=64)
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=8,
            num_stages=3,
        ),
        # Fallback for small blocks: shapes with extremely small unit dimensions
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
    ]


_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

# Hybrid dispatch.  conv_tbc is an implicit GEMM (A rows shift by (k-pad)*B
# per tap); the Triton kernel below wins every light conv shape, but the GCU
# native conv library is physically faster once the conv is GEMM-heavy *and*
# the channel dim is wide enough for it to vectorise: measured 33.8-92.5
# TFLOPS on the Cin>=256 bench shapes vs a 17-42 TFLOPS ceiling for Triton
# tl.dot here (every dot has K <= Cin <= 512, no large-K amortisation).
# Weight-MAC count (Cin*kW*Cout) alone is not a sufficient native-strength
# proxy: at Cin=128 the native library only reaches ~19 TFLOPS (bench shape
# (512,8,128,256,5,2) runs 58.5us Triton vs 69.2us native), so the 2^17 MAC
# threshold from the j20261118053344 campaign must be gated by channel width.
_NATIVE_MACS = 1 << 17
_NATIVE_CIN = 256

# Triton-arm safety envelope.  Kernel offsets are int32 (element products)
# and the launch grid is capped per axis on gcu400 (grid.y <= 255; the
# smallest tile in _conv_tbc_configs is 64, which sets the envelope below).
# Anything outside it keeps native semantics instead of risking OOB.
_MAX_TRITON_ELEMS = 2**31 - 1
_MAX_GRID_X = 65535
_MAX_GRID_Y = 255
_MIN_TILE = 64
_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _prefers_native(input, weight, bias, pad, Tout):
    """Return True when the native composite conv should keep the call."""
    if input.device.type != "gcu":
        return True
    if pad < 0:
        return True
    if input.dtype not in _TRITON_DTYPES or weight.dtype != input.dtype:
        return True
    T, B, Cin = input.shape
    kW, _, Cout = weight.shape
    if bias.numel() != Cout:
        return True
    if (T + 2 * pad + kW) * B * Cin > _MAX_TRITON_ELEMS:
        return True
    if kW * Cin * Cout > _MAX_TRITON_ELEMS or Tout * B * Cout > _MAX_TRITON_ELEMS:
        return True
    if Tout * B > _MAX_GRID_X * _MIN_TILE or Cout > _MAX_GRID_Y * _MIN_TILE:
        return True
    return Cin >= _NATIVE_CIN and Cin * kW * Cout >= _NATIVE_MACS


@triton.autotune(configs=_conv_tbc_configs(), key=["M", "N", "Cin", "kW"])
@triton.heuristics(
    {
        # Whether the column dimension is divisible is determined by each candidate config's own BLOCK_K/BLOCK_N:
        # when divisible, clamp and masks are removed entirely (on-chip DSM usage is also much lower); otherwise fall
        # back to clamp + zeroing
        "EVEN_K": lambda args: args["Cin"] % args["BLOCK_K"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
# Keep kW a runtime value.  Triton specialises an int argument equal to 1 as a
# constant, which unrolls the tap loop into straight-line code; the elementwise
# fusion then drops the f32->f16 conversion before the store and the compiler
# fails to legalize the DMA store ("requires matching element types, got
# value='f32' ptr='f16'", which also takes the process down).  Measured with
# T=32,B=2,Cin=64,Cout=32,fp16: kW=1 aborts, kW=2/3 are exact.
@triton.jit(do_not_specialize=["kW"])
def conv_tbc_kernel(
    inp_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,  # Tout * B
    N,  # Cout
    Cin,
    kW,
    B,
    T,
    pad,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    # out[m, n] = bias[n] + sum_k sum_ci inp_row(m, k)[ci] * weight[k, ci, n]
    # where m = to * B + b, and the input row for offset k is m + (k - pad) * B
    # (this shift preserves the batch index b automatically).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_valid = offs_m < M
    n_valid = offs_n < N
    # The clamped column index is only used to compute load addresses; out-of-bounds columns are zeroed in the result
    # via n_valid
    n_ld = offs_n if EVEN_N else tl.where(n_valid, offs_n, 0)

    num_k = tl.cdiv(Cin, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # The tap index and Cin block index are merged into a single loop level: the pipeline warms up only once,
    # about 10% faster than the nested `for k: for ki:` form when BLOCK_K>=128.
    for kk in range(kW * num_k):
        k = kk // num_k
        ki = kk % num_k

        # The input row corresponding to this tap (row index = to * B + b; the tap shifts by B rows only along the time
        # dimension)
        in_row = offs_m + (k - pad) * B
        row_valid = m_valid & (in_row >= 0) & (in_row < T * B)
        # Out-of-bounds rows (padding or tail overflow) always load from row 0 and are then zeroed by row_valid
        safe_row = tl.where(row_valid, in_row, 0)

        ck = ki * BLOCK_K + offs_k
        k_valid = ck < Cin
        ck_ld = ck if EVEN_K else tl.where(k_valid, ck, 0)

        a = tl.load(inp_ptr + safe_row[:, None] * Cin + ck_ld[None, :])
        if EVEN_K:
            a = tl.where(row_valid[:, None], a, 0.0)
        else:
            a = tl.where(row_valid[:, None] & k_valid[None, :], a, 0.0)

        w = tl.load(weight_ptr + k * Cin * N + ck_ld[:, None] * N + n_ld[None, :])
        if not (EVEN_K and EVEN_N):
            if EVEN_K:
                w = tl.where(n_valid[None, :], w, 0.0)
            elif EVEN_N:
                w = tl.where(k_valid[:, None], w, 0.0)
            else:
                w = tl.where(k_valid[:, None] & n_valid[None, :], w, 0.0)

        acc += tl.dot(a, w, allow_tf32=False)

    bias = tl.load(bias_ptr + n_ld, mask=n_valid, other=0.0).to(tl.float32)
    acc += bias[None, :]

    out_off = offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        out_ptr + out_off,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_valid[:, None] & n_valid[None, :],
    )


def _conv_tbc(input, weight, bias, pad, out=None):
    assert input.dim() == 3, "Input must be 3-dimensional (Time, Batch, Channels)"
    assert weight.dim() == 3, "Weight must be 3-dimensional (kW, Cin, Cout)"

    T, B, Cin = input.shape
    kW, Cin_w, Cout = weight.shape
    assert Cin == Cin_w, "Input channels must match weight input channels"

    Tout = T + 2 * pad - kW + 1

    # The delegate decision must happen before any `out` allocation: with a
    # live tensor the functional entry would take the `.out.redispatch` below,
    # whose composite body re-enters this registration via the full dispatcher
    # and loops forever (use_gems path; see the out-variant delegation trap in
    # knowledge/family_cases/flaggems_conv_tbc.md).
    if _prefers_native(input, weight, bias, pad, Tout):
        if out is None:
            return torch.ops.aten.conv_tbc.default.redispatch(
                _FALLBACK_KEYSET, input, weight, bias, pad
            )
        return torch.ops.aten.conv_tbc.out.redispatch(
            _FALLBACK_KEYSET, input, weight, bias, pad, out=out
        )

    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    if out is None:
        out = torch.empty((Tout, B, Cout), dtype=input.dtype, device=input.device)

    if Tout <= 0:
        return out

    M = Tout * B
    N = Cout

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    conv_tbc_kernel[grid](
        input,
        weight,
        bias,
        out,
        M,
        N,
        Cin,
        kW,
        B,
        T,
        pad,
    )
    return out


def conv_tbc(input, weight, bias, pad=0):
    logger.debug("GEMS_ENFLAME CONV_TBC")
    return _conv_tbc(input, weight, bias, pad)


def conv_tbc_out(input, weight, bias, pad=0, *, out):
    logger.debug("GEMS_ENFLAME CONV_TBC_OUT")
    return _conv_tbc(input, weight, bias, pad, out=out)
