import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops.dropout import dropout as _dropout
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

try:
    from triton.language.extra.cann.extension import sync_block_all
except Exception:
    sync_block_all = None

_BLOCK_B = 16
_BLOCK_H_MAX = 128

_MIN_TIMELOOP_BLOCK_B = 8

_HSPLIT_BLOCK_B_MAX = 64

_IGEMM_BLOCK_B = 32
_IGEMM_BLOCK_N = 128
_IGEMM_BLOCK_K = 32
_IGEMM_NUM_WARPS = 8
_IGEMM_NUM_STAGES = 2
_IGEMM_MM_MIN_MACS = 5e8

_TRANSPOSE_BLOCK = 32

_PACK_BLOCK = 256


@libentry()
@triton.jit
def _transpose_weight_kernel(
    src,
    dst,
    rows,
    cols,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Transpose to (cols, rows) so the GEMM B operand's gate dim is contiguous.
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_r[:, None] < rows) & (offs_c[None, :] < cols)
    vals = tl.load(src + offs_r[:, None] * cols + offs_c[None, :], mask=mask, other=0.0)
    tl.store(dst + offs_c[None, :] * rows + offs_r[:, None], vals, mask=mask)


@libentry()
@triton.jit
def _batch_offsets_kernel(
    batch_sizes_ptr,
    offsets_ptr,
    bs32_ptr,
    num_steps,
    BLOCK: tl.constexpr,
):
    # Exclusive prefix-sum of batch_sizes (plus an int32 copy for the recurrence mask),
    # in one program via tl.cumsum. Avoids torch.cumsum/sub dispatch under use_gems().
    offs = tl.arange(0, BLOCK)
    mask = offs < num_steps
    vals = tl.load(batch_sizes_ptr + offs, mask=mask, other=0).to(tl.int32)
    exclusive = tl.cumsum(vals, axis=0) - vals
    tl.store(offsets_ptr + offs, exclusive, mask=mask)
    tl.store(bs32_ptr + offs, vals, mask=mask)


@libentry()
@triton.jit
def _unpack_padded_kernel(
    data_ptr,
    x_ptr,
    offsets_ptr,
    batch_sizes_ptr,
    input_size,
    batch_size,
    data_stride_0,
    x_stride_s,
    x_stride_b,
    x_stride_f,
    BLOCK_F: tl.constexpr,
):
    # Scatter packed (sum(batch_sizes), input) data into zero-padded (num_steps, batch,
    # input). One program per (timestep, batch-row); padding rows are left zero.
    pid = tl.program_id(0)
    t = pid // batch_size
    b = pid - t * batch_size
    bs_t = tl.load(batch_sizes_ptr + t).to(tl.int32)
    active = b < bs_t
    row = tl.load(offsets_ptr + t).to(tl.int32) + b
    for f_block in range(0, tl.cdiv(input_size, BLOCK_F)):
        offs_f = f_block * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = offs_f < input_size
        vals = tl.load(
            data_ptr + row * data_stride_0 + offs_f,
            mask=active & f_mask,
            other=0.0,
        )
        tl.store(
            x_ptr + t * x_stride_s + b * x_stride_b + offs_f * x_stride_f,
            vals,
            mask=f_mask,
        )


@libentry()
@triton.jit
def _pack_output_kernel(
    out_padded_ptr,
    out_packed_ptr,
    offsets_ptr,
    batch_sizes_ptr,
    hidden_size,
    batch_size,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    packed_stride_0,
    BLOCK_F: tl.constexpr,
):
    # Gather padded (num_steps, batch, hidden) output back into packed
    # (sum(batch_sizes), hidden). Only active rows are copied.
    pid = tl.program_id(0)
    t = pid // batch_size
    b = pid - t * batch_size
    bs_t = tl.load(batch_sizes_ptr + t).to(tl.int32)
    active = b < bs_t
    row = tl.load(offsets_ptr + t).to(tl.int32) + b
    for f_block in range(0, tl.cdiv(hidden_size, BLOCK_F)):
        offs_f = f_block * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = offs_f < hidden_size
        vals = tl.load(
            out_padded_ptr
            + t * out_stride_s
            + b * out_stride_b
            + offs_f * out_stride_f,
            mask=f_mask,
            other=0.0,
        )
        tl.store(
            out_packed_ptr + row * packed_stride_0 + offs_f,
            vals,
            mask=active & f_mask,
        )


def _ceil_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _block_size(value: int, maximum: int) -> int:
    return min(max(_ceil_power_of_2(value), 16), maximum)


def _validate_weight(tensor: torch.Tensor, rows: int, cols: int):
    if tensor.numel() != rows * cols:
        raise RuntimeError(
            f"invalid GRU weight with {tensor.numel()} elements, expected {rows * cols}"
        )
    if tensor.dim() == 2:
        if tensor.shape != (rows, cols):
            raise RuntimeError(
                f"invalid GRU weight shape {tuple(tensor.shape)}, expected {(rows, cols)}"
            )
    elif tensor.dim() != 1:
        raise RuntimeError(f"GRU weights must be 1-D or 2-D, got {tensor.dim()}-D")


def _bias_stride(tensor: torch.Tensor, size: int) -> int:
    if tensor.numel() != size:
        raise RuntimeError(
            f"invalid GRU bias with {tensor.numel()} elements, expected {size}"
        )
    if tensor.dim() == 1:
        return tensor.stride(0)
    raise RuntimeError(f"GRU bias tensors must be 1-D, got {tensor.dim()}-D")


def _param_group(params, index: int, has_biases: bool):
    # Per-direction layout is [w_ih, w_hh, b_ih, b_hh] / [w_ih, w_hh]: 4/2 params per
    # state (same as LSTM) despite GRU's 3 gates.
    group_size = 4 if has_biases else 2
    base = index * group_size
    if has_biases:
        return params[base], params[base + 1], params[base + 2], params[base + 3]
    return params[base], params[base + 1], params[base], params[base]


# Transposed (K, 3H) weight cache (saves two transpose launches per direction). Values
# keep a strong ref to the source (data_ptr isn't unique across freed storage).
_transposed_weight_cache: dict = {}


def _transpose_weight(weight: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    key = _tensor_cache_key(weight)
    cached = _transposed_weight_cache.get(key)
    if cached is not None:
        return cached[1]

    transposed = torch.empty((cols, rows), dtype=weight.dtype, device=weight.device)
    grid = (
        triton.cdiv(rows, _TRANSPOSE_BLOCK),
        triton.cdiv(cols, _TRANSPOSE_BLOCK),
    )
    with torch_device_fn.device(weight.device):
        _transpose_weight_kernel[grid](
            weight,
            transposed,
            rows,
            cols,
            BLOCK_R=_TRANSPOSE_BLOCK,
            BLOCK_C=_TRANSPOSE_BLOCK,
        )
    _transposed_weight_cache[key] = (weight, transposed)
    return transposed


def _validate_args(input, hx, params, has_biases, num_layers, dropout, bidirectional):
    if input.dim() != 3:
        raise RuntimeError("gru: input must have 3 dimensions")
    if num_layers <= 0:
        raise RuntimeError("gru: num_layers must be greater than zero")
    if not 0.0 <= dropout <= 1.0:
        raise RuntimeError("gru: dropout probability must be between 0 and 1")
    # fp16 only: torch_npu's GRU op (DynamicGRUV2) accepts fp16 alone, so no other dtype
    # has a reference to check against on this platform.
    if input.dtype != torch.float16:
        raise NotImplementedError("FlagGems gru on Ascend supports float16 only")

    if hx.dim() != 3:
        raise RuntimeError("gru: hidden state must have 3 dimensions")
    num_directions = 2 if bidirectional else 1
    expected_states = num_layers * num_directions
    if hx.shape[0] != expected_states:
        raise RuntimeError(
            f"gru: expected {expected_states} hidden state rows, got {hx.shape[0]}"
        )
    expected_params = expected_states * (4 if has_biases else 2)
    if len(params) != expected_params:
        raise RuntimeError(
            f"gru: expected {expected_params} parameter tensors, got {len(params)}"
        )
    if hx.device != input.device:
        raise RuntimeError("gru: input and hidden state must share a device")
    if hx.dtype != input.dtype:
        raise RuntimeError("gru: input and hidden state must share a dtype")


# Cache of augmented weight matrices, otherwise rebuilt once per direction per call.
# Values hold strong refs to the sources so no freed tensor can alias a key.
_w_aug_cache: dict = {}


def _tensor_cache_key(t: torch.Tensor):
    return (
        t.data_ptr(),
        t.storage_offset(),
        t.numel(),
        t.dtype,
        getattr(t, "_version", 0),
    )


@libentry()
@triton.jit
def _gru_input_gemm_kernel(
    x_ptr,
    w_ih_ptr,
    b_ih_ptr,
    b_hh_ptr,
    u_ptr,
    batch_sizes_ptr,
    input_size,
    hidden_size,
    batch_size,
    x_stride_s,
    x_stride_b,
    x_stride_f,
    w_ih_stride_r,
    w_ih_stride_c,
    b_ih_stride,
    b_hh_stride,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    PACKED: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    seq_idx = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_mask = offs_b < batch_size
    n_mask = offs_n < 3 * hidden_size

    if PACKED:
        # Packed input: skip the GEMM for fully-inactive batch tiles (rows b >=
        # batch_sizes[seq]) and zero them; the recurrence freezes those rows anyway.
        bs_t = tl.load(batch_sizes_ptr + seq_idx).to(tl.int32)
        if pid_b * BLOCK_B >= bs_t:
            out_offsets = (
                seq_idx * u_stride_s
                + offs_b[:, None] * u_stride_b
                + offs_n[None, :] * u_stride_f
            )
            tl.store(
                u_ptr + out_offsets,
                tl.zeros((BLOCK_B, BLOCK_N), dtype=COMPUTE_DTYPE),
                mask=b_mask[:, None] & n_mask[None, :],
            )
            return

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=COMPUTE_DTYPE)
    for k_block in range(0, tl.cdiv(input_size, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr
            + seq_idx * x_stride_s
            + offs_b[:, None] * x_stride_b
            + offs_k[None, :] * x_stride_f,
            mask=b_mask[:, None] & (offs_k[None, :] < input_size),
            other=0.0,
        )
        w = tl.load(
            w_ih_ptr
            + offs_k[:, None] * w_ih_stride_r
            + offs_n[None, :] * w_ih_stride_c,
            mask=(offs_k[:, None] < input_size) & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x, w, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

    if HAS_BIAS:
        # b_ih and the r/z half of b_hh are folded into u; b_hn must NOT be, it is
        # scaled by r inside the n gate's tanh.
        b_ih = tl.load(b_ih_ptr + offs_n * b_ih_stride, mask=n_mask, other=0.0)
        b_hh = tl.load(
            b_hh_ptr + offs_n * b_hh_stride,
            mask=n_mask & (offs_n < 2 * hidden_size),
            other=0.0,
        )
        acc += (b_ih + b_hh)[None, :]

    out_offsets = (
        seq_idx * u_stride_s
        + offs_b[:, None] * u_stride_b
        + offs_n[None, :] * u_stride_f
    )
    tl.store(u_ptr + out_offsets, acc, mask=b_mask[:, None] & n_mask[None, :])


@libentry()
@triton.jit
def _gru_x_aug_kernel(
    x_ptr,
    x_aug_ptr,
    cols,
    batch,
    x_stride_s,
    x_stride_b,
    x_stride_f,
    BLOCK: tl.constexpr,
):
    # x (seq, batch, cols) -> x_aug (seq*batch, cols+1) with a trailing ones column so
    # the mm folds w_aug's bias row in. Affine 3D grid: a div/mod variant faults the AIV.
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_c = pid_c * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(
        x_ptr + pid_s * x_stride_s + pid_b * x_stride_b + offs_c * x_stride_f,
        mask=offs_c < cols,
        other=1.0,
    )
    tl.store(
        x_aug_ptr + (pid_s * batch + pid_b) * (cols + 1) + offs_c,
        val,
        mask=offs_c <= cols,
    )


@libentry()
@triton.jit
def _gru_w_aug_kernel(
    w_ptr,
    b_ih_ptr,
    b_hh_ptr,
    w_aug_ptr,
    rows,
    cols3,
    hidden_size,
    w_stride_0,
    w_stride_1,
    b_ih_stride,
    b_hh_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Transposed weights (rows, cols3) -> w_aug (rows+1, cols3); the last row holds
    # b_ih plus the r/z half of b_hh (b_hn stays in the recurrence kernel).
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * BLOCK + tl.arange(0, BLOCK)
    w_val = tl.load(
        w_ptr + pid_r * w_stride_0 + offs_c * w_stride_1,
        mask=(pid_r < rows) & (offs_c < cols3),
        other=0.0,
    )
    if HAS_BIAS:
        bias_val = tl.load(
            b_ih_ptr + offs_c * b_ih_stride,
            mask=(pid_r >= rows) & (offs_c < cols3),
            other=0.0,
        )
        bias_val += tl.load(
            b_hh_ptr + offs_c * b_hh_stride,
            mask=(pid_r >= rows) & (offs_c < cols3) & (offs_c < 2 * hidden_size),
            other=0.0,
        )
    else:
        bias_val = tl.zeros((BLOCK,), dtype=w_val.dtype)
    tl.store(w_aug_ptr + pid_r * cols3 + offs_c, w_val + bias_val, mask=offs_c < cols3)


@libentry()
@triton.jit
def _gru_timeloop_kernel(
    u_ptr,
    hx_ptr,
    w_hh_ptr,
    b_hh_ptr,
    out_ptr,
    batch_sizes_ptr,
    out_feature_offset,
    hidden_size,
    seq_len,
    state_idx,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    w_hh_stride_r,
    w_hh_stride_c,
    b_hh_stride,
    hx_stride_s,
    hx_stride_b,
    hx_stride_f,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    final_h_ptr,
    final_h_stride_s,
    final_h_stride_b,
    final_h_stride_f,
    batch_size,
    HAS_BIAS: tl.constexpr,
    REVERSE: tl.constexpr,
    PACKED: tl.constexpr,
    DIVISIBLE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # Fused recurrence: ``u`` holds the input-side pre-activations for all timesteps
    # (b_hn excluded); each program owns BLOCK_B rows and the full hidden dim -> no
    # barrier. h_prev comes from out[prev_t]; weights re-load per step (hoisting segfaults).
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = offs_b < batch_size
    hx_base = hx_ptr + state_idx * hx_stride_s
    # Fill value for masked loads; None on the mask-free DIVISIBLE path
    fill = None if DIVISIBLE else 0.0

    # Step 0 (h_prev = the hx slice) is unrolled: the hx/out layout select has no safe
    # runtime form here (a scalar tl.where on pointers segfaults BiSheng).
    t = seq_len - 1 if REVERSE else 0
    for h_tile in range(0, tl.cdiv(hidden_size, BLOCK_H)):
        offs_h = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
        if DIVISIBLE:
            # All tiles are in-bounds: skip the bounds masks entirely so the
            # scalar pipe stops computing mask predicates for every load.
            h_mask = None
            bh_mask = None
        else:
            h_mask = offs_h < hidden_size
            bh_mask = b_mask[:, None] & h_mask[None, :]
        u_base = (
            t * u_stride_s + offs_b[:, None] * u_stride_b + offs_h[None, :] * u_stride_f
        )
        r_acc = tl.load(u_ptr + u_base, mask=bh_mask, other=fill).to(COMPUTE_DTYPE)
        z_acc = tl.load(
            u_ptr + u_base + hidden_size * u_stride_f, mask=bh_mask, other=fill
        ).to(COMPUTE_DTYPE)
        n_in = tl.load(
            u_ptr + u_base + 2 * hidden_size * u_stride_f,
            mask=bh_mask,
            other=fill,
        ).to(COMPUTE_DTYPE)
        n_h_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)

        for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
            offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            if DIVISIBLE:
                bk_mask = None
                kh_mask = None
            else:
                k_mask = offs_k < hidden_size
                bk_mask = b_mask[:, None] & k_mask[None, :]
                kh_mask = k_mask[:, None] & h_mask[None, :]
            h = tl.load(
                hx_base + offs_b[:, None] * hx_stride_b + offs_k[None, :] * hx_stride_f,
                mask=bk_mask,
                other=fill,
            )
            w_r = tl.load(
                w_hh_ptr
                + offs_k[:, None] * w_hh_stride_r
                + offs_h[None, :] * w_hh_stride_c,
                mask=kh_mask,
                other=fill,
            )
            w_z = tl.load(
                w_hh_ptr
                + offs_k[:, None] * w_hh_stride_r
                + (hidden_size + offs_h[None, :]) * w_hh_stride_c,
                mask=kh_mask,
                other=fill,
            )
            w_n = tl.load(
                w_hh_ptr
                + offs_k[:, None] * w_hh_stride_r
                + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c,
                mask=kh_mask,
                other=fill,
            )
            r_acc += tl.dot(h, w_r, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
            z_acc += tl.dot(h, w_z, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
            n_h_acc += tl.dot(h, w_n, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

        if HAS_BIAS:
            # b_hn must be added before the r multiply (see the igemm note).
            b_hn = tl.load(
                b_hh_ptr + (2 * hidden_size + offs_h) * b_hh_stride,
                mask=h_mask,
                other=fill,
            )
            n_h_acc += b_hn[None, :]

        r_gate = tl.sigmoid(r_acc)
        z_gate = tl.sigmoid(z_acc)
        n_gate = tl_extra_shim.tanh(n_in + r_gate * n_h_acc)

        h_prev = tl.load(
            hx_base + offs_b[:, None] * hx_stride_b + offs_h[None, :] * hx_stride_f,
            mask=bh_mask,
            other=fill,
        ).to(COMPUTE_DTYPE)
        h_next = (1.0 - z_gate) * n_gate + z_gate * h_prev

        if PACKED:
            # Rows past the end of their sequence (b >= batch_sizes[t]) must not
            # advance: freeze them at h_prev (arithmetic, not tl.where).
            active = offs_b < tl.load(batch_sizes_ptr + t).to(tl.int32)
            h_next = h_prev + active[:, None].to(COMPUTE_DTYPE) * (h_next - h_prev)

        out_offsets = (
            t * out_stride_s
            + offs_b[:, None] * out_stride_b
            + (out_feature_offset + offs_h[None, :]) * out_stride_f
        )
        tl.store(out_ptr + out_offsets, h_next, mask=bh_mask)

    # Steps 1..seq_len-1: h_prev comes from out[prev_t].
    for step in range(1, seq_len):
        t = seq_len - 1 - step if REVERSE else step
        prev_t = t - 1 if not REVERSE else t + 1
        h_prev_base = out_ptr + prev_t * out_stride_s
        for h_tile in range(0, tl.cdiv(hidden_size, BLOCK_H)):
            offs_h = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
            if DIVISIBLE:
                h_mask = None
                bh_mask = None
            else:
                h_mask = offs_h < hidden_size
                bh_mask = b_mask[:, None] & h_mask[None, :]
            u_base = (
                t * u_stride_s
                + offs_b[:, None] * u_stride_b
                + offs_h[None, :] * u_stride_f
            )
            r_acc = tl.load(u_ptr + u_base, mask=bh_mask, other=fill).to(COMPUTE_DTYPE)
            z_acc = tl.load(
                u_ptr + u_base + hidden_size * u_stride_f, mask=bh_mask, other=fill
            ).to(COMPUTE_DTYPE)
            n_in = tl.load(
                u_ptr + u_base + 2 * hidden_size * u_stride_f,
                mask=bh_mask,
                other=fill,
            ).to(COMPUTE_DTYPE)
            n_h_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)

            for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
                offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
                if DIVISIBLE:
                    bk_mask = None
                    kh_mask = None
                else:
                    k_mask = offs_k < hidden_size
                    bk_mask = b_mask[:, None] & k_mask[None, :]
                    kh_mask = k_mask[:, None] & h_mask[None, :]
                h = tl.load(
                    h_prev_base
                    + offs_b[:, None] * out_stride_b
                    + (out_feature_offset + offs_k[None, :]) * out_stride_f,
                    mask=bk_mask,
                    other=fill,
                )
                w_r = tl.load(
                    w_hh_ptr
                    + offs_k[:, None] * w_hh_stride_r
                    + offs_h[None, :] * w_hh_stride_c,
                    mask=kh_mask,
                    other=fill,
                )
                w_z = tl.load(
                    w_hh_ptr
                    + offs_k[:, None] * w_hh_stride_r
                    + (hidden_size + offs_h[None, :]) * w_hh_stride_c,
                    mask=kh_mask,
                    other=fill,
                )
                w_n = tl.load(
                    w_hh_ptr
                    + offs_k[:, None] * w_hh_stride_r
                    + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c,
                    mask=kh_mask,
                    other=fill,
                )
                r_acc += tl.dot(h, w_r, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
                z_acc += tl.dot(h, w_z, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
                n_h_acc += tl.dot(h, w_n, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

            if HAS_BIAS:
                b_hn = tl.load(
                    b_hh_ptr + (2 * hidden_size + offs_h) * b_hh_stride,
                    mask=h_mask,
                    other=fill,
                )
                n_h_acc += b_hn[None, :]

            r_gate = tl.sigmoid(r_acc)
            z_gate = tl.sigmoid(z_acc)
            n_gate = tl_extra_shim.tanh(n_in + r_gate * n_h_acc)

            h_prev = tl.load(
                h_prev_base
                + offs_b[:, None] * out_stride_b
                + (out_feature_offset + offs_h[None, :]) * out_stride_f,
                mask=bh_mask,
                other=fill,
            ).to(COMPUTE_DTYPE)
            h_next = (1.0 - z_gate) * n_gate + z_gate * h_prev

            if PACKED:
                active = offs_b < tl.load(batch_sizes_ptr + t).to(tl.int32)
                h_next = h_prev + active[:, None].to(COMPUTE_DTYPE) * (h_next - h_prev)

            out_offsets = (
                t * out_stride_s
                + offs_b[:, None] * out_stride_b
                + (out_feature_offset + offs_h[None, :]) * out_stride_f
            )
            tl.store(out_ptr + out_offsets, h_next, mask=bh_mask)

    # Epilogue: the last computed timestep (t=0 for REVERSE). Each program re-reads only
    # its own rows, saving the separate host-side copy kernel (~14us/program).
    last_t = 0 if REVERSE else seq_len - 1
    for h_tile in range(0, tl.cdiv(hidden_size, BLOCK_H)):
        offs_h = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
        if DIVISIBLE:
            bh_mask = None
        else:
            bh_mask = b_mask[:, None] & (offs_h < hidden_size)[None, :]
        h_val = tl.load(
            out_ptr
            + last_t * out_stride_s
            + offs_b[:, None] * out_stride_b
            + (out_feature_offset + offs_h[None, :]) * out_stride_f,
            mask=bh_mask,
            other=fill,
        )
        tl.store(
            final_h_ptr
            + state_idx * final_h_stride_s
            + offs_b[:, None] * final_h_stride_b
            + offs_h[None, :] * final_h_stride_f,
            h_val,
            mask=bh_mask,
        )


@libentry()
@triton.jit
def _gru_timeloop_hsplit_kernel(
    u_ptr,
    hx_ptr,
    w_hh_ptr,
    b_hh_ptr,
    out_ptr,
    batch_sizes_ptr,
    out_feature_offset,
    hidden_size,
    seq_len,
    state_idx,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    w_hh_stride_r,
    w_hh_stride_c,
    b_hh_stride,
    hx_stride_s,
    hx_stride_b,
    hx_stride_f,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    final_h_ptr,
    final_h_stride_s,
    final_h_stride_b,
    final_h_stride_f,
    HAS_BIAS: tl.constexpr,
    REVERSE: tl.constexpr,
    PACKED: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    # This program's hidden columns of the gate output, and the slice of h_prev it
    # owns and writes back.
    offs_h = pid_n * BLOCK_H + tl.arange(0, BLOCK_H)
    hx_base = hx_ptr + state_idx * hx_stride_s

    # Step 0 (h_prev = the hx slice) is unrolled for the same reason as in the
    # barrier-free kernel: the hx/out layout select has no safe runtime form here.
    t = seq_len - 1 if REVERSE else 0
    u_base = (
        t * u_stride_s + offs_b[:, None] * u_stride_b + offs_h[None, :] * u_stride_f
    )
    r_acc = tl.load(u_ptr + u_base).to(COMPUTE_DTYPE)
    z_acc = tl.load(u_ptr + u_base + hidden_size * u_stride_f).to(COMPUTE_DTYPE)
    n_in = tl.load(u_ptr + u_base + 2 * hidden_size * u_stride_f).to(COMPUTE_DTYPE)
    n_h_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)

    for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        # Full hidden range: every program contracts over all of h_prev.
        h = tl.load(
            hx_base + offs_b[:, None] * hx_stride_b + offs_k[None, :] * hx_stride_f
        )
        w_r = tl.load(
            w_hh_ptr + offs_k[:, None] * w_hh_stride_r + offs_h[None, :] * w_hh_stride_c
        )
        w_z = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (hidden_size + offs_h[None, :]) * w_hh_stride_c
        )
        w_n = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c
        )
        r_acc += tl.dot(h, w_r, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
        z_acc += tl.dot(h, w_z, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
        n_h_acc += tl.dot(h, w_n, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

    if HAS_BIAS:
        n_h_acc += tl.load(b_hh_ptr + (2 * hidden_size + offs_h) * b_hh_stride)[None, :]

    r_gate = tl.sigmoid(r_acc)
    z_gate = tl.sigmoid(z_acc)
    n_gate = tl_extra_shim.tanh(n_in + r_gate * n_h_acc)

    h_prev = tl.load(
        hx_base + offs_b[:, None] * hx_stride_b + offs_h[None, :] * hx_stride_f
    ).to(COMPUTE_DTYPE)
    h_next = (1.0 - z_gate) * n_gate + z_gate * h_prev

    out_offsets = (
        t * out_stride_s
        + offs_b[:, None] * out_stride_b
        + (out_feature_offset + offs_h[None, :]) * out_stride_f
    )
    if PACKED:
        # The freeze depends on the batch row only, so each program applies it to its
        # own columns. Two masked stores rather than the arithmetic form the other
        # kernels use: a broadcast fp32 mask would not fit this kernel's UB budget.
        bs_t = tl.load(batch_sizes_ptr + t).to(tl.int32)
        tl.store(out_ptr + out_offsets, h_next, mask=(offs_b < bs_t)[:, None])
        tl.store(out_ptr + out_offsets, h_prev, mask=(offs_b >= bs_t)[:, None])
    else:
        tl.store(out_ptr + out_offsets, h_next)
    # Publish this program's h slice and wait for the others: the next step reads
    # h_prev across the full hidden range.
    sync_block_all("all", 0)

    for step in range(1, seq_len):
        t = seq_len - 1 - step if REVERSE else step
        prev_t = t - 1 if not REVERSE else t + 1
        h_prev_base = out_ptr + prev_t * out_stride_s

        u_base = (
            t * u_stride_s + offs_b[:, None] * u_stride_b + offs_h[None, :] * u_stride_f
        )
        r_acc = tl.load(u_ptr + u_base).to(COMPUTE_DTYPE)
        z_acc = tl.load(u_ptr + u_base + hidden_size * u_stride_f).to(COMPUTE_DTYPE)
        n_in = tl.load(u_ptr + u_base + 2 * hidden_size * u_stride_f).to(COMPUTE_DTYPE)
        n_h_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)

        for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
            offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            h = tl.load(
                h_prev_base
                + offs_b[:, None] * out_stride_b
                + (out_feature_offset + offs_k[None, :]) * out_stride_f
            )
            w_r = tl.load(
                w_hh_ptr
                + offs_k[:, None] * w_hh_stride_r
                + offs_h[None, :] * w_hh_stride_c
            )
            w_z = tl.load(
                w_hh_ptr
                + offs_k[:, None] * w_hh_stride_r
                + (hidden_size + offs_h[None, :]) * w_hh_stride_c
            )
            w_n = tl.load(
                w_hh_ptr
                + offs_k[:, None] * w_hh_stride_r
                + (2 * hidden_size + offs_h[None, :]) * w_hh_stride_c
            )
            r_acc += tl.dot(h, w_r, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
            z_acc += tl.dot(h, w_z, out_dtype=COMPUTE_DTYPE, allow_tf32=False)
            n_h_acc += tl.dot(h, w_n, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

        if HAS_BIAS:
            n_h_acc += tl.load(b_hh_ptr + (2 * hidden_size + offs_h) * b_hh_stride)[
                None, :
            ]

        r_gate = tl.sigmoid(r_acc)
        z_gate = tl.sigmoid(z_acc)
        n_gate = tl_extra_shim.tanh(n_in + r_gate * n_h_acc)

        h_prev = tl.load(
            h_prev_base
            + offs_b[:, None] * out_stride_b
            + (out_feature_offset + offs_h[None, :]) * out_stride_f
        ).to(COMPUTE_DTYPE)
        h_next = (1.0 - z_gate) * n_gate + z_gate * h_prev

        out_offsets = (
            t * out_stride_s
            + offs_b[:, None] * out_stride_b
            + (out_feature_offset + offs_h[None, :]) * out_stride_f
        )
        if PACKED:
            bs_t = tl.load(batch_sizes_ptr + t).to(tl.int32)
            tl.store(out_ptr + out_offsets, h_next, mask=(offs_b < bs_t)[:, None])
            tl.store(out_ptr + out_offsets, h_prev, mask=(offs_b >= bs_t)[:, None])
        else:
            tl.store(out_ptr + out_offsets, h_next)
        sync_block_all("all", 0)

    # Epilogue: same as the barrier-free kernel; each program re-reads only its own
    # columns, so still no barrier.
    last_t = 0 if REVERSE else seq_len - 1
    h_val = tl.load(
        out_ptr
        + last_t * out_stride_s
        + offs_b[:, None] * out_stride_b
        + (out_feature_offset + offs_h[None, :]) * out_stride_f
    )
    tl.store(
        final_h_ptr
        + state_idx * final_h_stride_s
        + offs_b[:, None] * final_h_stride_b
        + offs_h[None, :] * final_h_stride_f,
        h_val,
    )


@libentry()
@triton.jit
def _gru_unpack_padded_shifted_kernel(
    data_ptr,
    x_ptr,
    offsets_ptr,
    batch_sizes_ptr,
    row_shift,
    input_size,
    batch_size,
    data_stride_0,
    x_stride_s,
    x_stride_b,
    x_stride_f,
    BLOCK_F: tl.constexpr,
):
    # Same gather as _unpack_padded_kernel but reading rows row_shift + offsets[t] + b:
    # ATen's packed reverse flips the whole buffer, so a wider one consumes shifted rows.
    pid = tl.program_id(0)
    t = pid // batch_size
    b = pid - t * batch_size
    bs_t = tl.load(batch_sizes_ptr + t).to(tl.int32)
    active = b < bs_t
    row = row_shift + tl.load(offsets_ptr + t).to(tl.int32) + b
    for f_block in range(0, tl.cdiv(input_size, BLOCK_F)):
        offs_f = f_block * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = offs_f < input_size
        vals = tl.load(
            data_ptr + row * data_stride_0 + offs_f,
            mask=active & f_mask,
            other=0.0,
        )
        tl.store(
            x_ptr + t * x_stride_s + b * x_stride_b + offs_f * x_stride_f,
            vals,
            mask=f_mask,
        )


def _cube_core_num(device) -> int:
    props = torch_device_fn.get_device_properties(device)
    cores = getattr(props, "cube_core_num", None)
    if cores is None:
        cores = getattr(props, "multi_processor_count", None) or 16
    return cores


def _timeloop_block_b(batch_size: int, device) -> int:
    # Multi-wave bug: an in-kernel time loop returns wrong results when the grid exceeds
    # the AI core count (bit-exact <= 20 on 910B4, garbage >= 21 for seq_len >= 2).
    cores = triton.cdiv(batch_size, _cube_core_num(device))
    # Taking the _BLOCK_B floor for a smaller batch would pad every program AND make
    # batch_size % block_b != 0, turning DIVISIBLE off. A power-of-two batch that fills a
    # block exactly sizes the block to the batch: 30.9ms -> 20.9ms at (8,1000,80,512).
    if (
        _MIN_TIMELOOP_BLOCK_B <= batch_size < _BLOCK_B
        and (batch_size & (batch_size - 1)) == 0
    ):
        return max(batch_size, cores)
    return max(_BLOCK_B, cores)


def _hsplit_tiles(batch_size: int, hidden_size: int, device):
    if sync_block_all is None:
        return None
    block_h = _block_size(hidden_size, _BLOCK_H_MAX)
    if hidden_size % block_h != 0:
        return None
    n_n = hidden_size // block_h
    if n_n < 2:
        return None
    cores = _cube_core_num(device)
    floor = _timeloop_block_b(batch_size, device)
    for block_b in sorted({b for b in (floor, 32, 64) if b <= _HSPLIT_BLOCK_B_MAX}):
        n_b = batch_size // block_b
        if batch_size % block_b == 0 and n_b * n_n <= cores:
            return block_b, (n_b, n_n)
    return None


def _direction_prep(
    layer_input,
    params,
    state_idx,
    input_size,
    hidden_size,
    batch_size,
    seq_len,
    has_biases,
    batch_sizes,
):
    w_ih, w_hh, b_ih, b_hh = _param_group(params, state_idx, has_biases)
    _validate_weight(w_ih, 3 * hidden_size, input_size)
    _validate_weight(w_hh, 3 * hidden_size, hidden_size)
    if w_ih.dim() == 1:
        w_ih = w_ih.view(3 * hidden_size, input_size)
    if w_hh.dim() == 1:
        w_hh = w_hh.view(3 * hidden_size, hidden_size)
    # Keyed on the sources before transposition, so a hit skips both the transpose and
    # the w_aug build. has_biases is in the key: with no biases, _param_group returns
    # the weights themselves as the "bias" slots.
    w_aug_key = (
        _tensor_cache_key(w_ih),
        _tensor_cache_key(b_ih) if has_biases else None,
        _tensor_cache_key(b_hh) if has_biases else None,
        has_biases,
        input_size,
        hidden_size,
    )
    w_ih = _transpose_weight(w_ih, 3 * hidden_size, input_size)
    w_hh = _transpose_weight(w_hh, 3 * hidden_size, hidden_size)
    w_ih_stride_r, w_ih_stride_c = w_ih.stride(0), w_ih.stride(1)
    w_hh_stride_r, w_hh_stride_c = w_hh.stride(0), w_hh.stride(1)
    b_ih_stride = _bias_stride(b_ih, 3 * hidden_size) if has_biases else 1
    b_hh_stride = _bias_stride(b_hh, 3 * hidden_size) if has_biases else 1

    compute_dtype = tl.float32
    gate_dtype = torch.float16

    # Precompute the input-side pre-activations for every timestep in one batched GEMM.
    input_gates = torch.empty(
        (seq_len, batch_size, 3 * hidden_size),
        dtype=gate_dtype,
        device=layer_input.device,
    )
    input_gemm_grid = (
        triton.cdiv(batch_size, _IGEMM_BLOCK_B),
        seq_len,
        triton.cdiv(3 * hidden_size, _IGEMM_BLOCK_N),
    )

    macs = seq_len * batch_size * input_size * 3 * hidden_size
    use_mm = macs >= _IGEMM_MM_MIN_MACS
    with torch_device_fn.device(layer_input.device):
        if use_mm:
            m = seq_len * batch_size
            x_aug = torch.empty(
                (m, input_size + 1), dtype=gate_dtype, device=layer_input.device
            )
            cached = _w_aug_cache.get(w_aug_key)
            if cached is None:
                w_aug = torch.empty(
                    (input_size + 1, 3 * hidden_size),
                    dtype=gate_dtype,
                    device=layer_input.device,
                )
                _gru_w_aug_kernel[
                    (input_size + 1, triton.cdiv(3 * hidden_size, _PACK_BLOCK))
                ](
                    *(
                        w_ih,
                        b_ih,
                        b_hh,
                        w_aug,
                        input_size,
                        3 * hidden_size,
                        hidden_size,
                        w_ih_stride_r,
                        w_ih_stride_c,
                        b_ih_stride,
                        b_hh_stride,
                    ),
                    HAS_BIAS=has_biases,
                    BLOCK=_PACK_BLOCK,
                )
                # Strong refs keep the sources alive so no freed tensor can alias a key.
                _w_aug_cache[w_aug_key] = (w_ih, b_ih, b_hh, w_aug)
            else:
                w_aug = cached[3]
            _gru_x_aug_kernel[
                (seq_len, batch_size, triton.cdiv(input_size + 1, _PACK_BLOCK))
            ](
                *(
                    layer_input,
                    x_aug,
                    input_size,
                    batch_size,
                    layer_input.stride(0),
                    layer_input.stride(1),
                    layer_input.stride(2),
                ),
                BLOCK=_PACK_BLOCK,
            )
            # The out must be the 2D view: Ascend mm_out derives M/N from out.shape
            # and miscomputes for a 3D out tensor.
            flag_gems.mm_out(x_aug, w_aug, out=input_gates.view(m, 3 * hidden_size))
        else:
            _gru_input_gemm_kernel[input_gemm_grid](
                *(
                    layer_input,
                    w_ih,
                    b_ih,
                    b_hh,
                    input_gates,
                    batch_sizes if batch_sizes is not None else input_gates,
                    input_size,
                    hidden_size,
                    batch_size,
                    layer_input.stride(0),
                    layer_input.stride(1),
                    layer_input.stride(2),
                    w_ih_stride_r,
                    w_ih_stride_c,
                    b_ih_stride,
                    b_hh_stride,
                    input_gates.stride(0),
                    input_gates.stride(1),
                    input_gates.stride(2),
                ),
                PACKED=batch_sizes is not None,
                HAS_BIAS=has_biases,
                BLOCK_B=_IGEMM_BLOCK_B,
                BLOCK_N=_IGEMM_BLOCK_N,
                BLOCK_K=_IGEMM_BLOCK_K,
                COMPUTE_DTYPE=compute_dtype,
                num_warps=_IGEMM_NUM_WARPS,
                num_stages=_IGEMM_NUM_STAGES,
            )

    return {
        "input_gates": input_gates,
        "w_hh": w_hh,
        "b_hh": b_hh,
        "w_hh_stride_r": w_hh_stride_r,
        "w_hh_stride_c": w_hh_stride_c,
        "b_hh_stride": b_hh_stride,
        "compute_dtype": compute_dtype,
    }


def _run_direction(
    layer_input,
    hx,
    layer_output,
    final_h,
    params,
    state_idx: int,
    out_feature_offset: int,
    input_size: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    has_biases: bool,
    reverse: bool,
    batch_sizes=None,
):
    # Ascend direction runner: two kernels per direction instead of seq_len + 1 (the
    # input-side GEMM and the fused time-loop kernel); weight transposes are cached.
    if batch_size == 0:
        return
    prep = _direction_prep(
        layer_input,
        params,
        state_idx,
        input_size,
        hidden_size,
        batch_size,
        seq_len,
        has_biases,
        batch_sizes,
    )
    input_gates = prep["input_gates"]
    w_hh = prep["w_hh"]
    b_hh = prep["b_hh"]
    w_hh_stride_r = prep["w_hh_stride_r"]
    w_hh_stride_c = prep["w_hh_stride_c"]
    b_hh_stride = prep["b_hh_stride"]
    compute_dtype = prep["compute_dtype"]

    # BLOCK_H and BLOCK_K are the same width here (unlike the generic file), so one
    # divisibility check covers both tiles - the mask-free kernels rely on that.
    block_h = _block_size(hidden_size, _BLOCK_H_MAX)
    block_b = _timeloop_block_b(batch_size, layer_input.device)
    grid = (triton.cdiv(batch_size, block_b),)
    divisible = batch_size % block_b == 0 and hidden_size % block_h == 0
    hsplit = _hsplit_tiles(batch_size, hidden_size, layer_input.device)

    # Tail of the timeloop kernels' argument list, identical for both variants.
    tail_args = (
        input_gates.stride(0),
        input_gates.stride(1),
        input_gates.stride(2),
        w_hh_stride_r,
        w_hh_stride_c,
        b_hh_stride,
        hx.stride(0),
        hx.stride(1),
        hx.stride(2),
        layer_output.stride(0),
        layer_output.stride(1),
        layer_output.stride(2),
        final_h,
        final_h.stride(0),
        final_h.stride(1),
        final_h.stride(2),
    )
    with torch_device_fn.device(layer_input.device):
        timeloop_args = (
            input_gates,
            hx,
            w_hh,
            b_hh,
            layer_output,
            batch_sizes if batch_sizes is not None else input_gates,
            out_feature_offset,
            hidden_size,
            seq_len,
            state_idx,
        ) + tail_args
        timeloop_kwargs = dict(
            HAS_BIAS=has_biases,
            REVERSE=reverse,
            PACKED=batch_sizes is not None,
            BLOCK_H=block_h,
            BLOCK_K=block_h,
            COMPUTE_DTYPE=compute_dtype,
        )
        if hsplit is not None:
            hs_block_b, hs_grid = hsplit
            _gru_timeloop_hsplit_kernel[hs_grid](
                *timeloop_args, BLOCK_B=hs_block_b, **timeloop_kwargs
            )
        else:
            # batch_size is only read by this kernel: the h-split variant requires
            # divisible tiles, so its rows are always in-bounds.
            _gru_timeloop_kernel[grid](
                *timeloop_args,
                batch_size,
                BLOCK_B=block_b,
                DIVISIBLE=divisible,
                **timeloop_kwargs,
            )


def _gru_forward_impl(
    input_view,
    hx,
    params,
    output,
    final_h,
    num_layers,
    num_directions,
    hidden_size,
    input_size,
    batch_size,
    seq_len,
    has_biases,
    train,
    dropout,
    batch_sizes=None,
    layer_inputs=None,
):
    layer_input = input_view
    for layer in range(num_layers):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions
        if layer == num_layers - 1:
            layer_output = output
        else:
            layer_output = torch.empty(
                (seq_len, batch_size, hidden_size * num_directions),
                dtype=input_view.dtype,
                device=input_view.device,
            )
        for direction in range(num_directions):
            state_idx = layer * num_directions + direction
            reverse = direction == 1
            direction_input = layer_input
            if layer_inputs is not None:
                direction_input = layer_inputs.get((layer, direction), layer_input)
            _run_direction(
                direction_input,
                hx,
                layer_output,
                final_h,
                params,
                state_idx,
                direction * hidden_size,
                layer_input_size,
                hidden_size,
                batch_size,
                seq_len,
                has_biases,
                reverse,
                batch_sizes,
            )

        layer_input = layer_output
        if train and dropout != 0.0 and layer + 1 < num_layers:
            layer_input, _ = _dropout(layer_input, dropout, True)

    return layer_input


def gru(
    input,
    hx,
    params,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
    batch_first=False,
):
    logger.debug("GEMS_ASCEND GRU")
    _validate_args(input, hx, params, has_biases, num_layers, dropout, bidirectional)

    if batch_first:
        batch_size, seq_len, input_size = input.shape
        input_view = input.transpose(0, 1)
    else:
        seq_len, batch_size, input_size = input.shape
        input_view = input
    if seq_len == 0:
        raise RuntimeError("Expected sequence length to be larger than 0 in RNN")

    hidden_size = hx.shape[2]
    num_directions = 2 if bidirectional else 1

    final_h = torch.empty(
        (num_layers * num_directions, batch_size, hidden_size),
        dtype=input.dtype,
        device=input.device,
    )
    output_tf = torch.empty(
        (seq_len, batch_size, hidden_size * num_directions),
        dtype=input.dtype,
        device=input.device,
    )
    _gru_forward_impl(
        input_view,
        hx,
        params,
        output_tf,
        final_h,
        num_layers,
        num_directions,
        hidden_size,
        input_size,
        batch_size,
        seq_len,
        has_biases,
        train,
        dropout,
    )

    output = output_tf.transpose(0, 1) if batch_first else output_tf
    return output, final_h


def gru_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
):
    logger.debug("GEMS_ASCEND GRU_DATA")
    if data.dtype != torch.float16:
        raise NotImplementedError("FlagGems gru.data on Ascend supports float16 only")
    if data.dim() != 2:
        raise RuntimeError("gru.data: packed data must have 2 dimensions")
    if batch_sizes.dim() != 1:
        raise RuntimeError("gru.data: batch_sizes must be 1-dimensional")
    if num_layers <= 0:
        raise RuntimeError("gru.data: num_layers must be greater than zero")
    if not 0.0 <= dropout <= 1.0:
        raise RuntimeError("gru.data: dropout probability must be between 0 and 1")
    if hx.dim() != 3:
        raise RuntimeError("gru.data: hidden state must have 3 dimensions")
    num_directions = 2 if bidirectional else 1
    num_states = num_layers * num_directions
    if hx.shape[0] != num_states:
        raise RuntimeError(
            f"gru.data: expected {num_states} hidden state rows, got {hx.shape[0]}"
        )
    expected_params = num_states * (4 if has_biases else 2)
    if len(params) != expected_params:
        raise RuntimeError(
            f"gru.data: expected {expected_params} parameter tensors, got {len(params)}"
        )
    if hx.device != data.device:
        raise RuntimeError("gru.data: data and hidden state must share a device")
    if hx.dtype != data.dtype:
        raise RuntimeError("gru.data: data and hidden state must share a dtype")

    num_steps = batch_sizes.numel()
    input_size = data.shape[1]
    batch = hx.shape[1]
    hidden_size = hx.shape[2]
    hidden_total = hidden_size * num_directions
    num_packed_rows = int(batch_sizes.sum().item())

    # pack_padded_sequence produces batch_sizes on CPU; the kernels need it on the
    # data's device (no-op when already resident).
    batch_sizes = batch_sizes.to(data.device)

    # Exclusive prefix-sum plus an int32 copy for the recurrence mask, in a dedicated
    # kernel: torch.cumsum/sub dispatch crashes on the packed path under use_gems().
    offsets = torch.empty((num_steps,), dtype=torch.int32, device=data.device)
    bs32 = torch.empty((num_steps,), dtype=torch.int32, device=data.device)
    with torch_device_fn.device(data.device):
        _batch_offsets_kernel[(1,)](
            batch_sizes,
            offsets,
            bs32,
            num_steps,
            BLOCK=_ceil_power_of_2(num_steps),
        )

    # Gather the packed input into a zero-padded tensor so the batched recurrence can
    # be reused unchanged; padding rows are zeros.
    x_padded = torch.empty(
        (num_steps, batch, input_size), dtype=data.dtype, device=data.device
    )
    with torch_device_fn.device(data.device):
        _unpack_padded_kernel[(num_steps * batch,)](
            data,
            x_padded,
            offsets,
            bs32,
            input_size,
            batch,
            data.stride(0),
            x_padded.stride(0),
            x_padded.stride(1),
            x_padded.stride(2),
            BLOCK_F=_PACK_BLOCK,
        )

    # ATen's packed reverse flips the whole buffer, so on a wider one it consumes rows the
    # forward half never sees. Gather that copy once, for layer 0 direction 1 only.
    layer_inputs = None
    if num_directions == 2:
        row_shift = max(data.shape[0] - num_packed_rows, 0)
        if row_shift == 0:
            reverse_input = x_padded
        else:
            reverse_input = torch.empty(
                (num_steps, batch, input_size), dtype=data.dtype, device=data.device
            )
            with torch_device_fn.device(data.device):
                _gru_unpack_padded_shifted_kernel[(num_steps * batch,)](
                    data,
                    reverse_input,
                    offsets,
                    bs32,
                    row_shift,
                    input_size,
                    batch,
                    data.stride(0),
                    reverse_input.stride(0),
                    reverse_input.stride(1),
                    reverse_input.stride(2),
                    BLOCK_F=_PACK_BLOCK,
                )
        layer_inputs = {(0, 1): reverse_input}

    final_h = torch.empty(
        (num_states, batch, hidden_size), dtype=data.dtype, device=data.device
    )
    out_padded = torch.empty(
        (num_steps, batch, hidden_total), dtype=data.dtype, device=data.device
    )
    _gru_forward_impl(
        x_padded,
        hx,
        params,
        out_padded,
        final_h,
        num_layers,
        num_directions,
        hidden_size,
        input_size,
        batch,
        num_steps,
        has_biases,
        train,
        dropout,
        batch_sizes=bs32,
        layer_inputs=layer_inputs,
    )

    # Pack the padded output back into the (sum(batch_sizes), hidden_total) layout;
    # bidirectional rows carry the concatenated [forward | reverse] hidden states.
    out_packed = torch.empty(
        (num_packed_rows, hidden_total), dtype=data.dtype, device=data.device
    )
    with torch_device_fn.device(data.device):
        _pack_output_kernel[(num_steps * batch,)](
            out_padded,
            out_packed,
            offsets,
            bs32,
            hidden_total,
            batch,
            out_padded.stride(0),
            out_padded.stride(1),
            out_padded.stride(2),
            out_packed.stride(0),
            BLOCK_F=_PACK_BLOCK,
        )

    return out_packed, final_h
