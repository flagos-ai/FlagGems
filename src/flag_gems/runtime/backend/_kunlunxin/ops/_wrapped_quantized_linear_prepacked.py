import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._wrapped_linear_prepack import unpack_linear_weight
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_DOT_MIN = 16  # tl.dot / tl.arange floor on XPU3.


@triton.jit
def _rne(x):
    """Round to nearest, ties to even, for fp32 ``|x| < 2**22`` (pure adds)."""
    return (x + 12582912.0) - 12582912.0


@libentry()
@triton.jit
def _wqlp_quant_flat_kernel(
    input,
    input_scale,
    input_zero_point,
    s_out,
    BLOCK: tl.constexpr,
):
    """Flat contiguous quantize: ``s_out[i] = clamp(rne(input[i]/scale)+zp,0,255)-zp``.

    Used when the input is contiguous and ``Kp == K`` so input and output share
    the same flat layout.  ``total`` is a multiple of ``BLOCK`` (chosen on the
    host) so there is no tail clamp -- a clamped index degrades the load to a
    data-dependent gather and is ~40x slower (measured).
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    input_scale_v = tl.load(input_scale).to(tl.float32)
    input_zp_f = tl.load(input_zero_point).to(tl.float32)
    a = tl.load(input + off).to(tl.float32)
    q = _rne(a / input_scale_v) + input_zp_f
    q = tl.minimum(tl.maximum(q, 0.0), 255.0)
    tl.store(s_out + off, (q - input_zp_f).to(tl.float16))


@libentry()
@triton.jit
def _wqlp_quant_kernel(
    input,
    input_scale,
    input_zero_point,
    s_out,
    K,
    Kp,
    stride_im,
    stride_ik,
    BLOCK_K: tl.constexpr,
    RAGGED: tl.constexpr,
):
    """One row per program: ``s_out[row, col] = clamp(rne(in/scale)+zp,0,255)-zp``.

    Fully **affine**: grid row == ``row`` (no row clamping/gather) and the
    program covers ``BLOCK_K`` contiguous columns.  In the common case
    (``Kp == K``) the column read is affine too; only ragged ``K`` clamps the
    out-of-range tail to ``K-1`` and value-masks it to zero.  A ``mask=``/``other``
    load is avoided on purpose -- masked loads return garbage (NaN) in masked
    lanes on XPU3 (KERNEL_OPT rows 25/51).
    """
    row = tl.program_id(0)
    pid_k = tl.program_id(1)
    ck = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    input_scale_v = tl.load(input_scale).to(tl.float32)
    input_zp_f = tl.load(input_zero_point).to(tl.float32)
    if RAGGED:
        ckc = tl.minimum(ck, K - 1)
    else:
        ckc = ck
    a = tl.load(input + row * stride_im + ckc * stride_ik).to(tl.float32)
    q = _rne(a / input_scale_v) + input_zp_f
    q = tl.minimum(tl.maximum(q, 0.0), 255.0)
    s = q - input_zp_f
    if RAGGED:
        s = tl.where(ck < K, s, 0.0)
    tl.store(s_out + row * Kp + ck, s.to(tl.float16))


@libentry()
@triton.jit
def _wqlp_kernel(
    s_in,
    weight,
    bias,
    input_scale,
    weight_metadata,
    output_scale,
    output_zero_point,
    output,
    M,
    N,
    K,
    Kp,
    stride_sm,
    stride_wn,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RAGGED: tl.constexpr,
):
    """fp16 dot + float epilogue.  No reductions, no address masks.

    ``s`` tail (``k >= K``) is zero so ragged-K padding contributes exactly
    zero.  In the common (non-ragged, ``Kp == K``) case both operands are read
    with fully **affine** strides; only the ragged case clamps the weight read
    to ``K-1``.  A ``min(...)`` clamp in the hot loop degrades the load to a
    data-dependent gather (~50x slower, measured 142 ms), hence the constexpr
    branch.  The B operand is loaded as ``(BLOCK_N, BLOCK_K)`` and transposed
    (a direct ``(BLOCK_K, BLOCK_N)`` B tile is miscompiled).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = rm % M
    rbn = rn % N
    rk = tl.arange(0, BLOCK_K)

    weight_zp16 = tl.load(weight_metadata + 1).to(tl.float16)
    s_base = s_in + ram[:, None] * stride_sm
    w_base = weight + rbn[:, None] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, Kp, BLOCK_K):
        k = k0 + rk
        if RAGGED:
            kc = tl.minimum(k, K - 1)
        else:
            kc = k
        s = tl.load(s_base + k[None, :])
        w = tl.load(w_base + kc[None, :]).to(tl.float16) - weight_zp16
        acc += tl.dot(s, tl.trans(w), out_dtype=tl.float32, allow_tf32=False)

    input_scale_v = tl.load(input_scale).to(tl.float32)
    weight_scale_v = tl.load(weight_metadata).to(tl.float32)
    bias_v = tl.load(bias + rbn)
    real = acc * (input_scale_v * weight_scale_v) + bias_v[None, :]

    out_scale_v = tl.load(output_scale).to(tl.float32)
    out_zp_v = tl.load(output_zero_point).to(tl.float32)
    qo = _rne(real / out_scale_v) + out_zp_v
    qo = tl.minimum(tl.maximum(qo, 0.0), 255.0)
    deq = (qo - out_zp_v) * out_scale_v

    out_ptrs = output + rm[:, None] * stride_on + rn[None, :]
    tl.store(out_ptrs, deq)


@libentry()
@triton.jit
def _wqlp_empty_k_kernel(
    output,
    output_scale,
    output_zero_point,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = -tl.load(output_zero_point).to(tl.float32)
    value *= tl.load(output_scale).to(tl.float32)
    # Unmasked store into the clamped last element: every lane writes the same
    # constant, so trailing duplicates are harmless (masked store is not honoured
    # on XPU3, KERNEL_OPT row 23).
    tl.store(output + tl.minimum(offsets, numel - 1), value)


@libentry()
@triton.jit
def _wqlp_copy_kernel(
    src,
    dst,
    cols,
    stride_sr,
    stride_sc,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy the ``(rows, cols)`` top-left block of ``src`` into contiguous
    ``dst``.  Non-dot kernel; bounds via clamped addresses (no masked store)."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off = tl.minimum(offsets, total - 1)
    r = off // cols
    c = off % cols
    val = tl.load(src + r * stride_sr + c * stride_sc)
    tl.store(dst + off, val)


def _pick_block_m(M):
    if M <= _DOT_MIN:
        return _DOT_MIN
    if M <= 32:
        return 32
    if M <= 64:
        return 64
    return 128


def _pick_block_n(N):
    if N <= 16:
        return 16
    if N <= 32:
        return 32
    if N <= 64:
        return 64
    return 128


def _pick_block_k(K):
    if K % 128 == 0:
        return 128
    return 64


def _wrapped_quantized_linear_prepacked(
    input,
    input_scale,
    input_zero_point,
    packed_weight,
    output_scale,
    output_zero_point,
    out_channel,
):
    logger.debug("GEMS_KUNLUNXIN _WRAPPED_QUANTIZED_LINEAR_PREPACKED")
    if input.dtype != torch.float32:
        raise RuntimeError(f"Quantize only works on Float Tensor, got {input.dtype}")
    if input.ndim < 2:
        raise RuntimeError(
            "The dimension of input tensor should be larger than or equal to 2"
        )
    if out_channel < 0:
        raise RuntimeError("out_channel must be non-negative")
    for name, parameter in (
        ("input_scale", input_scale),
        ("input_zero_point", input_zero_point),
        ("output_scale", output_scale),
        ("output_zero_point", output_zero_point),
    ):
        if parameter.numel() != 1:
            raise RuntimeError(f"{name} must contain one element")
        if parameter.device != input.device:
            raise RuntimeError(f"{name} must be on the input device")
    if packed_weight.device != input.device:
        raise RuntimeError("packed_weight must be on the input device")

    K = input.shape[-1]
    quantized_weight, weight_metadata, bias = unpack_linear_weight(
        packed_weight, out_channel, K
    )

    output_shape = (*input.shape[:-1], out_channel)
    output = torch.empty(output_shape, dtype=torch.float32, device=input.device)
    M = 1
    for dimension in input.shape[:-1]:
        M *= dimension
    if M == 0 or out_channel == 0:
        return output

    device = input.device
    if K == 0:
        block_size = 256
        grid = (triton.cdiv(output.numel(), block_size),)
        with torch_device_fn.device(device):
            _wqlp_empty_k_kernel[grid](
                output,
                output_scale,
                output_zero_point,
                output.numel(),
                BLOCK_SIZE=block_size,
            )
        return output

    input_2d = input.reshape(M, K)
    N = out_channel
    block_k = _pick_block_k(K)
    Kp = triton.cdiv(K, block_k) * block_k
    block_m = _pick_block_m(M)
    block_n = _pick_block_n(N)
    Mp = triton.cdiv(M, block_m) * block_m
    Np = triton.cdiv(N, block_n) * block_n
    Mq = M
    s_in = torch.empty((Mq, Kp), dtype=torch.float16, device=device)
    total_q = M * Kp
    use_flat = input_2d.is_contiguous() and Kp == K
    with torch_device_fn.device(device):
        if use_flat:
            flat_block = 8192
            while total_q % flat_block != 0:
                flat_block //= 2
            _wqlp_quant_flat_kernel[(total_q // flat_block,)](
                input_2d,
                input_scale,
                input_zero_point,
                s_in,
                BLOCK=flat_block,
            )
        else:
            if Kp % 256 == 0:
                block_q = 256
            elif Kp % 128 == 0:
                block_q = 128
            else:
                block_q = 64
            _wqlp_quant_kernel[(M, Kp // block_q)](
                input_2d,
                input_scale,
                input_zero_point,
                s_in,
                K,
                Kp,
                input_2d.stride(0),
                input_2d.stride(1),
                BLOCK_K=block_q,
                RAGGED=(Kp != K),
            )

    scratch = torch.empty((Mp, Np), dtype=torch.float32, device=device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    with torch_device_fn.device(device):
        _wqlp_kernel[grid](
            s_in,
            quantized_weight,
            bias,
            input_scale,
            weight_metadata,
            output_scale,
            output_zero_point,
            scratch,
            M,
            N,
            K,
            Kp,
            s_in.stride(0),
            quantized_weight.stride(0),
            scratch.stride(0),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            RAGGED=(Kp != K),
        )

    if Mp == M and Np == N:
        return scratch.view(output_shape)
    if Np == N:
        return scratch[:M, :].view(output_shape)

    total_out = M * N
    block_size = 1024
    copy_grid = (triton.cdiv(total_out, block_size),)
    with torch_device_fn.device(device):
        _wqlp_copy_kernel[copy_grid](
            scratch,
            output,
            N,
            scratch.stride(0),
            scratch.stride(1),
            total_out,
            BLOCK_SIZE=block_size,
        )
    return output
