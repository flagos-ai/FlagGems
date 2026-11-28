"""Single-axis complex FFT and inverse real FFT."""

import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.triton_lang_helper import tl_extra_shim


@triton.jit
def _phase(angle):
    if angle.dtype == tl.float64:
        return tl_extra_shim.cos(angle), tl_extra_shim.sin(angle)
    return tl.cos(angle), tl.sin(angle)


@triton.jit
def _angle(index, denominator: tl.constexpr, sign: tl.constexpr, dtype: tl.constexpr):
    pi = tl.full((), 0x400921FB54442D18, tl.int64).to(tl.float64, bitcast=True)
    return index.to(dtype) * (pi.to(dtype) * sign / denominator)


@triton.jit
def _scale(N: tl.constexpr, MODE: tl.constexpr, DTYPE: tl.constexpr):
    length = tl.full((), N, DTYPE)
    if MODE == 2:
        root = tl.sqrt(length)
        return 1.0 / root
    if MODE == 1:
        return 1.0 / length
    return tl.full((), 1, DTYPE)


@triton.jit
def _radix2(re, im, lane, LOG_N: tl.constexpr, SIGN: tl.constexpr):
    for stage in tl.static_range(LOG_N):
        half = 1 << stage
        partner = lane ^ half
        other_re = tl.gather(re, partner, 0)
        other_im = tl.gather(im, partner, 0)
        angle = _angle(lane % half, half, SIGN, re.dtype)
        cos, sin = _phase(angle)
        upper = (lane & half) != 0
        even_re = tl.where(upper, other_re, re)
        even_im = tl.where(upper, other_im, im)
        odd_re = tl.where(upper, re, other_re)
        odd_im = tl.where(upper, im, other_im)
        rot_re = odd_re * cos - odd_im * sin
        rot_im = odd_re * sin + odd_im * cos
        re = even_re + tl.where(upper, -rot_re, rot_re)
        im = even_im + tl.where(upper, -rot_im, rot_im)
    return re, im


@triton.jit
def _bit_reverse(lane, LOG_N: tl.constexpr):
    reverse = tl.full(lane.shape, 0, tl.int32)
    for bit in tl.static_range(LOG_N):
        reverse = (reverse << 1) | ((lane >> bit) & 1)
    return reverse


@triton.jit
def _split_stage(re, im, HALF: tl.constexpr, GROUPS: tl.constexpr, SIGN: tl.constexpr):
    even_re, odd_re = tl.split(re.reshape(GROUPS, 2, HALF).permute(0, 2, 1))
    even_im, odd_im = tl.split(im.reshape(GROUPS, 2, HALF).permute(0, 2, 1))
    cos, sin = _phase(_angle(tl.arange(0, HALF), HALF, SIGN, re.dtype))
    rot_re = odd_re * cos[None, :] - odd_im * sin[None, :]
    rot_im = odd_re * sin[None, :] + odd_im * cos[None, :]
    re = tl.join(even_re + rot_re, even_re - rot_re).permute(0, 2, 1).reshape(re.shape)
    im = tl.join(even_im + rot_im, even_im - rot_im).permute(0, 2, 1).reshape(im.shape)
    return re, im


@triton.jit
def _split_radix2(re, im, WIDTH: tl.constexpr, LOG_N: tl.constexpr, SIGN: tl.constexpr):
    for stage in tl.static_range(LOG_N):
        re, im = _split_stage(re, im, 1 << stage, WIDTH >> (stage + 1), SIGN)
    return re, im


@triton.jit
def _split4(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    lo, hi = tl.split(x.reshape(GROUPS, 2, 2, STRIDE).permute(0, 2, 3, 1))
    x0, x1 = tl.split(lo.permute(0, 2, 1))
    x2, x3 = tl.split(hi.permute(0, 2, 1))
    return x0, x1, x2, x3


@triton.jit
def _join4(a, b, c, d):
    return tl.join(tl.join(a, b), tl.join(c, d)).permute(0, 3, 2, 1)


@triton.jit
def _radix3_part(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x0, x1, x2, _ = _split4(x, GROUPS, STRIDE)
    total = x1 + x2
    return x0 + total, x0 - 0.5 * total, 0.8660254037844386 * (x1 - x2)


@triton.jit
def _inverse_radix3(re, im, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    r0, r1, dr = _radix3_part(re, GROUPS, STRIDE)
    i0, i1, di = _radix3_part(im, GROUPS, STRIDE)
    zero = tl.full(r0.shape, 0, r0.dtype)
    return (
        _join4(r0, r1 - di, r1 + di, zero).reshape(re.shape),
        _join4(i0, i1 + dr, i1 - dr, zero).reshape(im.shape),
    )


@triton.jit
def _radix5_part(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    lo, hi = tl.split(x.reshape(GROUPS, 2, 4, STRIDE).permute(0, 2, 3, 1))
    x0, x1, x2, x3 = _split4(lo, GROUPS, STRIDE)
    x4, _, _, _ = _split4(hi, GROUPS, STRIDE)
    sum14 = x1 + x4
    sum23 = x2 + x3
    diff14 = x1 - x4
    diff23 = x2 - x3
    cos1: tl.constexpr = 0.3090169943749474
    cos2: tl.constexpr = -0.8090169943749475
    sin1: tl.constexpr = 0.9510565162951535
    sin2: tl.constexpr = 0.5877852522924731
    return (
        x0 + sum14 + sum23,
        x0 + cos1 * sum14 + cos2 * sum23,
        x0 + cos2 * sum14 + cos1 * sum23,
        sin1 * diff14 + sin2 * diff23,
        sin2 * diff14 - sin1 * diff23,
    )


@triton.jit
def _join5(a, b, c, d, e, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    zero = tl.full(a.shape, 0, a.dtype)
    lo = _join4(a, b, c, d).reshape(GROUPS, 4, STRIDE)
    hi = _join4(e, zero, zero, zero).reshape(GROUPS, 4, STRIDE)
    return tl.join(lo, hi).permute(0, 3, 1, 2).reshape(GROUPS * 8 * STRIDE)


@triton.jit
def _inverse_radix5(re, im, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    r0, r1, r2, dr1, dr2 = _radix5_part(re, GROUPS, STRIDE)
    i0, i1, i2, di1, di2 = _radix5_part(im, GROUPS, STRIDE)
    return (
        _join5(r0, r1 - di1, r2 - di2, r2 + di2, r1 + di1, GROUPS, STRIDE),
        _join5(i0, i1 + dr1, i2 + dr2, i2 - dr2, i1 - dr1, GROUPS, STRIDE),
    )


@triton.jit
def _irfft_240(
    X, Y, n_in, n_rows, MODE: tl.constexpr, COMPLEX: tl.constexpr, ROWS: tl.constexpr
):
    lane = tl.arange(0, 256 * ROWS)
    row = tl.program_id(0).to(tl.int64) * ROWS + lane // 256
    a = lane % 8
    b = (lane // 8) % 4
    c = (lane // 32) % 8
    # A 120-point complex IFFT gives the even and odd output samples as its
    # real and imaginary components. Use an 8-by-3-by-5 prime-factor FFT.
    j = (15 * _bit_reverse(a, 3) + 40 * b + 24 * c) % 120
    valid = (row < n_rows) & (b < 3) & (c < 5)
    pair = tl.arange(0, 2)
    if COMPLEX:
        mirror = 120 - j
        left = (2 * (row * n_in + j))[:, None] + pair[None, :]
        right = (2 * (row * n_in + mirror))[:, None] + pair[None, :]
        xr, xi = tl.split(tl.load(X + left, (valid & (j < n_in))[:, None], 0))
        yr, yi = tl.split(tl.load(X + right, (valid & (mirror < n_in))[:, None], 0))
        xi = tl.where(j == 0, 0, xi)
        yi = tl.where(j == 0, 0, yi)
    else:
        xr, xi = _read_sample(X, row, j, n_in, 240, False, True, tl.float32)
        yr, yi = _read_sample(X, row, 120 - j, n_in, 240, False, True, tl.float32)
    cos, sin = _phase(_angle(j, 120, 1, tl.float32))
    diff_re = xr - yr
    diff_im = xi + yi
    re = xr + yr - (diff_re * sin + diff_im * cos)
    im = xi - yi + (diff_re * cos - diff_im * sin)
    re, im = _split_radix2(re, im, 256 * ROWS, 3, 1)
    re, im = _inverse_radix3(re, im, 8 * ROWS, 8)
    re, im = _inverse_radix5(re, im, ROWS, 32)
    k = (120 - 15 * a + 40 * b + 96 * c) % 120
    scale = _scale(240, MODE, tl.float32)
    offsets = (row * 240 + 2 * k)[:, None] + pair[None, :]
    tl.store(Y + offsets, tl.join(re, im) * scale, valid[:, None])


@triton.jit
def _read_sample(
    X,
    row,
    j,
    n_in,
    N: tl.constexpr,
    COMPLEX: tl.constexpr,
    INVERSE_REAL: tl.constexpr,
    DTYPE: tl.constexpr,
):
    if INVERSE_REAL:
        mirror = j > N // 2
        bin = tl.where(mirror, N - j, j)
        offset = row * n_in + bin
        valid = (j < N) & (bin < n_in)
        if COMPLEX:
            re = tl.load(X + 2 * offset, valid, 0).to(DTYPE)
            imaginary = valid & (bin != 0) & (2 * bin != N)
            im = tl.load(X + 2 * offset + 1, imaginary, 0).to(DTYPE)
            return re, tl.where(mirror, -im, im)
        re = tl.load(X + offset, valid, 0).to(DTYPE)
        return re, tl.full(j.shape, 0, DTYPE)
    if COMPLEX:
        offset = 2 * (row * n_in + j)
        re = tl.load(X + offset, (j < n_in) & (j < N), 0).to(DTYPE)
        im = tl.load(X + offset + 1, (j < n_in) & (j < N), 0).to(DTYPE)
        return re, im
    re = tl.load(X + row * n_in + j, (j < n_in) & (j < N), 0).to(DTYPE)
    return re, tl.full(j.shape, 0, DTYPE)


@triton.jit
def _fft_real_256(X, Y, n_in, MODE: tl.constexpr):
    k = tl.arange(0, 128)
    row = tl.program_id(0).to(tl.int64)
    j = _bit_reverse(k, 7) * 2
    re = tl.load(X + row * n_in + j, j < n_in, 0)
    im = tl.load(X + row * n_in + j + 1, j + 1 < n_in, 0)
    re, im = _split_radix2(re, im, 128, 7, -1)
    mirror = (128 - k) % 128
    other_re = tl.gather(re, mirror, 0)
    other_im = tl.gather(im, mirror, 0)
    cos, sin = _phase(_angle(k, 128, -1, tl.float32))
    diff_re = re - other_re
    diff_im = im + other_im
    delta_re = diff_re * sin + diff_im * cos
    delta_im = diff_im * sin - diff_re * cos
    scale = 0.5 * _scale(256, MODE, tl.float32)
    out = Y + 2 * (row * 256 + k)
    tl.store(out, (re + other_re + delta_re) * scale)
    tl.store(out + 1, (im - other_im + delta_im) * scale)
    tl.store(out + 256, (re + other_re - delta_re) * scale)
    tl.store(out + 257, (im - other_im - delta_im) * scale)


@triton.jit
def _fft_chirp(
    C, S, N: tl.constexpr, WIDTH: tl.constexpr, LOG_N: tl.constexpr, SIGN: tl.constexpr
):
    lane = tl.arange(0, WIDTH)
    j = _bit_reverse(lane, LOG_N)
    distance = tl.minimum(j, WIDTH - j)
    angle = _angle((distance * distance) % (2 * N), N, -SIGN, C.dtype.element_ty)
    re, im = _phase(angle)
    re = tl.where(distance < N, re, 0)
    im = tl.where(distance < N, im, 0)
    re, im = _radix2(re, im, lane, LOG_N, -1)
    tl.store(C + lane, re)
    tl.store(S + lane, im)


@triton.jit
def _fft_radix2(
    X,
    Y,
    C,
    S,
    n_in,
    MODE: tl.constexpr,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    LOG_N: tl.constexpr,
    COMPLEX: tl.constexpr,
    INVERSE_REAL: tl.constexpr,
    DTYPE: tl.constexpr,
):
    lane = tl.arange(0, WIDTH)
    row = tl.program_id(0).to(tl.int64)
    j = _bit_reverse(lane, LOG_N)
    re, im = _read_sample(X, row, j, n_in, N, COMPLEX, INVERSE_REAL, DTYPE)
    SIGN: tl.constexpr = 1 if INVERSE_REAL else -1
    if WIDTH == N:
        re, im = _split_radix2(re, im, WIDTH, LOG_N, SIGN)
    else:
        angle = _angle((j * j) % (2 * N), N, SIGN, DTYPE)
        cos, sin = _phase(angle)
        re, im = re * cos - im * sin, re * sin + im * cos
        re, im = _radix2(re, im, lane, LOG_N, -1)
        c = tl.load(C + lane)
        s = tl.load(S + lane)
        re, im = re * c - im * s, re * s + im * c
        re, im = tl.gather(re, j, 0), tl.gather(im, j, 0)
        re, im = _radix2(re, im, lane, LOG_N, 1)
        angle = _angle((lane * lane) % (2 * N), N, SIGN, DTYPE)
        cos, sin = _phase(angle)
        re, im = (re * cos - im * sin) / WIDTH, (re * sin + im * cos) / WIDTH
    scale = _scale(N, MODE, DTYPE)
    if INVERSE_REAL:
        tl.store(Y + row * N + lane, re * scale, lane < N)
    else:
        tl.store(Y + 2 * (row * N + lane), re * scale, lane < N)
        tl.store(Y + 2 * (row * N + lane) + 1, im * scale, lane < N)


@triton.jit
def _fft_tiled_dft(
    X,
    Y,
    n_in,
    MODE: tl.constexpr,
    N: tl.constexpr,
    COMPLEX: tl.constexpr,
    INVERSE_REAL: tl.constexpr,
    DTYPE: tl.constexpr,
):
    K: tl.constexpr = 8
    J: tl.constexpr = 32
    tiles: tl.constexpr = triton.cdiv(N, K)
    pid = tl.program_id(0)
    row = (pid // tiles).to(tl.int64)
    k = (pid % tiles) * K + tl.arange(0, K)
    acc_re = tl.full((K,), 0, DTYPE)
    acc_im = tl.full((K,), 0, DTYPE)
    SIGN: tl.constexpr = 1 if INVERSE_REAL else -1
    for start in range(0, N, J):
        j = start + tl.arange(0, J)
        re, im = _read_sample(X, row, j, n_in, N, COMPLEX, INVERSE_REAL, DTYPE)
        angle = _angle((k[:, None].to(tl.int64) * j[None, :]) % N, N, 2 * SIGN, DTYPE)
        cos, sin = _phase(angle)
        acc_re += tl.sum(re[None, :] * cos - im[None, :] * sin, 1)
        if not INVERSE_REAL:
            acc_im += tl.sum(re[None, :] * sin + im[None, :] * cos, 1)
    scale = _scale(N, MODE, DTYPE)
    if INVERSE_REAL:
        tl.store(Y + row * N + k, acc_re * scale, k < N)
    else:
        tl.store(Y + 2 * (row * N + k), acc_re * scale, k < N)
        tl.store(Y + 2 * (row * N + k) + 1, acc_im * scale, k < N)


def _fft_1d(x, n, dim, norm, inverse_real, *, return_pairs=False):
    rank = x.ndim
    if not -rank <= dim < rank:
        raise IndexError(f"FFT dimension {dim} out of range for rank {rank}")
    dim %= rank
    n_in = x.shape[dim]
    if n is None:
        n = 2 * (n_in - 1) if inverse_real else n_in
    if n <= 0:
        raise RuntimeError(f"Invalid number of data points ({n}) specified")
    if norm not in (None, "backward", "forward", "ortho"):
        raise RuntimeError(f"Invalid normalization mode: {norm!r}")
    if not x.is_floating_point() and not x.is_complex():
        x = x.to(torch.get_default_dtype())
    complex_input = x.is_complex()
    if dim != rank - 1:
        x = x.movedim(dim, -1)
    if x.is_conj():
        x = x.resolve_conj()
    if x.is_neg():
        x = x.resolve_neg()
    x = torch.view_as_real(x) if complex_input else x
    if not x.is_contiguous():
        x = x.contiguous()
    batch = x.shape[:-2] if complex_input else x.shape[:-1]
    rows = math.prod(batch)
    dtype = x.dtype
    if dtype == torch.bfloat16:
        raise RuntimeError("Unsupported dtype BFloat16")
    out_shape = (*batch, n) if inverse_real else (*batch, n, 2)
    out = torch.empty(out_shape, device=x.device, dtype=dtype)
    if rows:
        mode = 2 if norm == "ortho" else int((norm == "forward") != inverse_real)
        compute = tl.float64 if dtype == torch.float64 else tl.float32
        work_n = n if n & (n - 1) == 0 else 2 * n - 1
        width = triton.next_power_of_2(work_n)
        with torch_device_fn.device(x.device):
            if (
                not inverse_real
                and not complex_input
                and n == 256
                and dtype == torch.float32
            ):
                _fft_real_256[(rows,)](x, out, n_in, mode, num_warps=1)
            elif inverse_real and n == 240 and dtype == torch.float32:
                rows_per_program = 4 if complex_input and rows >= 1024 else 1
                _irfft_240[(triton.cdiv(rows, rows_per_program),)](
                    x,
                    out,
                    n_in,
                    rows,
                    mode,
                    complex_input,
                    rows_per_program,
                    num_warps=max(1, rows_per_program // 2),
                )
            elif width <= 1024:
                log_n = width.bit_length() - 1
                chirp_re = chirp_im = out
                if width != n:
                    work_dtype = (
                        torch.float64 if dtype == torch.float64 else torch.float32
                    )
                    chirp_re = torch.empty(width, device=x.device, dtype=work_dtype)
                    chirp_im = torch.empty_like(chirp_re)
                    _fft_chirp[(1,)](
                        chirp_re, chirp_im, n, width, log_n, 1 if inverse_real else -1
                    )
                _fft_radix2[(rows,)](
                    x,
                    out,
                    chirp_re,
                    chirp_im,
                    n_in,
                    mode,
                    n,
                    width,
                    log_n,
                    complex_input,
                    inverse_real,
                    compute,
                    num_warps=1 if width == n and n <= 256 else 4,
                )
            else:
                _fft_tiled_dft[(rows * triton.cdiv(n, 8),)](
                    x, out, n_in, mode, n, complex_input, inverse_real, compute
                )
    if return_pairs:
        return out
    if not inverse_real:
        out = torch.view_as_complex(out)
    return out if dim == rank - 1 else out.movedim(-1, dim)


@triton.jit
def _fft_grad_kernel(
    X,
    Y,
    SIZE,
    INPUT_LENGTH: tl.constexpr,
    N: tl.constexpr,
    COMPLEX: tl.constexpr,
    INVERSE_REAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = idx // INPUT_LENGTH
    bin = idx % INPUT_LENGTH
    if INVERSE_REAL:
        present = (idx < SIZE) & (bin <= N // 2)
        weight = tl.where((bin == 0) | (bin * 2 == N), 1, 2)
    else:
        present = (idx < SIZE) & (bin < N)
        weight = 1
    offset = 2 * (row * N + bin)
    real = tl.load(X + offset, present, 0)
    if COMPLEX:
        imag = tl.load(X + offset + 1, present, 0)
        if not INVERSE_REAL:
            imag = -imag
        tl.store(Y + 2 * idx, real * weight, idx < SIZE)
        tl.store(Y + 2 * idx + 1, imag * weight, idx < SIZE)
    else:
        tl.store(Y + idx, real * weight, idx < SIZE)


class _FFTAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n, dim, norm, inverse_real):
        out = _fft_1d(x, n, dim, norm, inverse_real)
        ctx.dim = dim
        ctx.norm = norm
        ctx.n = out.shape[dim]
        ctx.input_length = x.shape[dim]
        ctx.inverse_real = inverse_real
        ctx.complex_input = x.is_complex()
        return out

    @staticmethod
    def backward(ctx, grad):
        result = _FFTAdjoint.apply(
            grad,
            ctx.n,
            ctx.dim,
            ctx.norm,
            ctx.inverse_real,
            ctx.input_length,
            ctx.complex_input,
        )
        return result, None, None, None, None


class _FFTAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, n, dim, norm, inverse_real, input_length, complex_input):
        ctx.transform = (n, dim, norm, inverse_real)
        if inverse_real:
            norm = {"ortho": "ortho", "forward": "backward"}.get(norm, "forward")
        else:
            grad = grad.conj()
        spectrum = _fft_1d(grad, n, dim, norm, False, return_pairs=True)
        shape = (*spectrum.shape[:-2], input_length)
        size = math.prod(shape)
        if complex_input:
            shape = (*shape, 2)
        result = torch.empty(shape, device=grad.device, dtype=spectrum.dtype)
        if size:
            with torch_device_fn.device(grad.device):
                _fft_grad_kernel[(triton.cdiv(size, 256),)](
                    spectrum,
                    result,
                    size,
                    input_length,
                    n,
                    complex_input,
                    inverse_real,
                    256,
                )
        if complex_input:
            result = torch.view_as_complex(result)
        return result.movedim(-1, dim)

    @staticmethod
    def backward(ctx, grad):
        # The adjoint of the adjoint is the original linear transform.
        result = _FFTAutograd.apply(grad, *ctx.transform)
        return result, None, None, None, None, None, None


def fft_fft(self, n=None, dim=-1, norm=None):
    if self.requires_grad and torch.is_grad_enabled():
        return _FFTAutograd.apply(self, n, dim, norm, False)
    return _fft_1d(self, n, dim, norm, False)


def fft_irfft(self, n=None, dim=-1, norm=None):
    if self.requires_grad and torch.is_grad_enabled():
        return _FFTAutograd.apply(self, n, dim, norm, True)
    return _fft_1d(self, n, dim, norm, True)
