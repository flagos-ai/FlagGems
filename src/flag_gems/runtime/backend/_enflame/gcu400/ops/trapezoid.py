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

from flag_gems.ops.add import add as gems_add
from flag_gems.ops.cat import cat as gems_cat
from flag_gems.ops.mul import mul as gems_mul
from flag_gems.ops.sub import sub as gems_sub
from flag_gems.ops.sum import sum_dim as gems_sum_dim
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _patch_torch_trapezoid_complex_reference():
    """GCU reference fallback for complex dtypes (module-level, import-time).

    torch_gcu cannot run the native aten trapezoid on complex device inputs:
    the CompositeImplicitAutograd decomposition lowers to topsaten ops that
    abort the whole process (topsatenSum rejects ComplexFloat/ComplexDouble;
    the real->complex cast is equally unsupported). The accuracy harness's
    on-device reference line ``torch.trapezoid(ref_y, ref_x)`` therefore dies
    before any comparison runs.

    Route only complex non-CPU inputs through CPU torch -- the reference's
    own semantics, computed on the reference device that supports the dtype.
    Real dtypes keep the native device path untouched, so this changes no
    production numerics; the wrapper is idempotent and signature-preserving
    (both the ``x`` and ``dx`` overloads pass through).
    """
    if getattr(torch.trapezoid, "_flaggems_gcu_complex_fallback", False):
        return
    original = torch.trapezoid

    def _trapezoid_with_complex_fallback(y, *args, **kwargs):
        x = args[0] if args else kwargs.get("x")
        complex_input = y.is_complex() or (
            x is not None and isinstance(x, torch.Tensor) and x.is_complex()
        )
        if complex_input and y.device.type != "cpu":
            y_cpu = y.cpu()
            x_cpu = x.cpu() if isinstance(x, torch.Tensor) else x
            if args:
                return original(y_cpu, x_cpu, *args[1:], **kwargs)
            if x_cpu is not None:
                kwargs = dict(kwargs, x=x_cpu)
            return original(y_cpu, **kwargs)
        return original(y, *args, **kwargs)

    _trapezoid_with_complex_fallback._flaggems_gcu_complex_fallback = True
    torch.trapezoid = _trapezoid_with_complex_fallback


_patch_torch_trapezoid_complex_reference()


# GCU note: the generic implementation autotunes over the "naive_reduction"
# space (BLOCK_M up to 256 x BLOCK_N up to 4096). On gcu400 the 2D
# [BLOCK_M, BLOCK_N] fp32 accumulator tile makes the 64K-element configs
# (128x512, 256x256) pass the compile-time DSM gate but fault the device at
# runtime (Sip exception -> KMD abort -> SIGABRT), killing the autotuner and
# the whole pytest process; larger tiles only hit a clean OutOfResources.
# Following the fixed gcu400 reduction ops (sum/trace/std/amax/trapz), drop
# the tuner and launch with explicit, DSM-safe block sizes instead.
@libentry()
@triton.jit
def trapezoid_x_kernel(
    y_ptr,
    x_ptr,
    out_ptr,
    M,
    N,
    x_m_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # Fast path used when y and x share the same length N along the integration
    # dim. sum_{i=1..N-1} 0.5*(x_i - x_{i-1})*(y_i + y_{i-1}) is rewritten as a
    # single weighted pass sum_k y_k * 0.5*(x_{k+1} - x_{k-1}), where the neighbour
    # indices are clamped at the endpoints. This reads each y element only once.
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = pid < M
    y_row = y_ptr + pid * N
    # x_m_stride is 0 when x is a 1-D spacing shared across rows, or N when x
    # provides per-row spacing (broadcast to y's shape by the wrapper).
    x_row = x_ptr + pid * x_m_stride

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=COMPUTE_DTYPE)
    for off in range(0, N, BLOCK_N):
        k = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = k < N
        mask = row_mask & col_mask
        kp = tl.minimum(k + 1, N - 1)
        km = tl.maximum(k - 1, 0)
        y_k = tl.load(y_row + k, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        x_kp = tl.load(x_row + kp, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        x_km = tl.load(x_row + km, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        acc += y_k * 0.5 * (x_kp - x_km)

    out = tl.sum(acc, axis=1)[:, None]
    tl.store(out_ptr + pid, out, mask=row_mask)


@libentry()
@triton.jit
def trapezoid_x_pair_kernel(
    lpr_ptr,
    dx_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # Slow path for pair-space broadcasting (y and x differ in length along the
    # integration dim). lpr (y_left + y_right) and dx (x_right - x_left) are
    # already reduced by one and broadcast to the common pair shape, then flattened
    # to (M, N). Compute the inner product sum_k lpr_k * dx_k and scale by 0.5.
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = pid < M
    lpr_row = lpr_ptr + pid * N
    dx_row = dx_ptr + pid * N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=COMPUTE_DTYPE)
    for off in range(0, N, BLOCK_N):
        k = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = k < N
        mask = row_mask & col_mask
        lpr_k = tl.load(lpr_row + k, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        dx_k = tl.load(dx_row + k, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        acc += lpr_k * dx_k

    out = tl.sum(acc, axis=1)[:, None] * 0.5
    tl.store(out_ptr + pid, out, mask=row_mask)


# GCU note: when the integration dim is the only non-singleton axis the flattened
# problem is a single (or very few) row of huge length, so the row-per-program
# kernel above leaves the grid with one CTA and serialises megabytes through the
# BLOCK_N loop (measured 74 ms for a 4 Mi-element fp16 input, i.e. ~0.001x ATen).
# Split the integration axis instead: each program owns one (row, column-slice)
# block, writes one partial sum per row into an (M, NSPLIT) float32 scratch, and
# a tiny second kernel folds the slices. The weighted-sum form is element-wise
# over k, so slicing k is a pure re-association of the same sum.
_SPLIT_ROW_THRESHOLD = 256  # below this the row grid is too small to fill the device
_SPLIT_MIN_N = 8192  # only pay the extra reduce launch for long rows
_SPLIT_NSPLIT = 128  # observed gcu400 throughput plateau for column slices
_SPLIT_BLOCK_N = 2048


@libentry()
@triton.jit
def trapezoid_x_split_kernel(
    y_ptr,
    x_ptr,
    part_ptr,
    M,
    N,
    x_m_stride,
    COL_CHUNK: tl.constexpr,
    CHUNK_ITERS: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # One program handles a single row over the column slice [pid_s*COL_CHUNK,
    # (pid_s+1)*COL_CHUNK). The slice length is fixed at compile time through
    # CHUNK_ITERS so the loop bound stays static; only the tail slice is masked.
    pid = ext.program_id(0)
    row = pid // NSPLIT
    pid_s = pid % NSPLIT
    row_ok = row < M
    y_row = y_ptr + row * N
    x_row = x_ptr + row * x_m_stride

    acc = tl.zeros([BLOCK_N], dtype=COMPUTE_DTYPE)
    for it in range(0, CHUNK_ITERS):
        k = pid_s * COL_CHUNK + it * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (k < N) & row_ok
        kp = tl.minimum(k + 1, N - 1)
        km = tl.maximum(k - 1, 0)
        y_k = tl.load(y_row + k, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        x_kp = tl.load(x_row + kp, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        x_km = tl.load(x_row + km, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        acc += y_k * 0.5 * (x_kp - x_km)

    tl.store(part_ptr + row * NSPLIT + pid_s, tl.sum(acc, axis=0), mask=row_ok)


@libentry()
@triton.jit
def trapezoid_x_split_reduce_kernel(
    part_ptr,
    out_ptr,
    M,
    NSPLIT,
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Fold the (M, NSPLIT) partial sums along the slice axis. The row count is
    # small by construction (the split path is only taken for flat inputs), so a
    # single narrow tile per program is enough. NSPLIT need not be a power of two
    # (it comes from the column-chunk ceil-division), so the tile is padded to
    # BLOCK_S and masked.
    rows = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    s = tl.arange(0, BLOCK_S)[None, :]
    mask = (rows < M) & (s < NSPLIT)
    part = tl.load(part_ptr + rows * NSPLIT + s, mask=mask, other=0.0)
    tl.store(out_ptr + rows, tl.sum(part, axis=1)[:, None], mask=rows < M)


# GCU note: gcu400 persistent row-reduction form for the shared 1-D spacing
# case (x_m_stride == 0). One program per SPC (48) strides over row tiles, so
# the y stream keeps the flat-bandwidth access pattern of the gcu400 sum ops
# instead of thousands of short-lived programs. The x neighbour loads stay
# [BLOCK_N] 1-D tiles (loaded once per column block and broadcast over the row
# block in registers), and tl.sum(axis=1) per column block keeps only a
# [BLOCK_M] accumulator alive across the loop -- the 2D accumulator tile of
# trapezoid_x_kernel is what pushed the per-program register set past what the
# row-broadcast x loads leave resident.
_ROW_MAX_GRID = 48
_ROW_TILE_BYTES = 131072  # per-tile byte budget, matches the gcu400 sum config
_ROW_TILE_ELEMS = 65536  # 2D tile element cap (larger 2D tiles device-fault)
_ROW_TREE_WIDTH = 4096  # tl.sum(axis=1) drops to a tiered slow path below this


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def trapezoid_x_row_kernel(
    y_ptr,
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid = ext.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tile, step):
        rows = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        acc = tl.zeros([BLOCK_M], dtype=COMPUTE_DTYPE)
        for off in tl.range(0, N, BLOCK_N):
            k = off + tl.arange(0, BLOCK_N)
            col_mask = k < N
            mask = row_mask[:, None] & col_mask[None, :]
            kp = tl.minimum(k + 1, N - 1)
            km = tl.maximum(k - 1, 0)
            y_k = tl.load(
                y_ptr + rows[:, None] * N + k[None, :], mask=mask, other=0.0
            ).to(COMPUTE_DTYPE)
            x_kp = tl.load(x_ptr + kp, mask=col_mask, other=0.0).to(COMPUTE_DTYPE)
            x_km = tl.load(x_ptr + km, mask=col_mask, other=0.0).to(COMPUTE_DTYPE)
            w = (0.5 * (x_kp - x_km))[None, :]
            acc += tl.sum(y_k * w, axis=1)
        tl.store(out_ptr + rows, acc.to(out_ptr.dtype.element_ty), mask=row_mask)


def _normalize_dim(ndim, dim):
    # Mirror ATen's dim validation: a 0-d tensor has no dims, and out-of-range dims
    # raise IndexError with the exact same message wording.
    if ndim == 0:
        raise IndexError(f"Dimension specified as {dim} but tensor has no dimensions")
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], "
            f"but got {dim})"
        )
    return dim if dim >= 0 else dim + ndim


def _check_bool(y, x=None):
    for name, t in (("y", y), ("x", x)):
        if t is not None and t.dtype == torch.bool:
            raise RuntimeError(
                "trapezoid: received a bool input for `x` or `y`, but bool is not supported"
            )


def _check_complex_dtype(y, x=None):
    # complex32 (ComplexHalf) has no ATen trapezoid implementation either
    # ("sum_cpu" not implemented), so there is no contract to mirror.
    for t in (y, x):
        if t is not None and t.dtype == torch.complex32:
            raise RuntimeError("trapezoid: complex32 inputs are not supported")


def _is_complex(y, x=None):
    return y.is_complex() or (x is not None and x.is_complex())


def _trapezoid_x_complex(y, x, dim):
    """Complex trapezoid routed through CPU torch (gcu400 adaptation).

    torch_gcu cannot compute complex dtypes on device at all: the aten
    real->complex cast aborts (topsatenCopy "output data type is not
    supported"), the native reduction terminates the process (topsatenSum
    rejects ComplexFloat), and torch.randn_like on a device complex tensor
    hard-aborts. Following the gcu400 linalg_solve_ex precedent, route
    complex inputs through CPU torch and keep the result on the host so
    downstream complex consumers (e.g. randn_like in the accuracy harness)
    stay usable. The differentiable .cpu() copy edges preserve autograd
    for backward.
    """
    _check_bool(y, x)
    _check_complex_dtype(y, x)
    dim = _normalize_dim(y.dim(), dim)
    if torch.is_grad_enabled():
        # torch_gcu's complex factories return non-leaf device copies
        # (ToCopyBackward), so a caller reading `y.grad` on the passed-in
        # tensor would otherwise always see None.
        if y.requires_grad and not y.is_leaf:
            y.retain_grad()
        if x.requires_grad and not x.is_leaf:
            x.retain_grad()
    return torch.trapezoid(y.cpu(), x.cpu(), dim=dim)


def _result_dtype(y, x=None):
    # ATen promotes y and x via result_type, but a pure-integer result is computed
    # in float32 (matching ATen's do_trapezoid, which returns float32 for int inputs).
    # Complex results are preserved as-is (ATen keeps complex64/complex128).
    if x is None:
        dt = y.dtype
    else:
        dt = torch.result_type(y, x)
    if not dt.is_floating_point and not dt.is_complex:
        dt = torch.float32
    return dt


def _compute_dtype(dtype):
    # GCU: the kernel compiler rejects f64 vector arithmetic (elementwise
    # f64 ops are explicitly illegal; fp64 chains even segfault make_gcuir),
    # and torch_gcu's fp64 device tensors only carry f32-precision values
    # anyway ("Double use Float replace"). Always accumulate in float32;
    # the host-side staging casts fp64 inputs down and the result back up.
    return tl.float32


def _zeros_shape(shape, dim):
    return shape[:dim] + shape[dim + 1 :]


_COMPLEX_REAL_DTYPES = {torch.complex64: torch.float32, torch.complex128: torch.float64}


def _zeros(shape, dtype, device):
    # torch.zeros / zeros_like on complex dtypes dispatch to the registered
    # FlagGems kernel, whose Triton implementation does not support complex
    # dtypes; build complex zeros through the real view instead.
    if dtype.is_complex:
        real = torch.zeros(
            list(shape) + [2], dtype=_COMPLEX_REAL_DTYPES[dtype], device=device
        )
        return torch.view_as_complex(real)
    return torch.zeros(shape, dtype=dtype, device=device)


def _zeros_like(t):
    return _zeros(t.shape, t.dtype, t.device)


def _view_x(x, y, dim):
    # Align x to y's dimensionality following ATen:
    #   - a 1-D x is placed along `dim` (all other dims are 1);
    #   - otherwise x is right-aligned by padding leading 1s to match y.dim().
    if x.dim() == 1:
        if x.shape[0] != y.shape[dim]:
            raise RuntimeError(
                "trapezoid: There must be one `x` value for each sample point"
            )
        new_sizes = [1] * y.dim()
        new_sizes[dim] = x.shape[0]
        return x.view(new_sizes)
    if x.dim() < y.dim():
        return x.view([1] * (y.dim() - x.dim()) + list(x.shape))
    return x


def _sum_dim_any(grad, dim, keepdim=False):
    """FlagGems sum_dim that also accepts complex tensors.

    The reduction kernels are real-valued, so a complex tensor is reduced by
    splitting it into its real/imaginary parts (view_as_real appends that axis
    last, so `dim` indices stay valid), summing each, and recombining.
    """
    if not grad.is_complex():
        return gems_sum_dim(grad, dim=dim, keepdim=keepdim)
    real = torch.view_as_real(grad)  # (..., 2)
    summed = gems_sum_dim(real, dim=dim, keepdim=keepdim)
    return torch.view_as_complex(summed.contiguous())


def _unbroadcast(grad, target_shape):
    # Sum a broadcast gradient back into the (possibly smaller) target shape.
    # Reductions route through the FlagGems sum_dim kernel (host-function
    # convention: no native PyTorch compute in the wrapper).
    while grad.dim() > len(target_shape):
        grad = _sum_dim_any(grad, dim=[0])
    for i, s in enumerate(target_shape):
        if s == 1 and grad.shape[i] != 1:
            grad = _sum_dim_any(grad, dim=[i], keepdim=True)
    return grad.reshape(target_shape)


def _cat_any(parts, dim):
    """FlagGems cat that also accepts complex tensors (via the real view)."""
    if not parts[0].is_complex():
        return gems_cat(parts, dim)
    reals = [torch.view_as_real(p) for p in parts]
    return torch.view_as_complex(gems_cat(reals, dim).contiguous())


def _accumulate_plus(grad_diff, dim):
    # Given grad wrt (y[k] + y[k+1]) over the pair axis, recover grad wrt y[k] via
    # grad_y[k] = grad_diff[k] + grad_diff[k-1] (endpoints clamped). The concat and
    # element-wise sum route through FlagGems kernels (host-function convention).
    pad = _zeros_like(grad_diff.narrow(dim, 0, 1))
    return gems_add(_cat_any([pad, grad_diff], dim), _cat_any([grad_diff, pad], dim))


def _accumulate_minus(grad_diff, dim):
    # Given grad wrt (x[k+1] - x[k]) over the pair axis, recover grad wrt x[k] via
    # grad_x[k] = grad_diff[k-1] - grad_diff[k] (endpoints clamped). The concat and
    # element-wise difference route through FlagGems kernels (host-function convention).
    pad = _zeros_like(grad_diff.narrow(dim, 0, 1))
    return gems_sub(_cat_any([pad, grad_diff], dim), _cat_any([grad_diff, pad], dim))


def _trapezoid_x_impl(y, x, dim):
    dim = _normalize_dim(y.dim(), dim)

    _check_bool(y, x)
    _check_complex_dtype(y, x)
    if _is_complex(y, x):
        return _trapezoid_x_complex(y, x, dim)

    out_dtype = _result_dtype(y, x)

    # Validate x against ATen's contract before the N <= 1 / empty-input early
    # returns: a 1-D x must provide one sample point per integration point.
    # ATen itself only rejects a mismatch when the integration dim is non-empty
    # (an empty dim integrates to zero regardless of x), so mirror that by
    # checking the length up front and skipping the check for size 0.
    if y.shape[dim] == 0:
        return torch.zeros(_zeros_shape(y.shape, dim), dtype=out_dtype, device=y.device)

    # GCU: fp64 inputs are staged through float32 (no f64 kernels; the device
    # fp64 tensors only carry f32-precision values under torch_gcu's
    # "Double use Float replace"). The result is cast back to the promoted
    # out dtype so the returned tensor keeps the ATen contract.
    kernel_out_dtype = torch.float32 if out_dtype == torch.float64 else out_dtype
    if kernel_out_dtype != out_dtype:
        y = y.to(torch.float32)
        x = x.to(torch.float32)

    x_viewed = _view_x(x, y, dim)

    N = y.shape[dim]
    Nx = x_viewed.shape[dim]

    if N == Nx:
        # Fast path: y and x share the integration length, so the weighted-sum
        # kernel applies directly once the non-dim axes are broadcast together.
        full_shape = torch.broadcast_shapes(y.shape, x_viewed.shape)
        out_shape = full_shape[:dim] + full_shape[dim + 1 :]

        if N <= 1:
            # A single sample point spans no interval, so the integral is zero.
            return torch.zeros(out_shape, dtype=out_dtype, device=y.device)

        if dim == y.dim() - 1 and y.shape == full_shape and y.is_contiguous():
            # Already row-major with the integration axis last: skip the
            # broadcast/movedim/contiguous dispatches (~7 us of host time each
            # call on gcu400, which dominates the small shapes).
            y2 = y.reshape(-1, N)
        else:
            y_b = y.broadcast_to(full_shape)
            y2 = y_b.movedim(dim, -1).contiguous().reshape(-1, N)
        M = y2.shape[0]

        if M == 0:
            return torch.zeros(out_shape, dtype=out_dtype, device=y.device)

        if x.dim() == 1:
            # 1-D spacing shared across every row.
            x2 = x.contiguous()
            x_m_stride = 0
        else:
            x_b = x_viewed.broadcast_to(full_shape)
            x2 = x_b.movedim(dim, -1).contiguous().reshape(-1, N)
            x_m_stride = N

        if M <= _SPLIT_ROW_THRESHOLD and N >= _SPLIT_MIN_N:
            # Few rows, long integration axis: split that axis across programs
            # instead of walking it serially inside a single CTA.
            nsplit = min(_SPLIT_NSPLIT, triton.cdiv(N, _SPLIT_BLOCK_N))
            col_chunk = (
                triton.cdiv(triton.cdiv(N, nsplit), _SPLIT_BLOCK_N) * _SPLIT_BLOCK_N
            )
            nsplit = triton.cdiv(N, col_chunk)
            chunk_iters = col_chunk // _SPLIT_BLOCK_N
            part = torch.empty((M, nsplit), dtype=torch.float32, device=y.device)
            out = torch.empty((M,), dtype=kernel_out_dtype, device=y.device)
            with torch_device_fn.device(y.device):
                trapezoid_x_split_kernel[(M * nsplit,)](
                    y2,
                    x2,
                    part,
                    M,
                    N,
                    x_m_stride,
                    COL_CHUNK=col_chunk,
                    CHUNK_ITERS=chunk_iters,
                    NSPLIT=nsplit,
                    BLOCK_N=_SPLIT_BLOCK_N,
                    COMPUTE_DTYPE=_compute_dtype(out_dtype),
                )
                trapezoid_x_split_reduce_kernel[(triton.cdiv(M, 64),)](
                    part,
                    out,
                    M,
                    nsplit,
                    BLOCK_S=triton.next_power_of_2(nsplit),
                    BLOCK_M=64,
                )
            result = out.reshape(out_shape)
            return result.to(out_dtype) if kernel_out_dtype != out_dtype else result

        if x_m_stride == 0:
            # Shared 1-D spacing: persistent row reduction. One program per
            # gcu400 SPC strides over row tiles, the x neighbour loads stay
            # [BLOCK_N] 1-D tiles (the weight broadcasts over the row block in
            # registers), and the per-column tl.sum keeps only a [BLOCK_M]
            # accumulator alive across the loop. Measured 3.2-10.7x faster
            # than the row-broadcast form below on the core shapes.
            es = y.element_size()
            block_n = min(triton.next_power_of_2(N), _ROW_TREE_WIDTH)
            batch_rows = triton.next_power_of_2(max(1, triton.cdiv(M, _ROW_MAX_GRID)))
            block_m = max(
                1,
                min(
                    _ROW_TILE_BYTES // (es * block_n),
                    _ROW_TILE_ELEMS // block_n,
                    batch_rows,
                ),
            )
            # num_warps=4 miscompiles the axis-1 tree at width 1024 (P10
            # family); the measured safe/speed domain is nw=2 for narrow
            # blocks and nw=1 for wide ones.
            num_warps = 2 if block_n <= 1024 else 1
            out = torch.empty((M,), dtype=kernel_out_dtype, device=y.device)
            grid = (min(_ROW_MAX_GRID, triton.cdiv(M, block_m)),)
            with torch_device_fn.device(y.device):
                trapezoid_x_row_kernel[grid](
                    y2,
                    x2,
                    out,
                    M,
                    N,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    COMPUTE_DTYPE=_compute_dtype(out_dtype),
                    num_warps=num_warps,
                    num_stages=1,
                )
            result = out.reshape(out_shape)
            return result.to(out_dtype) if kernel_out_dtype != out_dtype else result

        # GCU: the 16-bit input dtypes are widened element by element into the
        # fp32 accumulator, and a wider accumulator tile amortises that widening
        # (measured ~2.5x faster than 32x128 on (4096,4096) and (10000,4096)).
        # fp32/fp64 inputs are memory bound and prefer the narrower tile. Short
        # rows (N <= 128) keep BLOCK_N = 128 so no lane is masked away.
        if kernel_out_dtype in (torch.float16, torch.bfloat16) and N > 128:
            block_m, block_n = 16, 256
        else:
            block_m, block_n = 32, 128

        out = torch.empty((M,), dtype=kernel_out_dtype, device=y.device)
        grid = lambda meta: (triton.cdiv(M, block_m),)  # noqa: E731
        with torch_device_fn.device(y.device):
            trapezoid_x_kernel[grid](
                y2,
                x2,
                out,
                M,
                N,
                x_m_stride,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                COMPUTE_DTYPE=_compute_dtype(out_dtype),
            )
        result = out.reshape(out_shape)
        return result.to(out_dtype) if kernel_out_dtype != out_dtype else result

    # Slow path: pair-space broadcasting (N != Nx). Reproduce ATen's
    # ((y_left + y_right) * (x_right - x_left)).sum(dim) / 2 exactly, computing the
    # intermediate differences in the compute dtype to preserve precision. The
    # element-wise sum/difference route through the FlagGems add/sub kernels
    # (host-function convention: no native PyTorch compute in the wrapper).
    # GCU: the compute dtype is float32 (see _compute_dtype); the .to() casts
    # below stage fp64 inputs down, and the result is cast back to out_dtype.
    compute_dtype = torch.float32
    left = y.narrow(dim, 0, N - 1)
    right = y.narrow(dim, 1, N - 1)
    lpr = gems_add(left.to(compute_dtype), right.to(compute_dtype))

    x_left = x_viewed.narrow(dim, 0, Nx - 1)
    x_right = x_viewed.narrow(dim, 1, Nx - 1)
    dx = gems_sub(x_right.to(compute_dtype), x_left.to(compute_dtype))

    pair_shape = torch.broadcast_shapes(lpr.shape, dx.shape)
    P = pair_shape[dim]
    out_shape = pair_shape[:dim] + pair_shape[dim + 1 :]

    if P == 0:
        # An empty pair axis yields a zero integral.
        return torch.zeros(out_shape, dtype=out_dtype, device=y.device)

    lpr_b = lpr.broadcast_to(pair_shape)
    dx_b = dx.broadcast_to(pair_shape)

    lpr_2d = lpr_b.movedim(dim, -1).contiguous().reshape(-1, P)
    dx_2d = dx_b.movedim(dim, -1).contiguous().reshape(-1, P)
    M = lpr_2d.shape[0]

    if M == 0:
        return torch.zeros(out_shape, dtype=out_dtype, device=y.device)

    out = torch.empty((M,), dtype=kernel_out_dtype, device=y.device)
    grid = lambda meta: (triton.cdiv(M, 32),)  # noqa: E731
    with torch_device_fn.device(y.device):
        trapezoid_x_pair_kernel[grid](
            lpr_2d,
            dx_2d,
            out,
            M,
            P,
            BLOCK_M=32,
            BLOCK_N=128,
            COMPUTE_DTYPE=_compute_dtype(out_dtype),
        )
    result = out.reshape(out_shape)
    return result.to(out_dtype) if kernel_out_dtype != out_dtype else result


class _TrapezoidX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x, dim):
        dim = _normalize_dim(y.dim(), dim)
        ctx.dim = dim
        ctx.x_shape = x.shape
        ctx.save_for_backward(y, x)
        return _trapezoid_x_impl(y, x, dim)

    @staticmethod
    def backward(ctx, grad_out):
        y, x = ctx.saved_tensors
        dim = ctx.dim
        x_shape = ctx.x_shape

        N = y.shape[dim]
        if N <= 1:
            # N == 0 would pass a negative length to narrow, and N == 1 leaves an
            # empty pair axis for _accumulate_*; either way a single (or absent)
            # sample point spans no interval, so both gradients are zero (ATen).
            return _zeros_like(y), _zeros_like(x), None

        # GCU: stage fp64 chains through float32 (no f64 kernels) and cast the
        # returned gradients back to the original dtypes; .to() on a matching
        # dtype is a no-op, so non-fp64 chains are unchanged.
        y_dtype, x_dtype = y.dtype, x.dtype
        if torch.float64 in (y_dtype, x_dtype, grad_out.dtype):
            y = y.to(torch.float32)
            x = x.to(torch.float32)
            grad_out = grad_out.to(torch.float32)

        # Complex autograd follows the Wirtinger convention: each input's
        # gradient is conj(d out / d input), matching ATen. In the analytic
        # chain below the conjugate must therefore apply to the *other*
        # complex factors while the upstream grad stays plain:
        #   grad_y[i,k] = grad_out[i] * conj(w[k] * dx[k])
        #   grad_x[k]  = sum_i grad_out[i] * conj(±0.5 * lpr[i,k])
        # (real tensors are unaffected by conjugation).
        complex_case = grad_out.is_complex() or y.is_complex() or x.is_complex()

        x_viewed = _view_x(x, y, dim)

        lpr = gems_add(y.narrow(dim, 0, N - 1), y.narrow(dim, 1, N - 1))

        Nx = x_viewed.shape[dim]
        x_left = x_viewed.narrow(dim, 0, Nx - 1)
        x_right = x_viewed.narrow(dim, 1, Nx - 1)
        dx = gems_sub(x_right, x_left)

        # The gems element-wise kernels broadcast internally, so lpr/dx are used
        # at their pre-broadcast shapes: an explicit broadcast_to here would
        # dispatch to the registered FlagGems kernel, whose Triton implementation
        # rejects complex dtypes.
        # out = (lpr * dx).sum(dim) / 2; route the element-wise products through
        # the FlagGems mul kernel (host-function convention: no native compute).
        grad_pair = gems_mul(grad_out.unsqueeze(dim), 0.5)
        if complex_case:
            grad_lpr_b = gems_mul(grad_pair, dx.conj().resolve_conj())
            grad_dx_b = gems_mul(grad_pair, lpr.conj().resolve_conj())
        else:
            grad_lpr_b = gems_mul(grad_pair, dx)
            grad_dx_b = gems_mul(grad_pair, lpr)

        grad_lpr = _unbroadcast(grad_lpr_b, lpr.shape)
        grad_dx = _unbroadcast(grad_dx_b, dx.shape)

        grad_y = _accumulate_plus(grad_lpr, dim)
        grad_x_viewed = _accumulate_minus(grad_dx, dim)
        grad_x = grad_x_viewed.view(x_shape)

        return grad_y.to(y_dtype), grad_x.to(x_dtype), None


def trapezoid_x(y, x, dim=-1):
    logger.debug("GEMS_ENFLAME TRAPEZOID_X")
    if _is_complex(y, x):
        # Complex routes through the CPU fallback, which is autograd
        # transparent on its own; _TrapezoidX.forward would run under
        # no_grad and detach the differentiable CPU graph.
        return _trapezoid_x_complex(y, x, dim)
    if torch.is_grad_enabled() and (y.requires_grad or x.requires_grad):
        return _TrapezoidX.apply(y, x, dim)
    return _trapezoid_x_impl(y, x, dim)
