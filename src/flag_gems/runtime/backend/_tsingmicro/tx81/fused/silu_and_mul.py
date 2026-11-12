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

"""TsingMicro (txda) implementation of ``silu_and_mul``.

The serving profile is dominated by two shapes emitted by vLLM:

* the dense path calls ``silu_and_mul`` with ``A/B = buffer[:, :N]`` and
  ``buffer[:, N:]``, i.e. two strided views of one ``[M, 2N]`` buffer;
* the MoE path calls ``silu_and_mul_out`` with the same views and a dense
  ``[M, N]`` output.

``N`` is the per-TP intermediate size (64 for Qwen3.5 TP8) and ``M`` is the
number of tokens (decode: 1..24) or tokens x experts (prefill: 1e2..1.5e4).

The generic ``pointwise_dynamic`` path caps the grid at 16 CTAs and runs one
warp per CTA while walking the rows in a Python-generated loop, which leaves
most of the memory pipeline idle for the large MoE shapes and pays the
LibEntry metadata cost for the tiny decode shapes.

The kernels below take the collapsed ``(rows, N)`` view directly, use the row
strides of each operand (so the strided gate/up views are read in place), and
scale the grid with the row count.  A cached compiled-kernel launcher removes
the Triton JIT argument-binding cost, which dominates the decode shapes.
"""

import functools
import inspect
import logging
import os
import threading
import time

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

try:
    import triton.experimental.tle.language as _tle_ext
except ImportError:  # pragma: no cover - only available on txda
    _tle_ext = None

try:
    from triton.backends.tsingmicro.precision import get_precision_mode
except ImportError:  # pragma: no cover - only available on txda

    def get_precision_mode():
        return os.environ.get("PRECISION_MODE", "0")


logger = logging.getLogger(__name__)

# 16-byte aligned pointers let the backend emit vector loads/stores.
_ALIGN_BYTES = 16
# Keep one CTA's tile within a fraction of the 3 MB on-chip SPM.
_MAX_TILE_ELEMS = 65536
# The packed kernel loads the full 2N-wide row (A|B) and slices it with
# _tle_ext.extract_tile; the hardware tile can be twice as wide as the split kernel.
_MAX_PACKED_TILE_ELEMS = 131072
# One column tile covers short rows; wider rows are tiled along columns.
_MAX_BLOCK_N = 1024
# Under precision modes 0/1 the backend vectorises i32 address arithmetic as
# f32; offsets above 2**24 would silently wrap.  Larger tensors take the
# generic path, whose indexing is already precision-mode aware.
_MAX_FAST_OFFSET = 1 << 24
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# Cache of compiled-kernel launchers: key -> callable(*kernel args).
# The launcher captures the compiled binary and the grid, but not tensor
# pointers, so it is safe to reuse across calls/streams.
_DIRECT_LAUNCHERS = {}
# Triton compilation is not thread safe; only cache misses take this lock.
_LAUNCH_LOCK = threading.Lock()


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def silu_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_silu = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return x_silu * y


@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")], num_outputs=2
)
@triton.jit
def silu_and_mul_grad_kernel(x, y, dgrad):
    x_fp32 = x.to(tl.float32)
    sig = 1 / (1 + tl.exp(-x_fp32))
    x_silu = x_fp32 * sig
    d_x_silu = sig * (1 + x_fp32 * (1 - sig))
    dx = d_x_silu * dgrad * y
    dy = dgrad * x_silu
    return dx, dy


@triton.jit(do_not_specialize=["rows", "n", "sa", "sb", "so"])
def _silu_and_mul_fast_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    rows,
    n,
    sa,
    sb,
    so,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """res[i, j] = silu(a[i, j]) * b[i, j] over a collapsed (rows, n) view.

    ``sa``/``sb``/``so`` are the row strides; the innermost stride is 1.
    ``BLOCK_N == n`` (EVEN_N) lets the compiler drop the column mask, which is
    the hot path (n == 64).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rm[:, None] < rows
    if EVEN_N:
        mask = row_mask
    else:
        mask = row_mask & (rn[None, :] < n)
    a = tl.load(a_ptr + rm[:, None] * sa + rn[None, :], mask=mask, other=0.0)
    b = tl.load(b_ptr + rm[:, None] * sb + rn[None, :], mask=mask, other=0.0)
    x = a.to(tl.float32)
    y = b.to(tl.float32)
    res = x * tl.sigmoid(x) * y
    tl.store(o_ptr + rm[:, None] * so + rn[None, :], res, mask=mask)


@triton.jit(do_not_specialize=["rows"])
def _silu_and_mul_packed_kernel(
    a_ptr,
    o_ptr,
    rows,
    STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Optimised kernel for the vLLM gate/up layout.

    ``A`` and ``B`` are the two halves of one row-major ``[rows, 2 * N]``
    buffer (``B`` starts exactly ``N`` elements after ``A``).  Loading the full
    row in one contiguous tile and slicing it with ``_tle_ext.extract_tile`` keeps
    the 256-byte row in a single memory transaction; loading the halves with
    two separate strided loads halves the attainable bandwidth on TX81.

    This is the hot path: 79.8 us for the 15368-row MoE shape vs 126 us for the
    two-load kernel (profiled device time).
    """
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, 2 * N)
    mask = rm[:, None] < rows
    v = tl.load(a_ptr + rm[:, None] * STRIDE + rn[None, :], mask=mask, other=0.0)
    a = _tle_ext.extract_tile(v, (0, 0), (BLOCK_M, N))
    b = _tle_ext.extract_tile(v, (0, 1), (BLOCK_M, N))
    x = a.to(tl.float32)
    y = b.to(tl.float32)
    res = x * tl.sigmoid(x) * y
    tl.store(
        o_ptr + rm[:, None] * OUT_STRIDE + tl.arange(0, N)[None, :],
        res,
        mask=mask,
    )


def _next_pow2(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _row_stride(t: torch.Tensor, n: int):
    """Return the stride between consecutive rows of the collapsed view.

    ``None`` means the tensor cannot be viewed as ``[rows, n]`` with a single
    row stride without a copy (e.g. a transposed or oddly gapped layout).
    Size-1 leading dimensions are ignored because their stride is irrelevant.
    """
    nd = t.dim()
    if nd == 0:
        return 0 if n == 1 else None
    last = nd - 1
    if n != 1 and t.stride(last) != 1:
        return None
    if nd == 1:
        return 0
    row_stride = None
    inner = 1
    for dim in range(nd - 2, -1, -1):
        if t.shape[dim] == 1:
            continue
        if row_stride is None:
            row_stride = t.stride(dim)
        elif t.stride(dim) != inner * row_stride:
            return None
        inner *= t.shape[dim]
    return 0 if row_stride is None else row_stride


def _fast_path_layout(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    """Validate the serving layout and return ``(rows, n, sa, sb, so, packed)``.

    The fast path requires equal shapes, a unit innermost stride, 16-byte
    aligned base pointers, and a linearly addressable row-major collapse.
    Anything else falls back to the generic ``pointwise_dynamic`` kernel.
    """
    if not (torch.is_tensor(A) and torch.is_tensor(B) and torch.is_tensor(out)):
        return None
    if A.dtype not in _SUPPORTED_DTYPES:
        return None
    if B.dtype != A.dtype or out.dtype != A.dtype:
        return None
    shape = A.shape
    if B.shape != shape or out.shape != shape or len(shape) == 0:
        return None
    n = shape[-1]
    if n <= 0 or A.numel() == 0:
        return None
    sa = _row_stride(A, n)
    sb = _row_stride(B, n)
    so = _row_stride(out, n)
    if sa is None or sb is None or so is None:
        return None
    if (A.data_ptr() | B.data_ptr() | out.data_ptr()) & (_ALIGN_BYTES - 1):
        return None
    rows = A.numel() // n
    if rows <= 0:
        return None
    if rows >= _MAX_FAST_OFFSET:
        return None
    if rows * max(sa, sb, so) + 2 * n >= _MAX_FAST_OFFSET:
        return None
    # The packed kernel reads A|B as one contiguous row.  Requiring the same
    # row stride and B immediately after A also rules out separate dense
    # tensors that merely happen to be adjacent (sa == n < 2 * n).
    packed = (
        _tle_ext is not None
        and n <= _MAX_BLOCK_N
        and (n & (n - 1)) == 0
        and sb == sa
        and B.data_ptr() == A.data_ptr() + n * A.element_size()
        and (rows == 1 or sa >= 2 * n)
    )
    return rows, n, sa, sb, so, packed


def _choose_config(rows: int, n: int, packed: bool = False):
    """Pick (BLOCK_N, BLOCK_M, num_warps) for the collapsed shape.

    The thresholds were tuned on TX81 with the serving shapes.  Small decode
    launches are dominated by the ~150 us launch floor; larger shapes are
    scaled so the grid lands near one wave of the 16 SMs.
    """
    block_n = min(_next_pow2(n), _MAX_BLOCK_N)
    even_n = block_n == n
    if rows <= 32:
        block_m = min(_next_pow2(rows), 32)
        num_warps = 4 if block_m <= 16 else 8
    elif rows <= 256:
        block_m = min(max(_next_pow2((rows + 7) // 8), 8), 32)
        num_warps = 1
    elif rows <= 2048:
        block_m = min(max(_next_pow2((rows + 3) // 4), 64), 256)
        num_warps = 8 if block_m >= 128 else 4
    else:
        block_m = min(max(_next_pow2((rows + 15) // 16), 256), 2048)
        num_warps = 4
    tile_width = 2 * block_n if packed else block_n
    tile_budget = _MAX_PACKED_TILE_ELEMS if packed else _MAX_TILE_ELEMS
    block_m = min(block_m, max(1, tile_budget // tile_width))
    return block_n, block_m, num_warps, even_n


def _make_direct_launcher(kernel, runner, grid):
    """Build a minimal launcher that skips Triton's per-call Python wrapper.

    ``runner`` (``CompiledKernel.__getitem__``) resolves the current device and
    stream and builds launch metadata on every call.  When no profiler hooks
    are installed we can call the txda ``TXDALauncher.launch`` C function
    directly with pre-resolved metadata, saving ~15 us/call.  The returned
    launcher checks the profiler hooks on every call: with hooks installed it
    uses ``runner`` (so kernel traces are complete), otherwise it uses the
    direct C launch.  Any failure falls back to the normal runner.
    """
    try:
        import triton.runtime.knobs as knobs
        from triton.backends.tsingmicro.driver import logger as launch_logger
        from triton.backends.tsingmicro.driver import txda_tools
        from triton.backends.tsingmicro.logger_config import (
            logger_to_custom_level_number,
        )

        c_launcher = kernel._run
        if c_launcher is None or not hasattr(c_launcher, "launch"):
            return runner
        metadata = kernel.metadata
        dump_args = txda_tools.is_dump_args_profile()
        dump_dir = txda_tools.get_dump_dir()
        log_level = logger_to_custom_level_number(launch_logger)
        so_key = metadata.so_key
        kernel_path = metadata.kernel_path
        func_name = c_launcher.func_name
        packed_metadata = kernel.packed_metadata
        function = kernel.function
        grid_x, grid_y = grid
        counter = [0]

        def direct(*args):
            counter[0] += 1
            device = torch.txda.current_device()
            stream = torch.txda.current_stream(device).txda_stream
            result = c_launcher.launch(
                device,
                so_key,
                kernel_path,
                func_name,
                dump_args,
                dump_dir,
                log_level,
                counter[0],
                grid_x,
                grid_y,
                1,
                stream,
                function,
                packed_metadata,
                None,
                None,
                None,
                *args,
            )
            if getattr(result, "res", 0) != 0:
                raise RuntimeError(f"txda kernel launch failed: {result.res}")

        def launcher(*args):
            # A profiler may start after this launcher was cached; route through
            # the standard runner whenever launch hooks are installed so kernel
            # traces stay complete.  In normal serving this uses the direct C
            # launch (the ~15 us/call saving).
            if (
                knobs.runtime.launch_enter_hook is not None
                or knobs.runtime.launch_exit_hook is not None
            ):
                return runner(*args)
            return direct(*args)

        return launcher
    except Exception:  # pragma: no cover - private API guard
        return runner


def _launch_fast(A, B, out, layout):
    rows, n, sa, sb, so, packed = layout
    block_n, block_m, num_warps, even_n = _choose_config(rows, n, packed)
    grid_m = triton.cdiv(rows, block_m)
    grid_n = 1 if packed else triton.cdiv(n, block_n)
    precision_mode = get_precision_mode()
    layout_key = (sa, so) if packed else (0, 0)
    key = (
        A.dtype,
        packed,
        n,
        block_n,
        block_m,
        num_warps,
        even_n,
        grid_m,
        grid_n,
        layout_key,
        precision_mode,
    )
    launcher = _DIRECT_LAUNCHERS.get(key)
    if launcher is None:
        with _LAUNCH_LOCK:
            launcher = _DIRECT_LAUNCHERS.get(key)
            if launcher is None:
                if len(_DIRECT_LAUNCHERS) >= 256:
                    _DIRECT_LAUNCHERS.clear()
                if packed:
                    kernel = _silu_and_mul_packed_kernel.warmup(
                        A,
                        out,
                        rows,
                        STRIDE=sa,
                        OUT_STRIDE=so,
                        N=n,
                        BLOCK_M=block_m,
                        num_warps=num_warps,
                        grid=(1,),
                    )
                else:
                    kernel = _silu_and_mul_fast_kernel.warmup(
                        A,
                        B,
                        out,
                        rows,
                        n,
                        sa,
                        sb,
                        so,
                        BLOCK_N=block_n,
                        BLOCK_M=block_m,
                        EVEN_N=even_n,
                        num_warps=num_warps,
                        grid=(1,),
                    )
                runner = kernel[(grid_m, grid_n, 1)]
                launcher = _make_direct_launcher(kernel, runner, (grid_m, grid_n))
                _DIRECT_LAUNCHERS[key] = launcher
    if packed:
        launcher(A, out, rows, sa, so, n, block_m)
    else:
        launcher(A, B, out, rows, n, sa, sb, so, block_n, block_m, even_n)


def _fmt_value(value):
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, stride={tuple(value.stride())}, "
            f"dtype={value.dtype})"
        )
    return repr(value)


def log_args(func):
    """Print each argument on one line and the wall-clock execution time in ms.

    Gated behind ``FLAG_GEMS_LOG_SILU_AND_MUL=1`` so the per-call stdout
    formatting/flush doesn't tax the serving hot path.

    ``inspect.signature``/``bind`` are resolved once at decoration time: running
    them per call burns measurable CPU in the 8-worker serving run and inflates
    the very latency this instrumentation reports.
    """

    enabled = os.getenv("FLAG_GEMS_LOG_SILU_AND_MUL", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    once = os.getenv("FLAG_GEMS_LOG_SILU_AND_MUL_ONCE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    param_names = tuple(inspect.signature(func).parameters)
    seen = set()

    def _shape_key(value):
        if torch.is_tensor(value):
            return (value.shape, value.stride(), value.dtype)
        return value

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not enabled:
            return func(*args, **kwargs)

        prefix = f"[{func.__name__}]"
        key = tuple(_shape_key(value) for value in args) + tuple(
            (name, _shape_key(value)) for name, value in kwargs.items()
        )
        if not once or key not in seen:
            seen.add(key)
            for name, value in zip(param_names, args):
                print(f"{prefix} {name}: {_fmt_value(value)}", flush=True)
            positional = set(param_names[: len(args)])
            for name, value in kwargs.items():
                if name not in positional:
                    print(f"{prefix} {name}: {_fmt_value(value)}", flush=True)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"{prefix} elapsed_ms={elapsed_ms:.3f}", flush=True)

    return wrapper


class SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        logger.debug("GEMS_TSINGMICRO SILU_AND_MUL_FORWARD")
        ctx.save_for_backward(A, B)
        out = torch.empty(A.shape, dtype=A.dtype, device=A.device)
        layout = _fast_path_layout(A, B, out)
        if layout is not None:
            _launch_fast(A, B, out, layout)
        else:
            silu_and_mul_kernel(A, B, out0=out)
        return out

    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A, grad_B = silu_and_mul_grad_kernel(A, B, grad_output)
        return grad_A, grad_B


@log_args
def silu_and_mul(A, B):
    logger.debug("GEMS_TSINGMICRO SILU_AND_MUL")
    # Skip the autograd Function (and its dispatch overhead) in inference.
    if (torch.is_tensor(A) and A.requires_grad) or (
        torch.is_tensor(B) and B.requires_grad
    ):
        return SiluAndMul.apply(A, B)
    out = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    layout = _fast_path_layout(A, B, out)
    if layout is not None:
        _launch_fast(A, B, out, layout)
    else:
        silu_and_mul_kernel(A, B, out0=out)
    return out


@log_args
def silu_and_mul_out(A, B, out):
    logger.debug("GEMS_TSINGMICRO SILU_AND_MUL_OUT")
    layout = _fast_path_layout(A, B, out)
    if layout is not None:
        _launch_fast(A, B, out, layout)
    else:
        silu_and_mul_kernel(A, B, out0=out)
    return out
