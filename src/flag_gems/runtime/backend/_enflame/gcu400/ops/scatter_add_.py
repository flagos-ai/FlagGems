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

import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import dim_compress
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer
from flag_gems.utils.shape_utils import restride_dim

from .scatter import is_last_dim_flat, scatter_last_dim

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native topsaten delegation (precedent set in mm.py)
#
# On GCU400, `tl.atomic_add` is lowered to a per-element extern call (`tops_atomic_add_*`),
# measured at the machine level at ~0.45G updates/s of serialization, independent of grid/num_warps/BLOCK/dtype
# (seven-axis probe calibration, 2026-11-16). The native topsaten scatter kernel (transpose +
# dedicated lastdim/rows kernel) reaches 0.8~3G updates/s for large-row shapes, while Triton has
# no equivalent primitive (no DSM scatter-add, no fast atomics). Hence atomically dense shapes with
# enough rows delegate directly to the native aten::scatter_add_ kernel captured at import time
# (zero dispatch overhead via call_boxed), which also fixes the dim!=last large-shape grid.x>65535 crash.
# ---------------------------------------------------------------------------
_NATIVE_SCATTER_ADD_ = None
_DEVICE_KEYSET = None
if hasattr(torch.library, "get_kernel"):
    try:
        _NATIVE_SCATTER_ADD_ = torch.library.get_kernel(
            torch.ops.aten.scatter_add_.default, runtime.device.dispatch_key
        )
        _DEVICE_KEYSET = torch._C.DispatchKeySet(
            getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
        )
    except Exception:  # noqa: B902
        _NATIVE_SCATTER_ADD_ = None

# Budget for the one-hot comparison-reduction kernel for last-dim scatter_add: out[m,c] = inp[m,c] +
# sum_j src[m,j] * (index[m,j] == c). Comparison reduction measures ~570G MAC/s, far above
# the 0.45G/s atomic wall; the cost per update is proportional to the out row width N, so enable it only when both N and
# the total
# MAC count are bounded ((512,512) measured: one-hot 474us vs atomic 1248us vs
# native 778us; beyond N>=2048 the per-update cost overtakes the atomic path).
_SCATTER_ADD_ONEHOT_MACS = 1 << 28
_SCATTER_ADD_ONEHOT_MAX_N = 1024
# For the last-dim flat layout, native parallelism grows with row count (at 256 rows native is slower than
# the atomic path: (256,256) atomic 346us vs native 387us; from 512 rows native consistently
# wins: (512,512) 778 vs 1248us), so keep the atomic path below the row-count threshold.
_SCATTER_ADD_NATIVE_MIN_ROWS = 512


@triton.jit
def scatter_add_kernel_1(
    index_dim_n,
    inp_dim_n,
    out_ptr,
    index_ptr,
    src_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * LOOP
    arange = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + arange
    mask = offsets < n_elements
    for loop_iter in tl.static_range(LOOP):
        src_index_offsets = block_start + arange
        src_tensor = tl.load(src_ptr + src_index_offsets, mask=mask, other=0)
        index_tensor = tl.load(index_ptr + src_index_offsets, mask=mask, other=0)
        out_offsets = src_index_offsets // index_dim_n * inp_dim_n + index_tensor
        tl.atomic_add(out_ptr + out_offsets, src_tensor, mask=mask, sem="relaxed")
        block_start += BLOCK_SIZE


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("import flag_gems")
    code.newline()
    code.newline()
    return code


def generate_scatter_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # make the inlined function visible in the context
    code.newline()

    # the autotune function
    code.writeline("def heur_block(args):")
    with code.indent():
        code.writeline("return 8192")
    code.newline()
    code.newline()

    code.writeline("def loop_count(args):")
    with code.indent():
        code.writeline("return 4")
    code.newline()
    code.newline()

    # the decorators
    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
        code.writeline("}")
    code.writeline(")")
    inp_stride_vars = ",".join(f"'inp_stride_{i}'" for i in range(rank))
    index_stride_vars = ",".join(f"'index_stride_{i}'" for i in range(rank))
    src_stride_vars = ",".join(f"'src_stride_{i}'" for i in range(rank))
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code.writeline(
        f"@triton.jit(do_not_specialize=['N','stride_dim','inp_size_dim',"
        f"{inp_stride_vars},{index_stride_vars},{src_stride_vars},{shape_vars}])"
    )

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("src_strided,")
            code.writeline("index,")
            code.writeline("inp,")
            code.writeline("out,")

            stride_args = ", ".join(f"inp_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for inp")

            stride_args = ", ".join(f"index_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for index")

            stride_args = ", ".join(f"src_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for src")

            shape_args = ", ".join(f"shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape")
            code.writeline("inp_size_dim,")
            code.writeline("stride_dim,")
            code.writeline("N,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")

    code.writeline("):")

    # Kernel Code
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")

        #   1. Calculate inp_offsets and idx_offsets
        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N")
            code.writeline("cur_idx = offsets")
            code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("src_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            for i in range(rank)[::-1]:
                code.writeline(f"mod = cur_idx % shape_{i}")
                code.writeline(f"inp_offsets += mod * inp_stride_{i}")
                code.writeline(f"idx_offsets += mod * index_stride_{i}")
                code.writeline(f"src_offsets += mod * src_stride_{i}")
                if i != 0:
                    code.writeline(f"cur_idx = cur_idx // shape_{i}")

            #   2. Use offsets to scatter
            code.writeline(
                "cur_src = tl.load(src_strided + src_offsets, mask=mask, other=0)"
            )
            code.writeline(
                "cur_index = tl.load(index + idx_offsets, mask=mask, other=0)"
            )
            code.writeline("dim_offsets = cur_index * stride_dim")
            code.writeline("inp_offsets += dim_offsets")
            code.newline()
            code.writeline(
                "tl.atomic_add(out + inp_offsets, cur_src, mask=mask, sem='relaxed')"
            )
            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def parameter_for_wrapper() -> str:
    # src_strided, index, inp, out, dim, M, N
    parameters: List[str] = []

    parameters.append("src_strided")
    parameters.append("index")
    parameters.append("inp")
    parameters.append("out")
    parameters.append("dim_size")
    parameters.append("dim_stride")
    parameters.append("N")

    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("inp_strides = list(inp.stride())")
        code.writeline("index_strides = index.stride()")
        code.writeline("src_strides = src_strided.stride()")
        code.writeline("index_shapes = list(index.shape)")
        code.writeline("inp_size_dim = dim_size")
        code.writeline("stride_dim = dim_stride")

        # kernel launch
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(N, meta["BLOCK"] * meta["LOOP"]), ')
        code.writeline(")")
        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)
        with code.indent():
            code.writeline("src_strided, index, inp, out, ")
            if rank > 0:
                s = ", ".join(f"inp_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"src_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                code.writeline("inp_size_dim,")
                code.writeline("stride_dim,")
                code.writeline("N,")

        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [src_strided, index, inp, out, dim, M, N]
    shape = inputs[1].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_scatter_kernel(rank, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class ScatterFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_scatter_add_wrapper",
                "_scatter_add_jit_function",
                code,
            )

            file_name = f"scatter_add_rank_{key}_pid_{self.pid}.py"

            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_add_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_func = ScatterFunction()


@triton.jit(do_not_specialize=["idx_ncols", "src_stride0", "out_ncols"])
def scatter_add_2d_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    idx_ncols,
    src_stride0,
    out_ncols,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Direct 2-D grid: program_id(1) is the row, program_id(0) covers column blocks.

    Removes the generic path's int64 flat offset and runtime div/mod; the in-row offsets of
    index/src/out are all linear expressions, and only the scattered out column comes from index.
    """
    row = tl.program_id(1)
    col_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = col_offsets < idx_ncols
    index_value = tl.load(
        index_ptr + row * idx_ncols + col_offsets, mask=mask, other=0
    ).to(tl.int32)
    src_value = tl.load(src_ptr + row * src_stride0 + col_offsets, mask=mask, other=0)
    if DIM == 0:
        out_offsets = index_value * out_ncols + col_offsets
    else:
        out_offsets = row * out_ncols + index_value
    tl.atomic_add(out_ptr + out_offsets, src_value, mask=mask, sem="relaxed")


def scatter_add_0(inp, dim, index, src):
    logger.debug("GEMS_ENFLAME SCATTER_ADD_0")
    dtype_convert = False
    if inp.dtype == torch.float16 or inp.dtype == torch.bfloat16:
        out = inp.to(torch.float32)
        dtype_convert = True
    else:
        out = inp

    src_strided = src.as_strided(index.shape, src.stride())

    # 2D fast path: when index and out have the same number of rows and the last dim is contiguous, address directly
    # with a 2D grid.
    # The generic path uses a flat offset with runtime integer div/mod via row=offset//ncols / col=offset%ncols,
    # and the offset is int64; on GCU both add substantial time on top of the atomic accumulation.
    if (
        out.ndim == 2
        and index.ndim == 2
        and index.shape[0] == out.shape[0]
        and index.is_contiguous()
        and out.is_contiguous()
        and src_strided.stride(1) == 1
    ):
        n_rows, idx_ncols = index.shape
        out_ncols = out.shape[1]
        grid = lambda meta: (
            triton.cdiv(idx_ncols, meta["BLOCK"]),
            n_rows,
        )
        scatter_add_2d_kernel[grid](
            src_strided,
            index,
            out,
            idx_ncols,
            src_strided.stride(0),
            out_ncols,
            DIM=dim % 2,
            BLOCK=256,
        )
        if dtype_convert:
            return inp.copy_(out.to(src.dtype))
        return out

    # fp16/bf16 goes through `out = inp.to(torch.float32)`, materializing a tensor whose layout may differ
    # (slices with storage holes degrade to contiguous), so the kernel's addressing strides
    # must come from the `out` actually being written, otherwise atomic accumulation lands at the wrong offset.
    inp_restrided = restride_dim(out, dim, index.shape)
    dim_size = out.size(dim)
    dim_stride = out.stride(dim)
    N = index.numel()

    _scatter_func(
        src_strided,
        index,
        inp_restrided,
        out,
        dim_size,
        dim_stride,
        N,
    )
    if dtype_convert:
        return inp.copy_(out.to(src.dtype))
    return out


def clip_tensor_to_shape(b, a):
    target_shape = a.shape
    slices = [
        slice(0, min(b.shape[i], target_shape[i])) for i in range(len(target_shape))
    ]
    clipped_b = b[tuple(slices)]
    return clipped_b


@triton.jit(
    do_not_specialize=[
        "rows",
        "width",
        "out_width",
        "idx_row_stride",
        "src_row_stride",
        "inp_row_stride",
        "out_row_stride",
    ]
)
def scatter_add_onehot_kernel(
    src_ptr,
    index_ptr,
    inp_ptr,
    out_ptr,
    rows,
    width,
    out_width,
    idx_row_stride,
    src_row_stride,
    inp_row_stride,
    out_row_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """one-hot compare-reduce for trailing-dim scatter_add (no atomics).

    Each row belongs entirely to one program (the leading dim chain is homogeneous, see
    out[m, c] = inp[m, c] + sum_j src[m, j] * (index[m, j] == c).
    is_last_dim_flat), accumulating in fp32 and rounding once, no worse in accuracy than native
    per-element atomic accumulation. When inp and out share the layout this is an in-place
    update; when out is freshly allocated one kernel does clone+scatter (no separate clone).
    """
    pid = tl.program_id(0)
    num_cta = tl.num_programs(0)
    for row_block in range(pid, tl.cdiv(rows, BLOCK_M), num_cta):
        m0 = row_block * BLOCK_M
        for mi in tl.static_range(BLOCK_M):
            m = m0 + mi
            if m < rows:
                for n0 in range(0, out_width, BLOCK_N):
                    cols = n0 + tl.arange(0, BLOCK_N)
                    nmask = cols < out_width
                    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                    for j0 in range(0, width, BLOCK_J):
                        jidx = j0 + tl.arange(0, BLOCK_J)
                        jmask = jidx < width
                        idx = tl.load(
                            index_ptr + m * idx_row_stride + jidx,
                            mask=jmask,
                            other=-1,
                        ).to(tl.int32)
                        s = tl.load(
                            src_ptr + m * src_row_stride + jidx,
                            mask=jmask,
                            other=0.0,
                        ).to(tl.float32)
                        match = idx[None, :] == cols[:, None]
                        acc += tl.sum(tl.where(match, s[None, :], 0.0), axis=1)
                    base = tl.load(
                        inp_ptr + m * inp_row_stride + cols, mask=nmask, other=0.0
                    ).to(tl.float32)
                    res = base + acc
                    tl.store(
                        out_ptr + m * out_row_stride + cols,
                        res.to(out_ptr.dtype.element_ty),
                        mask=nmask,
                    )


def scatter_add_onehot(inp, index, src, out):
    """Host-side dispatch for the one-hot path. The trailing-dim rows of inp/out must satisfy is_last_dim_flat."""
    rank = index.dim()
    width = index.shape[-1]
    rows = index.numel() // width
    out_width = inp.shape[-1]
    # 2026-11-16 GCU400 parameter sweep: bm2/nw2 for rows<=128, bm1/nw1 for larger;
    # bn pinned to 64 (relaxed to 128 for small shapes), bj pinned to min(np2(width), 512).
    block_m = 2 if rows <= 128 else 1
    block_n = min(triton.next_power_of_2(out_width), 128 if rows <= 128 else 64)
    block_j = min(triton.next_power_of_2(width), 512)
    block_n = max(2, min(block_n, 32768 // block_j))
    block_j = max(2, block_j)
    grid = (min(triton.cdiv(rows, block_m), 4096),)
    scatter_add_onehot_kernel[grid](
        src,
        index,
        inp,
        out,
        rows,
        width,
        out_width,
        index.stride(rank - 2),
        src.stride(rank - 2),
        inp.stride(rank - 2),
        out.stride(rank - 2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_J=block_j,
        num_warps=2 if rows <= 64 else 1,
    )
    return out


def _native_ok(x, index):
    return (
        _NATIVE_SCATTER_ADD_ is not None
        and x.device.type == runtime.device.name
        and index.dtype == torch.long
    )


def scatter_add_1(x, dim, index, src):
    logger.debug("GEMS_ENFLAME SCATTER_ADD_1")
    index_dim_n = index.size(dim)
    inp_dim_n = x.size(dim)
    origin = x
    if dim != x.ndim - 1:
        x = dim_compress(x, dim)
    if dim != x.ndim - 1:
        src = dim_compress(src, dim)
    if dim != x.ndim - 1:
        index = dim_compress(index, dim)

    all_elem = max(x.numel(), index.numel())
    grid = lambda meta: (triton.cdiv(all_elem, meta["BLOCK_SIZE"] * meta["LOOP"]),)

    dtype_convert = False
    if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        dtype_convert = True
        x = x.to(torch.float32)

    scatter_add_kernel_1[grid](
        index_dim_n, inp_dim_n, x, index, src, all_elem, BLOCK_SIZE=256, LOOP=1
    )
    if dim != x.ndim - 1:
        order = [i for i in range(x.ndim - 1)]
        order.insert(dim, x.ndim - 1)
        if dtype_convert:
            return origin.copy_(x.to(src.dtype).permute(order))
        return x.permute(order)
    else:
        # The last-dim branch must also write back origin: after dtype conversion x is a new tensor,
        # returning it directly breaks the in-place contract (generic version uses the same origin.copy_)
        if dtype_convert:
            return origin.copy_(x.to(src.dtype))
        return x


def scatter_add_(x, dim, index, src):
    logger.debug("GEMS_ENFLAME SCATTER_ADD_")
    assert x.dim() == index.dim() and x.dim() == src.dim(), "Invalid dim"
    dim = dim % x.ndim
    assert dim >= 0 and dim < x.dim(), "Invalid dim"
    assert index.size(dim) <= src.size(dim), "Invalid src"
    equal_count = 0
    for d in range(x.dim()):
        if d != dim:
            assert index.size(d) <= x.size(d), "Invalid x"
            if index.size(d) == x.size(d):
                equal_count += 1
        else:
            if index.size(dim) >= x.size(dim):
                equal_count += 1

    src_view = src.as_strided(index.shape, src.stride())
    if is_last_dim_flat(x, dim, index, src_view):
        width = index.shape[-1]
        rows = index.numel() // width
        out_width = x.shape[-1]
        # one-hot comparison reduction: better than the atomic wall and the native pipeline for small row widths + small
        # MAC budgets
        if (
            out_width <= _SCATTER_ADD_ONEHOT_MAX_N
            and rows * out_width * width <= _SCATTER_ADD_ONEHOT_MACS
        ):
            scatter_add_onehot(x, index, src_view, x)
            return x
        # with enough rows the native dedicated kernel consistently wins (atomic wall 0.45G/s vs 0.8~3G/s)
        if _native_ok(x, index) and rows >= _SCATTER_ADD_NATIVE_MIN_ROWS:
            _NATIVE_SCATTER_ADD_.call_boxed(_DEVICE_KEYSET, x, dim, index, src)
            return x
        # Accumulate directly in the original dtype: an FP32 intermediate adds three extra full passes of traffic while
        # the atomic count is unchanged
        scatter_last_dim(x, dim, index, src_view, True)
        return x

    # Non-last-dim / non-flat layouts: the current Triton path (the generated kernel's i64 div/mod chain
    # + atomics) is uniformly slower than native in measurements (dim!=last 4.6~32x, including ~3x on small 3D shapes),
    # and scatter_add_1's grid.x=cdiv(N,256) exceeds the 65535 hard limit on large shapes and directly
    # raises OutOfResources — always delegate when native is available.
    if _native_ok(x, index):
        _NATIVE_SCATTER_ADD_.call_boxed(_DEVICE_KEYSET, x, dim, index, src)
        return x

    if equal_count == x.dim() and index.shape == src.shape and dim == x.ndim - 1:
        return scatter_add_1(x, dim, index, src)
    if (index.shape == src.shape and index.shape == x.shape and dim != x.ndim - 1) or (
        x.shape[0] == 4096 and x.numel() >= 9437184 and dim != x.ndim - 1
    ):
        if index.shape != src.shape:
            src = clip_tensor_to_shape(src, index)
        return scatter_add_1(x, dim, index, src)
    else:
        return scatter_add_0(x, dim, index, src)


def scatter_add(inp, dim, index, src):
    """Non-in-place scatter_add: prefer the single-kernel one-hot write, else clone and go in place."""
    logger.debug("GEMS_ENFLAME SCATTER_ADD")

    src_view = src.as_strided(index.shape, src.stride())
    if is_last_dim_flat(inp, dim, index, src_view):
        width = index.shape[-1]
        rows = index.numel() // width
        out_width = inp.shape[-1]
        if (
            out_width <= _SCATTER_ADD_ONEHOT_MAX_N
            and rows * out_width * width <= _SCATTER_ADD_ONEHOT_MACS
        ):
            # Do clone+scatter in a single kernel: each out position is written exactly once, with no atomics
            out = torch.empty_like(inp)
            scatter_add_onehot(inp, index, src_view, out)
            return out

    out = inp.clone()
    scatter_add_(out, dim, index, src)
    return out
