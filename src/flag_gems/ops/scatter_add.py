import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils import libentry
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer
from flag_gems.utils.shape_utils import restride_dim

from ..utils import dim_compress

logger = logging.getLogger(__name__)


def _heur_block_2d(args):
    # Vendor-specific override for metax/iluvatar (larger blocks for occupancy)
    if flag_gems.vendor_name in ["metax", "iluvatar"]:
        return 256
    # H20 has 78 SMs; smaller BLOCK for small N gives more parallelism,
    # 128 for large N balances occupancy and register pressure.
    N = args["N"]
    if N <= 16384:
        return 32
    elif N <= 65536:
        return 64
    return 128


def _heur_loop_2d(args):
    # LOOP=1: scatter_add uses atomic_add; LOOP>1 serializes atomics within
    # a program and increases contention, hurting throughput.
    return 1


@libentry()
@triton.heuristics({"BLOCK": _heur_block_2d, "LOOP": _heur_loop_2d})
@triton.jit(do_not_specialize=["N", "idx_ncols", "src_stride0", "out_ncols"])
def scatter_add_2d_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    N,
    idx_ncols,
    src_stride0,
    out_ncols,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(0)
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)
    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N
        row = offsets // idx_ncols
        col = offsets % idx_ncols
        idx_offsets = row * idx_ncols + col
        src_offsets = row * src_stride0 + col
        idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
        if DIM == 0:
            out_offsets = idx * out_ncols + col
        else:
            out_offsets = row * out_ncols + idx
        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0)
        tl.atomic_add(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")


@libentry()
@triton.jit
def _copy_contiguous_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets, mask=mask), mask=mask)


def _copy_contiguous(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy (and optionally cast) ``src`` into ``dst`` with a minimal Triton kernel.

    Lean replacement for ``src.clone()`` / ``src.to(dst.dtype)`` used to
    initialize the non-inplace ``scatter_add`` output.  ``src`` and ``dst`` must
    share shape (and, for a same-dtype copy, strides / storage offset, i.e.
    ``dst`` is obtained via ``torch.empty_like(src)``).  When the dtypes differ
    the load/store performs an implicit cast, which is used to upcast fp16/bf16
    inputs to fp32 (and back) without going through FlagGems' general ``copy_``.

    Under ``use_gems``, ``clone()`` and ``to()`` route through FlagGems ``copy_``
    (built on ``pointwise_dynamic``) and carry high dispatch overhead for small
    tensors (~0.04 ms per call, ~8x slower than native).  This dedicated kernel
    avoids that overhead while keeping the copy inside FlagGems/Triton (no
    fallback to the native aten implementation).
    """
    n = src.numel()
    if n == 0:
        return
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, BLOCK_SIZE),)
    _copy_contiguous_kernel[grid](src, dst, n, BLOCK_SIZE=BLOCK_SIZE)


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
        code.writeline("if(flag_gems.vendor_name in ['metax', 'iluvatar']):")
        with code.indent():
            code.writeline("return 256")
        code.writeline("return 128")
    code.newline()
    code.newline()

    code.writeline("def loop_count(args):")
    with code.indent():
        code.writeline("return 1")
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


def scatter_add_0(inp, dim, index, src):
    logger.debug("GEMS SCATTER_ADD_0")
    N = index.numel()
    dtype_convert = False
    if (inp.dtype == torch.float16 or inp.dtype == torch.bfloat16) and N > 131072:
        out = inp.to(torch.float32)
        dtype_convert = True
    else:
        out = inp

    # 2D fast path: specialized kernel with simple row/col decomposition.
    # Only beneficial for small N where the simpler index arithmetic
    # outweighs the N-dim kernel's better memory access pattern.
    # Large N falls through to the N-dim generated kernel which is faster.
    if inp.ndim == 2 and N <= 131072:
        src_strided = src.as_strided(index.shape, src.stride())
        dim_2d = dim % 2
        idx_ncols = index.shape[1]
        src_stride0 = src_strided.stride(0)
        out_ncols = out.shape[1]
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"] * meta["LOOP"]),)
        scatter_add_2d_kernel[grid](
            src_strided,
            index,
            out,
            N,
            idx_ncols,
            src_stride0,
            out_ncols,
            dim_2d,
        )
        if dtype_convert:
            return inp.copy_(out.to(src.dtype))
        return out

    src_strided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)

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


def scatter_add_1(x, dim, index, src):
    logger.debug("GEMS SCATTER_ADD_1")
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
    if (x.dtype == torch.float16 or x.dtype == torch.bfloat16) and all_elem > 131072:
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
        if dtype_convert:
            return origin.copy_(x.to(src.dtype))
        return x


def scatter_add_(x, dim, index, src):
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
    logger.debug("GEMS SCATTER_ADD")
    # Non-inplace variant: produce out = inp (copied) then scatter-add src into it.
    #
    # The naive implementation ``out = inp.clone(); return scatter_add_(out, ...)``
    # is slow under ``use_gems``: ``clone()`` (and the fp16/bf16 upcast that
    # ``scatter_add_`` performs via ``inp.to(float32)`` for large N) route through
    # FlagGems ``copy_``, which is built on ``pointwise_dynamic`` and carries high
    # dispatch overhead for small tensors (~0.04 ms per call, ~8x slower than
    # native).
    #
    # Instead we initialize the output with ``torch.empty_like`` + a minimal
    # Triton copy kernel.  ``scatter_add_`` only upcasts fp16/bf16 to fp32 when
    # the scatter is large (index.numel() > 131072, see scatter_add_0/_1); for
    # that case we do the upcast up-front with the same lean kernel, run the
    # scatter in fp32 (so ``scatter_add_`` skips its own slow ``.to()``), then
    # cast back -- keeping accumulation precision without the slow round-trips.
    # Everything stays inside FlagGems/Triton (no fallback to the native aten
    # implementation).  Non-contiguous inputs are densified via ``clone()`` so
    # the output layout matches PyTorch's (contiguous).
    if not inp.is_contiguous():
        out = inp.clone()
        return scatter_add_(out, dim, index, src)

    if inp.dtype in (torch.float16, torch.bfloat16) and index.numel() > 131072:
        # Upcast inp -> fp32 with a lean kernel, scatter in fp32, cast back.
        out_f32 = torch.empty(inp.shape, dtype=torch.float32, device=inp.device)
        _copy_contiguous(inp, out_f32)
        res_f32 = scatter_add_(out_f32, dim, index, src)
        out = torch.empty_like(inp)
        _copy_contiguous(res_f32, out)
        return out

    out = torch.empty_like(inp)
    _copy_contiguous(inp, out)
    return scatter_add_(out, dim, index, src)
