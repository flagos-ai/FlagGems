import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils.code_cache import cache_dir
from flag_gems.utils.code_utils import IndentedBuffer
from flag_gems.utils.shape_utils import restride_dim

from .nonzero import _device_int_tensor

logger = logging.getLogger(__name__)

# Shared tle.raw on-chip scatter-add kernel (2D dim==last). gather_backward is
# math-identical to zeros.scatter_add_(dim, index, grad); reuse the same
# hand-written cluster payload here so the rowvec case skips the slow
# Triton lane-serial scatter.
try:
    from .scatter_add_ import (
        _HAS_SA2D_TLE,
        _SA2D_TLE_BIG_KERNELS,
        _SA2D_TLE_KERNELS,
        _SA2D_TLE_TILE_KERNELS,
        SA2D_TILE_MAX_K,
        SA2D_TLE_BIG_MAX_K,
        SA2D_TLE_GRID,
        SA2D_TLE_MAX_K,
        _sa2d_grid,
    )
except Exception:
    _HAS_SA2D_TLE = False
    _SA2D_TLE_KERNELS = {}
    _SA2D_TLE_BIG_KERNELS = {}
    _SA2D_TLE_TILE_KERNELS = {}


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("import builtins")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils import triton_lang_extension as ext")

    code.newline()
    code.newline()
    return code


def generate_gather_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.newline()
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("inp,")
        code.writeline("out,")
        code.writeline("index,")
        code.writeline("dim: tl.constexpr,")
        code.writeline("stride_dim: tl.constexpr,")
        code.writeline("stride_last: tl.constexpr,")
        if rank > 1:
            stride_args = ", ".join(
                f"index_shape_{i}: tl.constexpr" for i in range(rank - 1)
            )
            code.writeline(f"{stride_args}, # shape of the leading axes (index)")
            stride_args = ", ".join(
                f"inp_stride_{i}: tl.constexpr" for i in range(rank - 1)
            )
            code.writeline(f"{stride_args}, # stride for inp leading axes")
        code.writeline("M: tl.constexpr,")
        code.writeline("N: tl.constexpr,")
        code.writeline("BLOCK_M: tl.constexpr,")
        code.writeline("BLOCK_N: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("pid_m = ext.program_id(0)")
        code.writeline("pid_n = ext.program_id(1)")
        code.writeline("rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)")
        code.writeline("cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)")

        if rank > 1:
            code.writeline("cur = rows")
            code.writeline("base = tl.zeros((BLOCK_M,), dtype=tl.int32)")
            for i in range(rank - 2, -1, -1):
                code.writeline(f"acc = cur % index_shape_{i}")
                code.writeline(f"base += tl.where(dim == {i}, 0, acc) * inp_stride_{i}")
                code.writeline(f"cur //= index_shape_{i}")

        code.writeline("offsets = rows[:, None] * N + cols[None, :]")
        if rank == 1:
            code.writeline("base = tl.zeros((BLOCK_M,), dtype=tl.int32)")

        code.writeline("mask = (rows < M)[:, None] & (cols < N)[None, :]")
        code.writeline("cur_index = tl.load(index + offsets, mask=mask, other=0)")
        code.writeline(
            "inp_offsets = base[:, None] + cur_index.to(tl.int32) * stride_dim"
        )
        if rank > 1:
            code.writeline("inp_offsets += cols[None, :] * stride_last")
        code.writeline("cur_inp = tl.load(inp + inp_offsets, mask=mask, other=0)")
        code.writeline("tl.store(out + offsets, cur_inp, mask=mask)")

    code.newline()
    code.newline()
    return code


def parameter_for_wrapper() -> str:
    parameters: List[str] = []

    parameters.append("inp")
    parameters.append("out")
    parameters.append("index")
    parameters.append("dim")
    parameters.append("stride_dim")
    parameters.append("stride_last")
    parameters.append("M")
    parameters.append("N")

    return ", ".join(parameters)


def generate_gather_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("index_shapes = list(index.shape)")
        code.writeline("inp_strides = list(inp.stride())")

        code.writeline("BLOCK_N = builtins.min(triton.next_power_of_2(N), 4096)")
        code.writeline(
            "BLOCK_M = builtins.min(triton.next_power_of_2(triton.cdiv(M, 12)), 8)"
        )

        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(M, meta["BLOCK_M"]),')
            code.writeline('triton.cdiv(N, meta["BLOCK_N"])')
        code.writeline(")")

        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)

        with code.indent():
            code.writeline("inp, out, index, ")
            code.writeline("dim,")
            code.writeline("stride_dim,")
            code.writeline("stride_last,")
            if rank > 1:
                s = ", ".join(f"index_shapes[{i}]" for i in range(rank - 1))
                code.writeline(f"{s},")
                s = ", ".join(f"inp_strides[{i}]" for i in range(rank - 1))
                code.writeline(f"{s},")
            code.writeline("M,")
            code.writeline("N,")
            code.writeline("BLOCK_M=BLOCK_M,")
            code.writeline("BLOCK_N=BLOCK_N,")
            code.writeline("buffer_size_limit=2048,")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    shape = inputs[2].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_gather_kernel(rank, kernel_name, code)
    code = generate_gather_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class GatherFunction:
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
                "_gather_wrapper",
                "_gather_jit_function",
                code,
            )

            file_name = f"gather_rank_{key}_pid_{self.pid}.py"

            with open(cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_gather_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_gather_func = GatherFunction()


def gather(inp, dim, index, out=None, sparse_grad=False):
    logger.debug("GEMS_KUNLUNXIN GATHER")
    if dim < 0:
        dim += inp.ndim
    if inp.ndim != index.ndim:
        raise IndexError(
            f"Index tensor must have the same number of dimensions as input tensor. "
            f"Got {index.ndim} and {inp.ndim}."
        )

    inp = inp.contiguous()
    index = index.contiguous()
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)
    out = out.contiguous()
    if index.numel() == 0:
        return out

    stride_dim = inp.stride(dim)
    N = list(index.shape)[index.ndim - 1]
    M = index.numel() // N
    inp_dim_size = inp.size(dim)

    if inp.numel() < 2**31 and index.numel() < 2**31:
        stride_last = inp.stride(index.ndim - 1) if dim != index.ndim - 1 else 0
        _gather_func(inp, out, index, dim, stride_dim, stride_last, M, N)
    else:
        inp_strided = restride_dim(inp, dim, index.shape)
        _gather_func_legacy(
            inp_strided, out, index, dim, stride_dim, inp_dim_size, M, N
        )
    return out


def generate_gather_legacy_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.newline()

    code.writeline("def heur_block_m(args):")
    with code.indent():
        code.writeline(
            'return builtins.min(triton.next_power_of_2(triton.cdiv(args["M"], 12)), 8)'
        )

    code.newline()

    code.writeline("def heur_block_n(args):")
    with code.indent():
        code.writeline('return builtins.min(triton.next_power_of_2(args["N"]), 4096)')

    code.newline()
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("values={")
        with code.indent():
            code.writeline('"BLOCK_M": heur_block_m,')
            code.writeline('"BLOCK_N": heur_block_n,')
        code.writeline("},")
    code.writeline(")")
    code.writeline("@triton.jit")

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("inp,")
            code.writeline("out,")
            code.writeline("index,")

            stride_args = ", ".join(
                f"inp_stride_{i}: tl.constexpr" for i in range(rank)
            )
            code.writeline(f"{stride_args}, # stride for inp")

            stride_args = ", ".join(
                f"index_stride_{i}: tl.constexpr" for i in range(rank)
            )
            code.writeline(f"{stride_args}, # stride for index")

            shape_args = ", ".join(
                f"index_shape_{i}: tl.constexpr" for i in range(rank)
            )
            code.writeline(f"{shape_args}, # shape for index")

            code.writeline("dim: tl.constexpr,")
            code.writeline("stride_dim: tl.constexpr,")
            code.writeline("inp_dim_size: tl.constexpr,")
            code.writeline("M: tl.constexpr,")
            code.writeline("N: tl.constexpr,")
            code.writeline("BLOCK_M: tl.constexpr,")
            code.writeline("BLOCK_N: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("pid_x = ext.program_id(0)")
        code.writeline("pid_y = ext.program_id(1)")
        code.writeline(
            "rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]"
        )
        code.writeline(
            "cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]"
        )
        code.writeline("rows_mask = rows_offsets < M")
        code.writeline("cols_mask = cols_offsets < N")

        code.writeline("offsets = (rows_offsets * N + cols_offsets).to(tl.int64)")
        code.writeline("mask = rows_mask & cols_mask")

        code.writeline("inp_offsets = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int64)")
        code.writeline("idx_offsets = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int64)")
        code.writeline("cur_idx = rows_offsets * N + cols_offsets")

        for i in range(rank):
            code.writeline(f"mod = cur_idx % index_shape_{i}")
            code.writeline(
                f"inp_offsets += tl.where(dim == {i}, 0, mod) * inp_stride_{i}"
            )
            code.writeline(f"idx_offsets += mod * index_stride_{i}")
            if i != (rank - 1):
                code.writeline(f"cur_idx //= index_shape_{i}")

        code.writeline("cur_index = tl.load(index + idx_offsets, mask=mask, other=0)")
        code.writeline("inp_offsets += cur_index * stride_dim")
        code.writeline("cur_inp = tl.load(inp + inp_offsets, mask=mask, other=0)")
        code.writeline("tl.store(out + idx_offsets, cur_inp, mask=mask)")

    code.newline()
    code.newline()
    return code


def parameter_for_legacy_wrapper() -> str:
    parameters: List[str] = []

    parameters.append("inp_strided")
    parameters.append("out")
    parameters.append("index")
    parameters.append("dim")
    parameters.append("stride_dim")
    parameters.append("inp_dim_size")
    parameters.append("M")
    parameters.append("N")

    return ", ".join(parameters)


def generate_gather_legacy_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_legacy_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("inp_strides = inp_strided.stride()")
        code.writeline("index_strides = index.stride()")
        code.writeline("index_shapes = list(index.shape)")

        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(M, meta["BLOCK_M"]),')
            code.writeline('triton.cdiv(N, meta["BLOCK_N"])')
        code.writeline(")")

        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)

        with code.indent():
            code.writeline("inp_strided, out, index, ")
            if rank > 0:
                s = ", ".join(f"inp_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                code.writeline("dim,")
                code.writeline("stride_dim,")
                code.writeline("inp_dim_size,")
                code.writeline("M,")
                code.writeline("N,")
                code.writeline("buffer_size_limit=2048,")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_legacy_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    shape = inputs[2].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_gather_legacy_kernel(rank, kernel_name, code)
    code = generate_gather_legacy_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class GatherFunctionLegacy:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_legacy_code(
                args,
                "_gather_legacy_wrapper",
                "_gather_legacy_jit_function",
                code,
            )

            file_name = f"gather_legacy_rank_{key}_pid_{self.pid}.py"

            with open(cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_legacy_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_gather_legacy_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_gather_func_legacy = GatherFunctionLegacy()


@triton.jit
def _gather_backward_sum_kernel(
    grad,
    index,
    output,
    d0,
    d1,
    d2,
    i_t0,
    i_t1,
    i_t2,
    is0,
    is1,
    is2,
    S,
    total,
    dim: tl.constexpr,
    ndim: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
    BLOCK_INDEX: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(0)
    oo = pid * BLOCK_OUTPUT + tl.arange(0, BLOCK_OUTPUT)
    ov = oo < total
    cur = tl.minimum(oo, total - 1)
    n = tl.zeros((BLOCK_OUTPUT,), dtype=tl.int32)
    base = tl.zeros((BLOCK_OUTPUT,), dtype=tl.int32)
    # coord_ok masks off output elements whose NON-gather coordinates fall
    # outside index's extent along that axis. gather only writes grad into the
    # sub-block reachable by index; positions beyond index's non-gather extent
    # must stay 0. Without this, base=x*i_t reads OOB/wrong index & grad and
    # produces spurious nonzero output (partial-index bug).
    coord_ok = ov
    if ndim == 3:
        x2 = cur % d2
        cur = cur // d2
        x1 = cur % d1
        cur = cur // d1
        x0 = cur
        if dim == 0:
            n = x0
            base = x1 * i_t1 + x2 * i_t2
            sdim = i_t0
            coord_ok = coord_ok & (x1 < is1) & (x2 < is2)
        elif dim == 1:
            n = x1
            base = x0 * i_t0 + x2 * i_t2
            sdim = i_t1
            coord_ok = coord_ok & (x0 < is0) & (x2 < is2)
        else:
            n = x2
            base = x0 * i_t0 + x1 * i_t1
            sdim = i_t2
            coord_ok = coord_ok & (x0 < is0) & (x1 < is1)
    else:
        x1 = cur % d1
        cur = cur // d1
        x0 = cur
        if dim == 0:
            n = x0
            base = x1 * i_t1
            sdim = i_t0
            coord_ok = coord_ok & (x1 < is1)
        else:
            n = x1
            base = x0 * i_t0
            sdim = i_t1
            coord_ok = coord_ok & (x0 < is0)
    acc = tl.zeros((BLOCK_OUTPUT,), dtype=tl.float32)
    for jt in range(0, LOOP):
        j = jt * BLOCK_INDEX + tl.arange(0, BLOCK_INDEX)
        jm = j < S
        jc = tl.minimum(j, S - 1)
        m = coord_ok[:, None] & jm[None, :]
        off = base[:, None] + jc[None, :] * sdim
        gi = tl.load(index + off, mask=m, other=0)
        gv = tl.load(grad + off, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(m & (gi == n[:, None]), gv, 0.0), axis=1)
    tl.store(output + oo, tl.where(coord_ok, acc, 0.0), mask=ov)


@triton.jit(
    do_not_specialize=[
        "N",
        "SLICE",
        "index_dim_size",
        "stride_dim",
        "i_s0",
        "i_s1",
        "i_s2",
        "o_s0",
        "o_s1",
        "o_s2",
    ]
)
def _gather_backward_scatter_kernel(
    index,
    grad,
    output,
    N,
    SLICE,
    index_dim_size,
    stride_dim,
    i_s0,
    i_s1,
    i_s2,
    o_s0,
    o_s1,
    o_s2,
    dim: tl.constexpr,
    rank: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    base = tl.program_id(0).to(tl.int32) * SLICE
    ar = tl.arange(0, BLOCK)
    for i in tl.static_range(LOOP):
        iter_off = i * BLOCK + ar
        mask = iter_off < SLICE
        offs = base + iter_off
        v = tl.load(index + offs, mask=mask, other=0).to(tl.int32)
        val = tl.load(grad + offs, mask=mask, other=0.0).to(tl.float32)
        cur = offs
        o = tl.zeros((BLOCK,), dtype=tl.int32)
        if rank == 3:
            mod = cur % i_s2
            if dim != 2:
                o += mod * o_s2
            cur = cur // i_s2
            mod = cur % i_s1
            if dim != 1:
                o += mod * o_s1
            cur = cur // i_s1
            mod = cur % i_s0
            if dim != 0:
                o += mod * o_s0
        else:
            mod = cur % i_s1
            if dim != 1:
                o += mod * o_s1
            cur = cur // i_s1
            mod = cur % i_s0
            if dim != 0:
                o += mod * o_s0
        o += v * stride_dim
        tl.atomic_add(output + o, val, mask=mask, sem="relaxed")


def _gather_backward_scatter(grad, self, dim, index_contiguous, result):
    ndim = self.ndim
    index_shape = list(index_contiguous.shape)
    N = index_contiguous.numel()

    SLICE = 1
    for s in index_shape[dim:]:
        SLICE *= s
    BLOCK = min(1024, max(1, triton.next_power_of_2(SLICE)))
    while (SLICE + BLOCK - 1) // BLOCK > 32 and BLOCK < 32768:
        BLOCK *= 2
    LOOP = (SLICE + BLOCK - 1) // BLOCK

    o_s = [1] * 3
    for k in range(ndim - 1, -1, -1):
        o_s[k] = 1 if k == ndim - 1 else o_s[k + 1] * self.shape[k + 1]
    pad = [1] * (3 - ndim)

    if grad.dtype in (torch.float16, torch.bfloat16):
        acc = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        _gather_backward_scatter_kernel[(N // SLICE,)](
            index_contiguous,
            grad.contiguous(),
            acc,
            N,
            SLICE,
            index_shape[-1],
            o_s[dim],
            *(index_shape + pad),
            *(o_s),
            dim=dim,
            rank=ndim,
            BLOCK=BLOCK,
            LOOP=LOOP,
        )
        return result.copy_(acc)
    _gather_backward_scatter_kernel[(N // SLICE,)](
        index_contiguous,
        grad.contiguous(),
        result,
        N,
        SLICE,
        index_shape[-1],
        o_s[dim],
        *(index_shape + pad),
        *(o_s),
        dim=dim,
        rank=ndim,
        BLOCK=BLOCK,
        LOOP=LOOP,
    )
    return result


def _try_gather_backward_tle(grad, self, dim, index_contiguous, result, zinit=False):
    # gather_backward == zeros_like(self).scatter_add_(dim, index, grad). When
    # dim is the last axis, the leading dims of index match self, and the output
    # row K fits LM, route to the shared tle.raw on-chip scatter kernel (no GM
    # atomics, whole out row resident in LM). Beats the Triton lane-serial rowvec
    # path (~25x slower than aten on dense shapes). Returns result or None.
    # When zinit=True the payload zero-inits each accumulator row ON-CHIP, so the
    # caller may pass an UNINITIALIZED `result` (grad.new_empty) and skip a
    # separate new_zeros memset. The rowvec invariant (index leading dims ==
    # self leading dims) guarantees every out row is written, so no uninitialized
    # leakage.
    if not _HAS_SA2D_TLE:
        return None
    if dim != self.ndim - 1 or self.ndim < 1:
        return None
    if grad.is_complex():
        return None
    if not result.is_contiguous() or not index_contiguous.is_contiguous():
        return None
    if list(index_contiguous.shape[:-1]) != list(self.shape[:-1]):
        return None
    K = self.shape[-1]
    # small K keeps the out row in per-core LM; big K (<=65536) keeps it in the
    # 256KB cluster SM with 64-core cooperation. Both cover fp32/fp16/bf16.
    if K == 0 or K > SA2D_TLE_BIG_MAX_K:
        return None
    S = index_contiguous.shape[-1]
    R = index_contiguous.numel() // S if S else 0
    if R == 0:
        return None
    if K <= SA2D_TLE_MAX_K:
        kernel = _SA2D_TLE_KERNELS.get(result.dtype)
    elif K <= SA2D_TILE_MAX_K:
        kernel = _SA2D_TLE_TILE_KERNELS.get(result.dtype)
    else:
        kernel = _SA2D_TLE_BIG_KERNELS.get(result.dtype)
    if kernel is None:
        return None

    # The tle payload addresses every tensor as a flat [rows, cols] buffer via the
    # explicit R/K/S/src_rs scalars (out+r*K, index+r*S, src+r*src_rs) -- it never
    # reads a tensor .shape. So a tensor that is ALREADY the target 2D shape needs
    # NO .view() (each .view is a dispatched op that, for the common 2D benchmark
    # case, is a pure-overhead identity ~0.0012ms x3). Only reshape when needed.
    def _as2d(t, rows, cols):
        if t.ndim == 2 and t.shape[0] == rows and t.shape[1] == cols:
            return t
        return t.view(rows, cols)

    out = _as2d(result, R, K)
    idx_i64 = (
        index_contiguous
        if index_contiguous.dtype == torch.int64
        else index_contiguous.to(torch.int64)
    )
    idx_i64 = _as2d(idx_i64, R, S)
    # Zero-copy fast path: a grad already contiguous in the op's own dtype viewed
    # as [R,S] has row stride S and needs no materializing copy (which under
    # use_gems would hit the slow gems copy kernel). The tle payload widens the
    # native dtype to an f32 LM accumulator on-chip, so NO host .to(f32) is done.
    # Otherwise do a single dtype/contiguous fixup.
    if grad.dtype == result.dtype and grad.is_contiguous():
        grad_c = _as2d(grad, R, S)
        src_rs = grad_c.stride(0)
    else:
        grad_c = grad.contiguous()
        if grad_c.dtype != result.dtype:
            grad_c = grad_c.to(result.dtype)
        grad_c = _as2d(grad_c, R, S)
        src_rs = grad_c.stride(0)
    # out is the accumulator. When zinit the payload zero-inits each row on-chip
    # (INP unused but must be a valid ptr, so pass out); else it reads+accumulates
    # onto out (INP == OUT).
    kernel[(_sa2d_grid(K, R),)](
        out, out, idx_i64, grad_c, R, K, S, src_rs, 1 if zinit else 0
    )
    return result


def gather_backward(grad, self, dim, index, sparse_grad):
    logger.debug("GEMS_KUNLUNXIN GATHER_BACKWARD")
    if sparse_grad:
        raise RuntimeError("gather_backward with sparse_grad=True is not supported")

    if self.numel() == 0:
        return grad.new_zeros(self.shape)

    # ---- LEAN tle hot path -------------------------------------------------
    # For the dominant case (gather dim == last axis, index+grad contiguous,
    # int64 index, grad dtype == out dtype, K within the on-chip band) launch
    # the shared on-chip scatter kernel with the MINIMUM python before the
    # launch. The general path below re-derives the same facts through
    # index.contiguous() + _try_gather_backward_tle (two calls, list() shape
    # compares, _as2d indirection) which adds ~5us of pure python -- on a
    # ~25us tiny-shape op that alone costs ~0.15x of speedup. This block
    # replicates _try's guards exactly; any input that does not match falls
    # through untouched (zero regression risk for the general path).
    if (
        _HAS_SA2D_TLE
        and self.ndim >= 1
        and not grad.is_complex()
        and index.dtype == torch.int64
        and index.is_contiguous()
        and grad.is_contiguous()
        and grad.dtype == self.dtype
        and (dim % self.ndim) == self.ndim - 1
    ):
        _ss = self.shape
        _K = _ss[-1]
        if (
            0 < _K <= SA2D_TLE_BIG_MAX_K
            and index.shape[:-1] == _ss[:-1]
            and index.numel() != 0
            and self.numel() < 2**31
            and index.numel() < 2**31
        ):
            if _K <= SA2D_TLE_MAX_K:
                _kern = _SA2D_TLE_KERNELS.get(self.dtype)
            elif _K <= SA2D_TILE_MAX_K:
                _kern = _SA2D_TLE_TILE_KERNELS.get(self.dtype)
            else:
                _kern = _SA2D_TLE_BIG_KERNELS.get(self.dtype)
            if _kern is not None:
                _S = index.shape[-1]
                _R = index.numel() // _S
                _zinit = _K <= SA2D_TILE_MAX_K
                result = grad.new_empty(_ss) if _zinit else grad.new_zeros(_ss)
                _out2 = result if result.ndim == 2 else result.view(_R, _K)
                _idx2 = index if index.ndim == 2 else index.view(_R, _S)
                _grad2 = grad if grad.ndim == 2 else grad.view(_R, _S)
                _kern[(_sa2d_grid(_K, _R),)](
                    _out2,
                    _out2,
                    _idx2,
                    _grad2,
                    _R,
                    _K,
                    _S,
                    _grad2.stride(0),
                    1 if _zinit else 0,
                )
                return result
    # ---- end lean hot path -------------------------------------------------

    dim = dim % self.ndim
    index_contiguous = index.contiguous()
    if index_contiguous.numel() == 0:
        return grad.new_zeros(self.shape)

    if self.numel() >= 2**31 or index_contiguous.numel() >= 2**31:
        result = grad.new_zeros(self.shape)
        return _gather_backward_legacy(grad, self, dim, index_contiguous, result)

    # tle on-chip fast-path. zinit lets the payload zero-init each accumulator
    # row ON-CHIP so the caller passes an UNINITIALIZED out (new_empty) and skips
    # the new_zeros memset (~26% of tiny-shape latency). The small-K LM band and
    # the moderate-big-K TILE band both zero-init in LM cheaply; only the big-K SM
    # band keeps zinit=0 (a scalar SM memset is slower than the GM->SM load of a
    # zeroed row, and new_zeros is negligible vs the multi-ms SM-atomic kernel).
    K_last = self.shape[-1]
    use_zinit = (
        _HAS_SA2D_TLE
        and dim == self.ndim - 1
        and K_last != 0
        and K_last <= SA2D_TILE_MAX_K
    )
    if use_zinit:
        result = grad.new_empty(self.shape)
        tle_out = _try_gather_backward_tle(
            grad, self, dim, index_contiguous, result, zinit=True
        )
        if tle_out is not None:
            return tle_out

    result = grad.new_zeros(self.shape)
    tle_out = _try_gather_backward_tle(
        grad, self, dim, index_contiguous, result, zinit=False
    )
    if tle_out is not None:
        return tle_out

    # Fast atomic-free row-vectorized path (dim == last, contiguous, matching
    # non-gather dims). Replaces the globally-serialized tl.atomic_add scatter
    # (which runs ~90-116ns/atomic -> seconds for large shapes) and the
    # O(output x S) sum scan. See _gather_backward_rowvec for the invariant.
    if _rowvec_ok(grad, self, dim, index_contiguous):
        return _gather_backward_rowvec(grad, self, dim, index_contiguous, result)

    if index_contiguous.shape[dim] <= 512 and self.ndim <= 3:
        return _gather_backward_sum(grad, self, dim, index_contiguous, result)

    # Non-last-dim case: permute the gather dim to the last axis and reuse the
    # atomic-free rowvec kernel. Ownership is by the non-gather coordinate (each
    # lane owns one output slice and walks the gather dim serially), so the RMW
    # is conflict-free with NO atomics -- avoids the catastrophically slow global
    # tl.atomic_add scatter (~90-116ns/atomic -> seconds). Requires index's
    # non-gather dims to match self (rowvec invariant); else fall back to atomic.
    if _permute_rowvec_ok(grad, self, dim, index_contiguous):
        return _gather_backward_permute_rowvec(
            grad, self, dim, index_contiguous, result
        )

    return _gather_backward_scatter(grad, self, dim, index_contiguous, result)


@triton.jit
def _gather_backward_rowvec_kernel(
    index,
    grad,
    acc,
    R,
    K,
    S,
    BR: tl.constexpr,
    BJ: tl.constexpr,
    LOOP: tl.constexpr,
):
    # Each LANE owns a distinct output row r in [0, R); the gather dim (length
    # S) is walked serially. In every step the BR scatter addresses
    # (r*K + gi) are distinct across lanes (rows differ) and the serial j-loop
    # makes the read-modify-write accumulate correctly within a row -> CORRECT
    # with NO atomics (no XPU global-atomic serialization). `acc` is padded to
    # grid*BR rows so masked tail lanes discard into their own distinct rows.
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rm = rows < R
    ir = tl.minimum(rows, R - 1)  # clamp ONLY index/grad reads to avoid OOB
    obase = rows * K  # distinct per lane (acc padded to cover pad rows)
    ibase = ir * S
    for jt in range(0, LOOP):
        for jj in tl.static_range(BJ):
            j = jt * BJ + jj
            jm = rm & (j < S)
            off = ibase + j
            gi = tl.load(index + off, mask=jm, other=0).to(tl.int32)
            gv = tl.load(grad + off, mask=jm, other=0.0).to(tl.float32)
            addr = obase + gi
            cur = tl.load(acc + addr, mask=jm, other=0.0)
            tl.store(acc + addr, cur + gv, mask=jm)


def _rowvec_ok(grad, self, dim, index_contiguous):
    return (
        dim == self.ndim - 1
        and self.ndim >= 1
        and not grad.is_complex()
        and grad.is_contiguous()
        and index_contiguous.is_contiguous()
        and list(index_contiguous.shape[:-1]) == list(self.shape[:-1])
    )


def _pick_rowvec_br(R, S):
    # Each block owns BR output rows (SIMD lanes) and walks the gather dim
    # (length S) serially. Wall time is minimized by giving the device enough
    # blocks to fill its compute clusters (grid ~ 8 measured best on P800 card 6,
    # 2026-09-28) while keeping BR wide enough for SIMD efficiency, so target
    # BR ~ R/8 (rounded up to a power of two). For very large S prefer more
    # blocks (smaller BR cap) to hide the long serial-loop latency.
    # BR MUST stay >= 64: at BR <= 32 the XPU backend miscompiles the
    # data-dependent scatter store and produces wrong results (verified
    # 2026-09-28). Capped at next_pow2(R).
    target = triton.next_power_of_2(max(1, (R + 7) // 8))
    br = min(128, target) if S >= 65536 else min(512, target)
    return max(64, min(br, triton.next_power_of_2(R)))


def _gather_backward_rowvec(grad, self, dim, index_contiguous, result):
    K = self.shape[-1]
    S = index_contiguous.shape[-1]
    R = index_contiguous.numel() // S if S else 0
    if R == 0 or K == 0:
        return result

    BR = _pick_rowvec_br(R, S)
    BJ = 1
    LOOP = (S + BJ - 1) // BJ
    grid_n = (R + BR - 1) // BR
    padded = grid_n * BR

    grad_c = grad.contiguous()
    # fp32 accumulator, padded so masked tail lanes discard into own rows.
    if result.dtype == torch.float32 and padded == R:
        acc = result.view(R, K)
    else:
        acc = torch.zeros((padded, K), dtype=torch.float32, device=self.device)

    _gather_backward_rowvec_kernel[(grid_n,)](
        index_contiguous,
        grad_c,
        acc,
        R,
        K,
        S,
        BR=BR,
        BJ=BJ,
        LOOP=LOOP,
        num_warps=8,
        buffer_size_limit=8192,
    )
    if acc.data_ptr() != result.data_ptr():
        result.copy_(acc[:R].view(self.shape))
    return result


def _permute_rowvec_ok(grad, self, dim, index_contiguous):
    # Rowvec (via permute) needs index's non-gather dims to match self exactly,
    # so that after moving `dim` to the last axis the leading dims collapse to a
    # clean [R, S] with R == prod(self non-gather dims).
    if grad.is_complex():
        return False
    for d in range(self.ndim):
        if d != dim and index_contiguous.shape[d] != self.shape[d]:
            return False
    return True


def _gather_backward_permute_rowvec(grad, self, dim, index_contiguous, result):
    ndim = self.ndim
    order = [d for d in range(ndim) if d != dim] + [dim]
    K = self.shape[dim]
    S = index_contiguous.shape[dim]
    R = 1
    for d in order[:-1]:
        R *= self.shape[d]
    if R == 0 or K == 0:
        return result

    grad_2d = grad.permute(order).contiguous().reshape(R, S)
    index_2d = index_contiguous.permute(order).contiguous().reshape(R, S)

    BR = _pick_rowvec_br(R, S)
    BJ = 1
    LOOP = (S + BJ - 1) // BJ
    grid_n = (R + BR - 1) // BR
    padded = grid_n * BR
    acc = torch.zeros((padded, K), dtype=torch.float32, device=self.device)

    _gather_backward_rowvec_kernel[(grid_n,)](
        index_2d,
        grad_2d,
        acc,
        R,
        K,
        S,
        BR=BR,
        BJ=BJ,
        LOOP=LOOP,
        num_warps=8,
        buffer_size_limit=8192,
    )

    perm_shape = [self.shape[d] for d in order]  # [*non_dim, K]
    inv = [0] * ndim
    for i, d in enumerate(order):
        inv[d] = i
    out = acc[:R].reshape(perm_shape).permute(inv).contiguous()
    return result.copy_(out.to(result.dtype))


def _gather_backward_sum(grad, self, dim, index_contiguous, result):
    ndim = self.ndim
    index_shape = list(index_contiguous.shape)
    index_strides = list(index_contiguous.stride())
    S = index_shape[dim]
    total = result.numel()
    pad = [1] * (3 - ndim)

    out_shapes = list(self.shape) + pad
    idx_strides = index_strides + pad
    idx_shapes = index_shape + pad

    BO, nw = 64, 4
    BI = min(512, max(32, triton.next_power_of_2(S)))
    LOOP = (S + BI - 1) // BI

    index32 = index_contiguous.to(torch.int32)

    _gather_backward_sum_kernel[(triton.cdiv(total, BO),)](
        grad.contiguous(),
        index32,
        result,
        out_shapes[0],
        out_shapes[1],
        out_shapes[2],
        idx_strides[0],
        idx_strides[1],
        idx_strides[2],
        idx_shapes[0],
        idx_shapes[1],
        idx_shapes[2],
        S,
        total,
        dim=dim,
        ndim=ndim,
        BLOCK_OUTPUT=BO,
        BLOCK_INDEX=BI,
        LOOP=LOOP,
        num_warps=nw,
        buffer_size_limit=2048,
    )
    return result


@triton.jit
def _gather_backward_kernel(
    grad,
    index,
    output,
    self_shape,
    index_shape,
    index_strides,
    total,
    index_dim_size,
    dim: tl.constexpr,
    ndim: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
    BLOCK_INDEX: tl.constexpr,
):
    output_offsets_flat = tl.program_id(0) * BLOCK_OUTPUT + tl.arange(0, BLOCK_OUTPUT)
    output_valid_flat = output_offsets_flat < total

    remaining = output_offsets_flat
    index_base = tl.zeros((BLOCK_OUTPUT,), dtype=tl.int64)
    output_dim_index = tl.zeros((BLOCK_OUTPUT,), dtype=tl.int64)
    coordinate_valid = output_valid_flat
    for axis in tl.static_range(ndim - 1, -1, -1):
        axis_size = tl.load(self_shape + axis)
        coordinate = remaining % axis_size
        remaining //= axis_size
        if axis == dim:
            output_dim_index = coordinate
        else:
            index_axis_size = tl.load(index_shape + axis)
            index_axis_stride = tl.load(index_strides + axis)
            coordinate_valid &= coordinate < index_axis_size
            index_base += coordinate * index_axis_stride

    index_dim_offsets = tl.arange(0, BLOCK_INDEX)[None, :]
    index_dim_stride = tl.load(index_strides + dim)
    valid = coordinate_valid[:, None] & (index_dim_offsets < index_dim_size)
    gather_offsets = index_base[:, None] + index_dim_offsets * index_dim_stride
    gathered_indices = tl.load(index + gather_offsets, mask=valid, other=-1)
    grad_values = tl.load(grad + gather_offsets, mask=valid, other=0.0)
    grad_values = tl.where(
        valid & (gathered_indices == output_dim_index[:, None]),
        grad_values.to(tl.float32),
        0.0,
    )
    result = tl.sum(grad_values, axis=1)
    result = tl.where(coordinate_valid, result, 0.0)
    tl.store(output + output_offsets_flat, result, mask=output_valid_flat)


def _gather_backward_legacy(grad, self, dim, index_contiguous, result):
    self_shape = _device_int_tensor(self.shape, torch.int64, self.device)
    index_shape = _device_int_tensor(index_contiguous.shape, torch.int64, self.device)
    index_strides = _device_int_tensor(
        index_contiguous.stride(), torch.int64, self.device
    )
    index_dim_size = index_contiguous.shape[dim]
    block_index = triton.next_power_of_2(index_dim_size)
    block_output = min(64, max(1, 2048 // block_index))
    _gather_backward_kernel[(triton.cdiv(result.numel(), block_output),)](
        grad.contiguous(),
        index_contiguous,
        result,
        self_shape,
        index_shape,
        index_strides,
        result.numel(),
        index_dim_size,
        dim=dim,
        ndim=self.ndim,
        BLOCK_OUTPUT=block_output,
        BLOCK_INDEX=block_index,
        num_warps=1,
        buffer_size_limit=2048,
        isCloseVectorization=True,
    )
    return result
