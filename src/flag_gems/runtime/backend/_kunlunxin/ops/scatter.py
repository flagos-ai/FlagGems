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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import (
    MemOverlap,
    has_internal_overlapping,
    restride_dim,
)

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("import flag_gems")
    # code.writeline("from flag_gems.utils import triton_lang_extension as ext")
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
        # BOUNDED BLOCK: the kernel materializes BLOCK-element offset vectors
        # (inp_offsets / cur_idx / mod). The old unbounded
        # next_power_of_2(cdiv(cdiv(N, 12), 4)) is the UNBOUNDED-BLOCK
        # anti-pattern (same family as gather): for large N it produces
        # BLOCK up to 2^20, a giant tile expanded per element in
        # ConvertTritonXPUToLLVM, and grid = cdiv(N, BLOCK * LOOP) collapses
        # to a handful of programs -> severe under-parallelization.
        # Cap at 4096 to keep tiles small and the grid wide.
        code.writeline(
            "return min("
            'triton.next_power_of_2(triton.cdiv(triton.cdiv(args["N"], 12), 4)), 4096)'
        )  # LOOP = 4
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
            # reduce options
            code.writeline("IS_ADD: tl.constexpr,")
            code.writeline("IS_MUL: tl.constexpr,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")
            code.writeline("INT32_OFFSET: tl.constexpr,")
            code.writeline("DENSE_SRC_IDX: tl.constexpr")

    code.writeline("):")

    # Kernel Code
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")

        #   1. Calculate inp_offsets and idx_offsets
        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N")
            code.writeline("cur_idx = offsets")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("else:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
            # Fast path: when src/index are contiguous over index.shape, the
            # linear iteration offset IS the element offset. Using it directly
            # keeps the src/index gm2lm contiguous (block DMA) instead of the
            # modulo-based address math that forces discrete/scalar transfers.
            code.writeline("if DENSE_SRC_IDX:")
            with code.indent():
                code.writeline("idx_offsets = offsets")
                code.writeline("src_offsets = offsets")
            code.writeline("else:")
            with code.indent():
                code.writeline("if INT32_OFFSET:")
                with code.indent():
                    code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                    code.writeline("src_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("else:")
                with code.indent():
                    code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                    code.writeline("src_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
            for i in range(rank)[::-1]:
                code.writeline("if INT32_OFFSET:")
                with code.indent():
                    code.writeline(f"shape_{i} = shape_{i}.to(tl.int32)")
                    code.writeline(f"inp_stride_{i} = inp_stride_{i}.to(tl.int32)")
                    code.writeline(f"index_stride_{i} = index_stride_{i}.to(tl.int32)")
                    code.writeline(f"src_stride_{i} = src_stride_{i}.to(tl.int32)")
                code.writeline(f"mod = cur_idx % shape_{i}")
                code.writeline(f"inp_offsets += mod * inp_stride_{i}")
                code.writeline("if not DENSE_SRC_IDX:")
                with code.indent():
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
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("cur_index = cur_index.to(tl.int32)")
                code.writeline("stride_dim = stride_dim.to(tl.int32)")

            code.writeline("dim_offsets = cur_index * stride_dim")
            code.writeline("inp_offsets += dim_offsets")
            code.newline()
            code.writeline("if IS_ADD: ")
            with code.indent():
                code.writeline(
                    "tl.atomic_add(out + inp_offsets, cur_src, mask=mask, sem='relaxed')"
                )
            code.writeline("elif IS_MUL: ")
            with code.indent():
                code.writeline(
                    "tl.atomic_mul(out + inp_offsets, cur_src, mask=mask, sem='relaxed')"
                )

            code.writeline("else: ")
            with code.indent():
                code.writeline("tl.store(out + inp_offsets, cur_src, mask=mask)")

            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def parameter_for_wrapper() -> str:
    # src_strided, index, inp, out, dim, M, N, reduce
    parameters: List[str] = []

    parameters.append("src_strided")
    parameters.append("index")
    parameters.append("inp")
    parameters.append("out")
    parameters.append("dim_size")
    parameters.append("dim_stride")
    parameters.append("N")
    parameters.append("reduce: tl.constexpr=None")
    parameters.append("int32_offset: tl.constexpr=None")

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

        code.writeline('IS_ADD = reduce == "add"')
        code.writeline('IS_MUL = reduce == "multiply"')
        code.writeline("int32_offset = int32_offset or True")

        # src/index are dense (row-major contiguous over index.shape) iff their
        # strides equal the contiguous strides of index.shape. In that case the
        # linear offset can be used directly for the src/index loads.
        code.writeline("_cont = [1] * len(index_shapes)")
        code.writeline("_acc = 1")
        code.writeline("for _i in range(len(index_shapes) - 1, -1, -1):")
        with code.indent():
            code.writeline("_cont[_i] = _acc")
            code.writeline("_acc *= index_shapes[_i]")
        code.writeline(
            "dense_src_idx = list(src_strides) == _cont and list(index_strides) == _cont"
        )

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
                # reduce options
                code.writeline("IS_ADD,")
                code.writeline("IS_MUL,")
                code.writeline("INT32_OFFSET=int32_offset,")
                code.writeline("DENSE_SRC_IDX=dense_src_idx,")
                # code.writeline("buffer_size_limit=512,")
                # code.writeline("isCloseUnrollControl=True,")

        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [src_strided, index, inp, out, dim, M, N, reduce]
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
                "_scatter_wrapper",
                "_scatter_jit_function",
                code,
            )

            file_name = f"scatter_rank_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_func = ScatterFunction()


# ---- tle.raw LM-staging fast path for innermost-dim scatter_src ---------------
# The codegen kernel below lowers the data-dependent scatter store to a discrete
# (stride == -1) per-element GM store.  Staging the output row in local memory,
# doing the data-dependent write there, and writing the row back as one
# contiguous block DMA removes that discrete store.
#
# Two things this path deliberately avoids:
#  * device-side index arithmetic to build the per-row base offsets -- the payload
#    unravelling is done from scalar args.  (kunlunxin `arange` is unreliable for
#    grid >= 2 and can drop a whole block of its result; building bases with it is
#    what made four earlier attempts at this path look broken.  See
#    harness/solution/arange/root_cause.md.)
#  * dtype casts.  `index.to(torch.int32)` costs ~7.5 ms for 33M elements on this
#    backend (~53 GB/s, i.e. 10x slower than a plain copy) and dominated the whole
#    op, so the payload reads the index in its native width instead.
#
# Built lazily to keep backend import side-effect free.  Shapes/dtypes/layouts
# outside the fast path defer to the codegen kernel (dispatch, not fallback).
_TLE_NDM = 4  # max leading dims (ndim - 1); covers ndim <= 5
_TLE_GRID = 8  # physical clusters on this part
# (elt, index C type, WT, KT, simd).  LM budget is ~6-7 KB, so the wider int64
# index halves KT.
#
# The fp32 entries are SIMD: the data-dependent inner loop scatters 16 lanes at a
# time with `vscatter_lm_float32x16` instead of one scalar store per element.  The
# byte-offset vector comes straight out of the LM index tile, so no per-lane LM
# write is needed (that per-lane write is what made the fp16 route a net loss):
#   * int32 index -> `vload_lm_int32x16(li + j0)` (offset == element index)
#   * int64 index -> `vgather_lm_int32x16(li + j0, strideoff)` with strideoff =
#     [0,8,..,120] bytes, i.e. the low 32-bit word of each int64 == the `(int)li[j]`
#     truncation the scalar loop does.  No dtype cast / re-layout anywhere.
# 16-bit dtypes stay on the scalar payload (their vscatter needs an int16x32
# offset, which has no arithmetic/narrowing intrinsics).
_TLE_VARIANTS = (
    ("short", "int", 2048, 512, False),
    ("short", "long", 2048, 256, False),
    ("int", "int", 1024, 256, True),
    ("int", "long", 1024, 256, True),
)
_TLE = {"init": False, "ok": False, "kern": {}}


def _tle_scatter_init():
    if _TLE["init"]:
        return _TLE["ok"]
    _TLE["init"] = True
    try:
        import os as _os

        import triton as _triton
        import triton.experimental.tle as _tle

        # Precompiled, secrecy-hardened device objects (packed from
        # payload/src/scatter_lm_*.xpu via
        # docs/xpu3/how_to_pack_payload/pack_payload.py --obj-dir, then merged
        # into one object with `ld.lld -r`).  We ship the .o (not the .xpu
        # source): the stub name must equal the entry symbol and the signature
        # must match the C++ ABI.  In object= mode the merge-time signature
        # check is skipped (the payload isn't in IR), so a wrong signature fails
        # at runtime, not compile -- keep the stub in lockstep with the C++.
        # Re-pack after editing any .xpu: the compile-cache key is the .o
        # content digest, so a stale .o silently keeps the old code.
        _obj = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "payload",
            "obj",
            "scatter_src.o",
        )

        # `_wt`/`_kt`/`_simd` are baked into the packed payloads now; the tuple
        # stays as the record of the variant set and the tile sizes it used.
        for elt, idx_t, _wt, _kt, _simd in _TLE_VARIANTS:
            sym = f"scatter_lm_{elt}_{idx_t}"
            ns = {"_tle": _tle, "_OBJ": _obj}
            exec(
                "@_tle.raw.dialect('xpu3', object=_OBJ, arch=3)\n"
                f"def {sym}(out, base, index, src, R, W, K, *md):\n"
                "    ...\n",
                ns,
            )
            callee = ns[sym]

            def _mk(callee=callee):
                @_triton.jit
                def _k(
                    O,
                    B,
                    I,
                    S,
                    R,
                    W,
                    K,
                    L0,
                    L1,
                    L2,
                    L3,
                    O0,
                    O1,
                    O2,
                    O3,
                    I0,
                    I1,
                    I2,
                    I3,
                    S0,
                    S1,
                    S2,
                    S3,
                ):
                    _tle.raw.call(
                        callee,
                        (
                            O,
                            B,
                            I,
                            S,
                            R,
                            W,
                            K,
                            L0,
                            L1,
                            L2,
                            L3,
                            O0,
                            O1,
                            O2,
                            O3,
                            I0,
                            I1,
                            I2,
                            I3,
                            S0,
                            S1,
                            S2,
                            S3,
                        ),
                    )

                return _k

            _TLE["kern"][(elt, idx_t)] = _mk()
        _TLE["ok"] = True
    except Exception as _e:  # pragma: no cover - keep backend importable
        logger.debug("GEMS_KUNLUNXIN SCATTER tle.raw init failed: %r", _e)
        _TLE["ok"] = False
    return _TLE["ok"]


def _tle_scatter_src(inp, dim, index, src):
    """Innermost-dim scatter_src via tle.raw LM staging; None -> defer to codegen.

    Allocates its own output so the clone can be skipped.  The payload rewrites
    every visited output row in full (it loads the row's initial values from
    `base`, applies the scatter in LM, and writes the whole row back), and the
    row loop enumerates exactly index's leading dims.  So when the index covers
    every output row (`index.shape[:-1] == inp.shape[:-1]`) every element of the
    output is written and a bare `empty_like(inp)` (with `base = inp`) suffices
    -- no clone.  With partial row coverage the rows the kernel never visits must
    keep inp's values, so `out = inp.clone()` and `base = out`.

    `base` shares `out`'s leading strides in the payload (one `os[]` array serves
    both the read and the write row base), which is why the clone-skip branch
    requires `inp.is_contiguous()`.
    """
    try:
        if dim != inp.ndim - 1:
            return None
        if inp.ndim - 1 > _TLE_NDM or index.ndim != inp.ndim:
            return None
        if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return None
        if src.dtype != inp.dtype or index.dtype not in (torch.int32, torch.int64):
            return None
        if not src.is_contiguous():
            return None
        # Any dtype cast here would cost more than the kernel (see header note).
        if not index.is_contiguous():
            return None
        ndim = inp.ndim
        W = inp.size(-1)
        K = index.size(-1)
        if W <= 0 or K <= 0:
            return None
        R = index.numel() // K
        if R <= 0:
            return None
        if any(index.size(d) > inp.size(d) for d in range(ndim - 1)):
            return None
        if any(
            t.stride(d) * t.size(d) >= 2**32
            for t in (inp, index, src)
            for d in range(ndim)
        ):
            return None
        if not _tle_scatter_init():
            return None

        if inp.dtype == torch.float32:
            # fp32 takes the SIMD payload, whose C signature is `float*`.
            elt = "int"
        else:
            elt = "short"
        idx_t = "long" if index.dtype == torch.int64 else "int"
        kern = _TLE["kern"].get((elt, idx_t))
        if kern is None:
            return None

        # Output allocation (only reached once we are committed to this path).
        # `full_rows` == the row loop visits every output row.  When it does and
        # inp is contiguous, `empty_like(inp)` plus `base = inp` is complete and
        # the clone is skipped; otherwise every unwritten row must keep inp's
        # values, so clone.
        full_rows = all(index.size(d) == inp.size(d) for d in range(ndim - 1))
        if full_rows and inp.is_contiguous():
            out = torch.empty_like(inp)
            base = inp
        else:
            out = inp.clone()
            if has_internal_overlapping(out) == MemOverlap.Yes:
                out = out.contiguous()
            base = out
        if not out.is_contiguous():
            return None

        if elt == "int":
            O, B, S = out, base, src
        else:
            O, B, S = (
                out.view(torch.int16),
                base.view(torch.int16),
                src.view(torch.int16),
            )

        pad = lambda v: (list(v) + [0] * _TLE_NDM)[:_TLE_NDM]
        md = (
            pad([index.shape[d] for d in range(ndim - 1)])
            + pad([out.stride(d) for d in range(ndim - 1)])
            + pad([index.stride(d) for d in range(ndim - 1)])
            + pad([src.stride(d) for d in range(ndim - 1)])
        )

        with torch_device_fn.device(inp.device):
            kern[(_TLE_GRID,)](O, B, index, S, R, W, K, *md)
        return out
    except Exception as _e:  # pragma: no cover - never break correctness
        logger.debug("GEMS_KUNLUNXIN SCATTER tle path deferred: %r", _e)
        return None


def scatter(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_KUNLUNXIN SCATTER")
    if dim < 0:
        dim += inp.ndim

    if reduce is not None:
        assert inp.dtype not in (
            torch.bfloat16,
        ), "Unsupported operation: reduce scatter bfloat tensors."

    if reduce is None:
        # The TLE LM-staging path allocates its own output and, when the index
        # covers every output row, skips the clone entirely (see
        # _tle_scatter_src).  Returns None -> defer to the codegen kernel.
        res = _tle_scatter_src(inp, dim, index, src)
        if res is not None:
            return res

    out = inp.clone()

    if has_internal_overlapping(out) == MemOverlap.Yes:
        out = out.contiguous()

    src_strided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    int32_size_dim = lambda x: x.stride(dim) * x.size(dim) < 2**32
    use_int32_offset = all(map(int32_size_dim, (inp, index, src)))

    _scatter_func(
        src_strided,
        index,
        inp_restrided,
        out,
        dim_size,
        dim_stride,
        N,
        reduce,
        int32_offset=use_int32_offset,
    )

    return out


def scatter_(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_KUNLUNXIN SCATTER_")
    if dim < 0:
        dim += inp.ndim
    out = inp

    if reduce is not None:
        assert inp.dtype not in (
            torch.bfloat16,
        ), "Unsupported operation: reduce scatter bfloat tensors."

    assert (
        has_internal_overlapping(out) != MemOverlap.Yes
    ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."

    src_restrided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    int32_size_dim = lambda x: x.stride(dim) * x.size(dim) < 2**32
    use_int32_offset = all(map(int32_size_dim, (inp, index, src)))
    _scatter_func(
        src_restrided,
        index,
        inp_restrided,
        out,
        dim_size,
        dim_stride,
        N,
        reduce,
        int32_offset=use_int32_offset,
    )

    return inp
