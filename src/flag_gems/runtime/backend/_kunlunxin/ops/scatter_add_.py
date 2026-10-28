import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils import dim_compress, libentry
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer
from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(__name__)

# --- tle.raw on-chip scatter-add (2D dim==1) --------------------------------
# A hand-written XPU cluster payload spliced into a Triton kernel. Row r is
# owned by a single thread that keeps the whole out row resident in LM and
# accumulates all contributions serially, so it needs NO GM atomics (unlike the
# atomic-based kernels below). Requires the out row width K <= SA2D_TLE_MAX_K.
# Beats aten scatter_add_ ~1.4-2.3x. Falls back to the atomic path when the
# preconditions do not hold.
SA2D_TLE_MAX_K = 1024
# P800 (KL3) has 12 physical clusters. Earlier cards (dev6/dev7) reported 8 via a
# CUDA-compat shim, and grid=8 left 4 of 12 clusters idle (device occupancy 8/12
# == 0.667, matching the profiled ratio 0.696). grid=12 uses all clusters and is
# the hard upper bound: grid>12 makes cluster_id() wrap (aliasing) so two programs
# own the same row -> race -> wrong results. Lifts big-K scatter/gather_backward
# (4096^2 0.52->0.70, 65536 0.73->1.08) with err=0.
SA2D_TLE_GRID = 12
# Moderate-big-K variant (1024 < K <= 8192): single-owner-per-row, output row
# TILED in LM, NON-atomic (mirrors XDNN run_core_batch). Beats the SM-atomic big
# path for K<=8192 (K=2048 0.71x, K=4096 0.545x vs the atomic ~0.43x); for K>8192
# the K/1024 rescan multiplier explodes so the SM-atomic big path is kept.
SA2D_TILE_MAX_K = 8192
# Big-K variant: out row exceeds per-core LM (K>1024) but fits the 256KB
# per-cluster shared memory (K<=65536 f32). One row per cluster, 64 cores
# cooperate with SM atomic-add. Beats the catastrophic GM-atomic fallback.
SA2D_TLE_BIG_MAX_K = 65536


def _sa2d_grid(K, R):
    # Pick the launch grid (# of clusters). The P800 (KL3) has 12 physical
    # clusters; the big-K/tile bands (K>1024) process 4096/1024 rows so using all
    # 12 clusters is a decisive win (4096^2 0.52->0.83, 65536 0.73->1.13). The
    # small-K LM band keeps the long-validated grid=8: its shapes are launch-bound,
    # and going to 12 (more launch overhead) or 1 (too few clusters for the
    # out-of-place read+write path) both REGRESS the tiny 64^2/256^2 cases. grid is
    # hard-capped at 12 -- grid>12 makes cluster_id() wrap (aliasing) so two
    # programs own the same row -> race -> wrong results.
    return SA2D_TLE_GRID if K > SA2D_TLE_MAX_K else 8


try:
    import os as _os

    import triton.experimental.tle as tle

    # Precompiled, secrecy-hardened device object (packed from scatter_add_2d.xpu
    # via docs/xpu3/how_to_pack_payload/pack_payload.py --obj-dir). We ship the .o
    # (not the .xpu source): each stub name must equal its entry symbol and the
    # signature must match the C++ ABI. In object= mode the merge-time signature
    # check is skipped (payload isn't in IR), so a wrong signature fails at runtime,
    # not compile -- keep every stub in lockstep with the C++. Re-pack after editing
    # scatter_add_2d.xpu: the compile-cache key is the .o content digest, so a stale
    # .o silently keeps the old code.
    _SA2D_OBJ = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "payload",
        "obj",
        "scatter_add.o",
    )

    @tle.raw.dialect("xpu3", object=_SA2D_OBJ, arch=3)
    def _scatter_add_2d_payload(out, inp, index, src, R, K, S, src_rs, zinit): ...

    @triton.jit(do_not_specialize=["ZINIT"])
    def _scatter_add_2d_tle_kernel(OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT):
        tle.raw.call(
            _scatter_add_2d_payload, (OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT)
        )

    @tle.raw.dialect("xpu3", object=_SA2D_OBJ, arch=3)
    def _scatter_add_2d_payload_fp16(out, inp, index, src, R, K, S, src_rs, zinit): ...

    @triton.jit(do_not_specialize=["ZINIT"])
    def _scatter_add_2d_tle_kernel_fp16(OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT):
        tle.raw.call(
            _scatter_add_2d_payload_fp16, (OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT)
        )

    @tle.raw.dialect("xpu3", object=_SA2D_OBJ, arch=3)
    def _scatter_add_2d_payload_bf16(out, inp, index, src, R, K, S, src_rs, zinit): ...

    @triton.jit(do_not_specialize=["ZINIT"])
    def _scatter_add_2d_tle_kernel_bf16(OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT):
        tle.raw.call(
            _scatter_add_2d_payload_bf16, (OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT)
        )

    # dtype -> on-chip kernel. fp16/bf16 accumulate in an f32 LM row (widen on
    # load, narrow on store) so no host-side .to(f32) copy (slow under use_gems).
    _SA2D_TLE_KERNELS = {
        torch.float32: _scatter_add_2d_tle_kernel,
        torch.float16: _scatter_add_2d_tle_kernel_fp16,
        torch.bfloat16: _scatter_add_2d_tle_kernel_bf16,
    }

    # Big-K (1024 < K <= 65536): cluster-cooperative, out row resident in SM.
    @tle.raw.dialect("xpu3", object=_SA2D_OBJ, arch=3)
    def _scatter_add_2d_big_payload(out, inp, index, src, R, K, S, src_rs, zinit): ...

    @triton.jit(do_not_specialize=["ZINIT"])
    def _scatter_add_2d_big_tle_kernel(OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT):
        tle.raw.call(
            _scatter_add_2d_big_payload, (OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT)
        )

    @tle.raw.dialect("xpu3", object=_SA2D_OBJ, arch=3)
    def _scatter_add_2d_big_payload_fp16(
        out, inp, index, src, R, K, S, src_rs, zinit
    ): ...

    @triton.jit(do_not_specialize=["ZINIT"])
    def _scatter_add_2d_big_tle_kernel_fp16(OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT):
        tle.raw.call(
            _scatter_add_2d_big_payload_fp16,
            (OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT),
        )

    @tle.raw.dialect("xpu3", object=_SA2D_OBJ, arch=3)
    def _scatter_add_2d_big_payload_bf16(
        out, inp, index, src, R, K, S, src_rs, zinit
    ): ...

    @triton.jit(do_not_specialize=["ZINIT"])
    def _scatter_add_2d_big_tle_kernel_bf16(OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT):
        tle.raw.call(
            _scatter_add_2d_big_payload_bf16,
            (OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT),
        )

    _SA2D_TLE_BIG_KERNELS = {
        torch.float32: _scatter_add_2d_big_tle_kernel,
        torch.float16: _scatter_add_2d_big_tle_kernel_fp16,
        torch.bfloat16: _scatter_add_2d_big_tle_kernel_bf16,
    }

    # Moderate-big-K (1024 < K <= 8192): single-owner tiled, NON-atomic.
    @tle.raw.dialect("xpu3", object=_SA2D_OBJ, arch=3)
    def _scatter_add_2d_tile_payload(out, inp, index, src, R, K, S, src_rs, zinit): ...

    @triton.jit(do_not_specialize=["ZINIT"])
    def _scatter_add_2d_tile_tle_kernel(OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT):
        tle.raw.call(
            _scatter_add_2d_tile_payload, (OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT)
        )

    @tle.raw.dialect("xpu3", object=_SA2D_OBJ, arch=3)
    def _scatter_add_2d_tile_payload_fp16(
        out, inp, index, src, R, K, S, src_rs, zinit
    ): ...

    @triton.jit(do_not_specialize=["ZINIT"])
    def _scatter_add_2d_tile_tle_kernel_fp16(
        OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT
    ):
        tle.raw.call(
            _scatter_add_2d_tile_payload_fp16,
            (OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT),
        )

    @tle.raw.dialect("xpu3", object=_SA2D_OBJ, arch=3)
    def _scatter_add_2d_tile_payload_bf16(
        out, inp, index, src, R, K, S, src_rs, zinit
    ): ...

    @triton.jit(do_not_specialize=["ZINIT"])
    def _scatter_add_2d_tile_tle_kernel_bf16(
        OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT
    ):
        tle.raw.call(
            _scatter_add_2d_tile_payload_bf16,
            (OUT, INP, IDX, SRC, R, K, S, SRC_RS, ZINIT),
        )

    _SA2D_TLE_TILE_KERNELS = {
        torch.float32: _scatter_add_2d_tile_tle_kernel,
        torch.float16: _scatter_add_2d_tile_tle_kernel_fp16,
        torch.bfloat16: _scatter_add_2d_tile_tle_kernel_bf16,
    }

    _HAS_SA2D_TLE = True
except Exception:  # tle unavailable / import failure -> keep atomic fallback
    _HAS_SA2D_TLE = False
    _SA2D_TLE_KERNELS = {}
    _SA2D_TLE_BIG_KERNELS = {}
    _SA2D_TLE_TILE_KERNELS = {}


# On the CI flagtree Triton build, tle.raw with object= may either be rejected at
# decoration (caught above -> _HAS_SA2D_TLE False) OR accepted yet silently run
# the wrong code / no-op, leaving the output buffer untouched and producing wrong
# results with no error. _HAS_SA2D_TLE only proves the dialect decorated, not that
# the payload actually computes. Probe once with a tiny known scatter on a
# sentinel-filled out; only trust the on-chip path when it reproduces the
# reference, else fall back to the atomic / Triton paths.
_SA2D_TLE_OK = None


def _sa2d_tle_usable(device):
    global _SA2D_TLE_OK
    if _SA2D_TLE_OK is None:
        if not _HAS_SA2D_TLE:
            _SA2D_TLE_OK = False
            return _SA2D_TLE_OK
        try:
            kernel = _SA2D_TLE_KERNELS.get(torch.float32)
            if kernel is None:
                _SA2D_TLE_OK = False
                return _SA2D_TLE_OK
            R, K, S = 2, 4, 4
            idx = torch.tensor(
                [[0, 1, 2, 3], [3, 2, 1, 0]], device=device, dtype=torch.int64
            )
            src = torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                device=device,
                dtype=torch.float32,
            )
            # out[r, idx[r, s]] += src[r, s] with zero-init (zinit=1): a permutation
            # index + S==K means each out cell receives exactly one add.
            ref = torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [8.0, 7.0, 6.0, 5.0]], dtype=torch.float32
            )
            # Sentinel fill (not present in ref) so a silent no-op payload that
            # leaves the buffer untouched is detectable, rather than passing by
            # chance on allocator-reused memory.
            out = torch.full((R, K), -123.0, device=device, dtype=torch.float32)
            kernel[(_sa2d_grid(K, R),)](out, out, idx, src, R, K, S, src.stride(0), 1)
            _SA2D_TLE_OK = torch.equal(out.cpu(), ref)
        except Exception as e:  # pragma: no cover - any failure -> atomic fallback
            logger.debug("scatter_add tle probe failed, using fallback: %s", e)
            _SA2D_TLE_OK = False
    return _SA2D_TLE_OK


def span_for_slice(slice_n: int, block: int) -> int:
    if slice_n <= 0:
        return block
    if slice_n >= block:
        return slice_n
    return slice_n * (block // slice_n)


LOOP_CAP = 32
BLOCK_CAP = 16384


def block_for_span(span: int, block: int) -> int:
    while span // block > LOOP_CAP and block < BLOCK_CAP:
        block *= 2
    return block


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
    SPAN: tl.constexpr,
    EXACT_SPAN: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * SPAN
    if not EXACT_SPAN:
        limit = tl.minimum(block_start + SPAN, n_elements)
    arange = tl.arange(0, BLOCK_SIZE)
    for loop_iter in tl.static_range(LOOP):
        src_index_offsets = block_start + arange
        if EXACT_SPAN:
            mask = src_index_offsets < n_elements
        else:
            mask = src_index_offsets < limit
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
    code.newline()

    code.writeline("def base_block(args):")
    with code.indent():
        code.writeline("if(flag_gems.vendor_name in ['metax', 'iluvatar']):")
        with code.indent():
            code.writeline("return 256")
        code.writeline("return 128")
    code.newline()
    code.newline()

    code.writeline("def heur_span(args):")
    with code.indent():
        code.writeline("span = base_block(args)")
        code.writeline('slice_n = args["SLICE"]')
        code.writeline("if slice_n >= span:")
        with code.indent():
            code.writeline("span = slice_n")
        code.writeline("else:")
        with code.indent():
            code.writeline("span = slice_n * (span // slice_n)")
        code.writeline("return span")
    code.newline()
    code.newline()

    code.writeline("def heur_block(args):")
    with code.indent():
        code.writeline("block = base_block(args)")
        code.writeline("span = heur_span(args)")
        code.writeline(f"while span // block > {LOOP_CAP} and block < {BLOCK_CAP}:")
        with code.indent():
            code.writeline("block *= 2")
        code.writeline("return block")
    code.newline()
    code.newline()

    code.writeline("def loop_count(args):")
    with code.indent():
        code.writeline("return triton.cdiv(heur_span(args), heur_block(args))")
    code.newline()
    code.newline()

    code.writeline("def heur_exact(args):")
    with code.indent():
        code.writeline("return heur_span(args) % heur_block(args) == 0")
    code.newline()
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"SPAN": heur_span,')
            code.writeline('"LOOP": loop_count,')
            code.writeline('"EXACT_SPAN": heur_exact,')
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
            code.writeline("SLICE,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")
            code.writeline("SPAN: tl.constexpr,")
            code.writeline("EXACT_SPAN: tl.constexpr,")

    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("base = pid * SPAN")
        code.writeline("if not EXACT_SPAN:")
        with code.indent():
            code.writeline("limit = tl.minimum(base + SPAN, N)")
        code.writeline("offsets = base + tl.arange(0, BLOCK)")

        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("if EXACT_SPAN:")
            with code.indent():
                code.writeline("mask = offsets < N")
            code.writeline("else:")
            with code.indent():
                code.writeline("mask = offsets < limit")
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
    parameters: List[str] = []

    parameters.append("src_strided")
    parameters.append("index")
    parameters.append("inp")
    parameters.append("out")
    parameters.append("dim")
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

        code.writeline("SLICE = 1")
        code.writeline("for _i in range(dim, len(index_shapes)):")
        with code.indent():
            code.writeline("SLICE *= index_shapes[_i]")

        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(N, meta["SPAN"]), ')
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
                code.writeline("SLICE,")

        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
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


@libentry()
@triton.jit(do_not_specialize=["idx_ncols", "src_stride0", "out_ncols"])
def scatter_add_2d_kernel(
    idx_ptr,
    src_ptr,
    out_ptr,
    idx_ncols,
    src_stride0,
    out_ncols,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pr = tl.program_id(0)
    rowbase = pr.to(tl.int64) * idx_ncols
    srcbase = pr.to(tl.int64) * src_stride0
    outbase = pr.to(tl.int64) * out_ncols
    offs = tl.arange(0, BLOCK)
    for loop_iter in tl.static_range(LOOP):
        mask = offs < idx_ncols
        idx = tl.load(idx_ptr + rowbase + offs, mask=mask, other=0).to(tl.int64)
        srcv = tl.load(src_ptr + srcbase + offs, mask=mask, other=0)
        tl.atomic_add(out_ptr + outbase + idx, srcv, mask=mask, sem="relaxed")
        offs += BLOCK


def scatter_add_0(inp, dim, index, src):
    logger.debug("GEMS_KUNLUNXIN SCATTER_ADD_0")
    dtype_convert = False
    if inp.dtype == torch.float16 or inp.dtype == torch.bfloat16:
        out = inp.to(torch.float32)
        dtype_convert = True
    else:
        out = inp

    src_strided = src.as_strided(index.shape, src.stride())
    dim = dim % inp.ndim
    if inp.ndim == 2 and dim == 1 and index.is_contiguous():
        out_ncols = out.shape[1]
        idx_ncols = index.shape[1]
        src_stride0 = src_strided.stride(0)
        BLOCK = block_for_span(idx_ncols, 128)
        LOOP = triton.cdiv(idx_ncols, BLOCK)
        grid = (index.shape[0],)
        scatter_add_2d_kernel[grid](
            index,
            src_strided,
            out,
            idx_ncols,
            src_stride0,
            out_ncols,
            BLOCK=BLOCK,
            LOOP=LOOP,
        )
        if dtype_convert:
            return inp.copy_(out.to(src.dtype))
        return out

    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    _scatter_func(
        src_strided,
        index,
        inp_restrided,
        out,
        dim,
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
    logger.debug("GEMS_KUNLUNXIN SCATTER_ADD_1")
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
    BLOCK_SIZE = 256
    SPAN = span_for_slice(index_dim_n, BLOCK_SIZE)
    BLOCK_SIZE = block_for_span(SPAN, BLOCK_SIZE)
    LOOP = triton.cdiv(SPAN, BLOCK_SIZE)
    EXACT_SPAN = SPAN % BLOCK_SIZE == 0
    grid = (triton.cdiv(all_elem, SPAN),)

    dtype_convert = False
    if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        dtype_convert = True
        x = x.to(torch.float32)

    scatter_add_kernel_1[grid](
        index_dim_n,
        inp_dim_n,
        x,
        index,
        src,
        all_elem,
        BLOCK_SIZE=BLOCK_SIZE,
        LOOP=LOOP,
        SPAN=SPAN,
        EXACT_SPAN=EXACT_SPAN,
    )
    if dim != x.ndim - 1:
        order = [i for i in range(x.ndim - 1)]
        order.insert(dim, x.ndim - 1)
        if dtype_convert:
            return origin.copy_(x.to(src.dtype).permute(order))
        return x.permute(order)
    else:
        return x.to(src.dtype)


def _try_scatter_add_2d_tle(
    x, dim, index, src, inp=None, require_full_rows=False, zinit=False
):
    # Unified tle.raw on-chip fast-path for the 2D, dim==last scatter-add case.
    # Covers BOTH the dense (S>=K) case that otherwise falls to scatter_add_1's
    # slow atomic kernel AND the sparse (K>S) case. Row r resident in LM, no GM
    # atomics. Requires out row width K<=SA2D_TLE_MAX_K, contiguous 2D index,
    # and an fp32/fp16/bf16 target. Returns the result tensor, or None if the
    # preconditions do not hold (caller falls back to the generic paths).
    #
    # `inp` is the base tensor the payload reads each row FROM before scattering
    # onto it; the result is written to `x`. When inp is None the op is in-place
    # (inp == x). For the out-of-place scatter_add path the caller passes a FRESH
    # x (native torch.empty_like, NOT a clone) plus inp == the original input,
    # fusing the input copy INTO the payload and avoiding a clone/copy_ that would
    # be re-dispatched to a slow gems copy under flag_gems.use_gems(). Because a
    # fresh x has uninitialized rows [R, x.shape[0]) that the payload never
    # touches, the caller must set require_full_rows=True for that case (only fire
    # when index rows cover every out row).
    if not _HAS_SA2D_TLE:
        return None
    if not _sa2d_tle_usable(x.device):
        return None
    if x.ndim != 2 or dim != 1:
        return None
    if not index.is_contiguous() or not x.is_contiguous():
        return None
    if inp is not None and (not inp.is_contiguous() or inp.dtype != x.dtype):
        return None
    K = x.shape[1]
    R = index.shape[0]
    S = index.shape[1]
    # Pick the on-chip kernel: small K (<=1024) keeps the whole out row in per-core
    # LM; moderate-big K (<=8192) keeps a TILED out row in LM, single-owner, NON-
    # atomic (beats the SM-atomic path for K<=8192); very-big K (<=65536) keeps the
    # out row in SM with 64-core SM-atomic cooperation (the tile rescan explodes
    # past 8192). All bands cover fp32/fp16/bf16 (2-byte dtypes widen to f32).
    if K <= SA2D_TLE_MAX_K:
        kernel = _SA2D_TLE_KERNELS.get(x.dtype)
    elif K <= SA2D_TILE_MAX_K:
        kernel = _SA2D_TLE_TILE_KERNELS.get(x.dtype)
    elif K <= SA2D_TLE_BIG_MAX_K:
        kernel = _SA2D_TLE_BIG_KERNELS.get(x.dtype)
    else:
        kernel = None
    if kernel is None:
        return None

    # A fresh (uninitialized) out is only safe when every out row is written.
    if require_full_rows and R != x.shape[0]:
        return None
    base = x if inp is None else inp
    src_strided = src.as_strided(index.shape, src.stride())
    # Feed the row-strided (but column-contiguous) src view DIRECTLY, passing its
    # row stride to the payload. This avoids src_strided.contiguous(), which under
    # an active flag_gems.use_gems() context gets re-dispatched to a slow gems copy
    # kernel (~3.5ms for a 1024x2048 strided copy vs ~0.013ms native) and was the
    # dominant cost on the real benchmark path. The fp16/bf16 payloads widen to f32
    # on-chip, so x stays in its native dtype (NO host .to(f32)/copy-back) too.
    if src_strided.dtype == x.dtype and src_strided.stride(1) == 1:
        src_c = src_strided
        src_rs = src_strided.stride(0)
    else:
        src_c = src_strided.to(x.dtype).contiguous()
        src_rs = src_c.stride(0)
    idx_i64 = index if index.dtype == torch.int64 else index.to(torch.int64)
    # zinit=1 tells the payload to zero-init the accumulator row on-chip instead
    # of loading it from `base`; the caller then allocates an UNINITIALIZED out
    # (new_empty) and skips a separate new_zeros/clone. Only valid with
    # require_full_rows (every out row written).
    kernel[(_sa2d_grid(K, R),)](
        x, base, idx_i64, src_c, R, K, S, src_rs, 1 if zinit else 0
    )
    return x


def scatter_add_(x, dim, index, src):
    assert x.dim() == index.dim() and x.dim() == src.dim(), "Invalid dim"
    dim = dim % x.ndim
    assert dim >= 0 and dim < x.dim(), "Invalid dim"
    assert index.size(dim) <= src.size(dim), "Invalid src"

    tle_out = _try_scatter_add_2d_tle(x, dim, index, src)
    if tle_out is not None:
        return tle_out

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
    logger.debug("GEMS_KUNLUNXIN SCATTER_ADD")
    # Fused out-of-place fast path: allocate a FRESH out (native torch.empty_like,
    # cheap ~0.005ms) and let the on-chip payload read each base row from `inp`
    # and write the scattered result to out. This skips inp.clone(), whose
    # internal copy_ is re-dispatched to a slow gems copy under use_gems (~0.057ms
    # on tiny shapes -> dominated the tiny-shape latency). require_full_rows guards
    # against a fresh out leaking uninitialized rows when index rows < out rows.
    if _HAS_SA2D_TLE and inp.ndim == 2 and dim % inp.ndim == 1 and inp.is_contiguous():
        out = torch.empty_like(inp)
        fused = _try_scatter_add_2d_tle(
            out, dim % inp.ndim, index, src, inp=inp, require_full_rows=True
        )
        if fused is not None:
            return fused
    out = inp.clone()
    return scatter_add_(out, dim, index, src)
