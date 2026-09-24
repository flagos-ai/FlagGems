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

import inspect
import logging
import os
import threading
import time

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from flag_gems.ops.contiguous import contiguous as _gems_contiguous
from flag_gems.ops.copy import copy_ as _gems_copy_
from flag_gems.ops.rrelu_with_noise import (
    _check_rrelu_with_noise_args,
    _rrelu_with_noise_impl,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu

try:
    # Direct C bindings, skipping the backend_register/python wrapper chain
    # (~6 us per query through the driver shim vs ~1 us direct).
    from torch_npu._C import _npu_getCurrentRawStream as _raw_stream
except ImportError:  # pragma: no cover - torch_npu always ships it
    _raw_stream = None

logger = logging.getLogger(__name__)

DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 0.3333333333333333

# Element offsets use int32 below this numel; larger tensors take the WIDE
# variant with int64 addressing.
_WIDE_NUMEL_THRESHOLD = 1 << 30


def _pop_all_blocks_parallel():
    # TRITON_ALL_BLOCKS_PARALLEL (set globally by the fused sparse-attention
    # module at import) is read at BOTH kernel compile and launch time by this
    # triton-ascend build, and the two must agree: with it set, kernels that
    # read and write the same buffer (the in-place variants here) misbehave.
    # Pop it around compile+launch and restore afterwards (same pattern as
    # linalg_matrix_rank). Plain function pair instead of a contextmanager to
    # keep the per-launch overhead minimal.
    return os.environ.pop("TRITON_ALL_BLOCKS_PARALLEL", None)


def _restore_all_blocks_parallel(saved):
    if saved is not None:
        os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = saved


# Golden-ratio constant for folding high counter/seed bits into the hash key.
_GOLDEN32 = 0x9E3779B9


def _rng_key(seed, offset):
    """Fold the 64-bit generator seed and offset into the kernel's hash inputs.

    The kernels hash a 32-bit per-element counter; folding the high halves of
    seed and offset into a per-call key keeps sequences distinct when seeds
    differ only in their high bits or the counter wraps past 2**32 elements.
    Both results are returned as signed int32 bit patterns so the triton
    signature is stable for any generator state.
    """
    key = (seed ^ (seed >> 32)) & 0xFFFFFFFF
    key ^= ((offset >> 32) * _GOLDEN32) & 0xFFFFFFFF
    off_lo = offset & 0xFFFFFFFF
    if key >= 1 << 31:
        key -= 1 << 32
    if off_lo >= 1 << 31:
        off_lo -= 1 << 32
    return key, off_lo


def _u64_as_i64(v):
    # torch.tensor rejects ints above the signed 64-bit range; store the
    # seed's bit pattern instead (the graph kernel reads it back as uint64).
    return v - (1 << 64) if v >= (1 << 63) else v


@triton.jit
def _rrelu_noise_u32(h):
    # lowbias32 integer finalizer: a few vectorized uint32 mul/shift/xor ops.
    # tl.philox is not used on purpose: it compiles to a prohibitively slow
    # sequence on the current triton-ascend toolchain.
    h ^= h >> 16
    h = h * 0x21F0AAAD
    h ^= h >> 15
    h = h * 0x735A2D97
    h ^= h >> 15
    return h


@triton.jit(do_not_specialize=["key", "off_lo", "N"])
def fused_rrelu_with_noise_train_kernel(
    x_ptr,
    out_ptr,
    noise_ptr,
    N,
    lower,
    span,
    key,
    off_lo,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ctr32 = off.to(tl.int32)
    else:
        off = pid * BLOCK + tl.arange(0, BLOCK)
        ctr32 = off
    mask = off < N
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    # Keep the counter arithmetic in 32 bits: int64 vector ops are emulated
    # on the vector cores. key (host-folded from the full seed and the high
    # offset bits) keeps streams distinct across counter wraparounds.
    ctr = (ctr32 + off_lo).to(tl.uint32) ^ key.to(tl.uint32)
    if WIDE:
        # Fold the high index bits in as well: tensors past 2**32 elements
        # would otherwise repeat the sequence every 2**32 lanes.
        ctr ^= ((off >> 32).to(tl.int32) * (-1640530511)).to(tl.uint32)
    u = (_rrelu_noise_u32(ctr) & 0x007FFFFF).to(tl.int32, bitcast=True).to(
        tl.float32
    ) * (1.0 / 8388608.0)
    n = lower + span * u
    # CPU/CUDA ATen samples non-positive inputs; signed zero draws a slope and
    # NaN takes the unit-slope path.
    neg = x <= 0.0
    tl.store(
        out_ptr + off,
        tl.where(neg, x * n, x).to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        noise_ptr + off,
        tl.where(neg, n, 1.0).to(noise_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit(do_not_specialize=["N"])
def fused_rrelu_with_noise_train_graph_kernel(
    x_ptr,
    out_ptr,
    noise_ptr,
    so_ptr,
    N,
    lower,
    span,
    BLOCK: tl.constexpr,
):
    # Graph-captured variant: seed/offset come from a two-element int64
    # device buffer so replays stay valid while the RNG stream advances
    # (see _rrelu_advance_offset_kernel below). The 64-bit folding matches
    # _rng_key exactly, so replayed and eager draws agree.
    seed_u = tl.load(so_ptr).to(tl.uint64, bitcast=True)
    off_u = tl.load(so_ptr + 1).to(tl.uint64, bitcast=True)
    key = (seed_u ^ (seed_u >> 32) ^ ((off_u >> 32) * 0x9E3779B9)).to(tl.uint32)
    off_lo = off_u.to(tl.int32)
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    ctr = (off + off_lo).to(tl.uint32) ^ key
    u = (_rrelu_noise_u32(ctr) & 0x007FFFFF).to(tl.int32, bitcast=True).to(
        tl.float32
    ) * (1.0 / 8388608.0)
    n = lower + span * u
    neg = x <= 0.0
    tl.store(
        out_ptr + off,
        tl.where(neg, x * n, x).to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        noise_ptr + off,
        tl.where(neg, n, 1.0).to(noise_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit(do_not_specialize=["inc"])
def _rrelu_advance_offset_kernel(so_ptr, inc):
    # Runs after the training kernel inside the captured graph so each replay
    # consumes a fresh counter range, matching philox increment semantics.
    tl.store(so_ptr + 1, tl.load(so_ptr + 1) + inc)


@triton.jit(do_not_specialize=["N"])
def rrelu_with_noise_eval_kernel(
    x_ptr,
    out_ptr,
    N,
    slope,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    # Several tiles per program: on this backend every program pays a fixed
    # setup cost that dominates small tiles, so unrolling roughly doubles
    # large-shape bandwidth (16M fp16: 74 -> 40 us; fp32: 148 -> 96 us).
    for i in tl.static_range(UNROLL):
        if WIDE:
            off = pid.to(tl.int64) * (BLOCK * UNROLL) + i * BLOCK + tl.arange(0, BLOCK)
        else:
            off = pid * (BLOCK * UNROLL) + i * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        # Compute in the element dtype: the upcast path is measurably slower
        # for 2-byte dtypes, and the fp16/bf16 rounding matches torch_npu.
        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        y = tl.where(x > 0.0, x, x * slope.to(out_ptr.dtype.element_ty))
        tl.store(out_ptr + off, y, mask=mask)


# Direct CompiledKernel launches bypass JITFunction.run, whose per-call
# overhead on this triton-ascend build (~0.4 ms) dwarfs the kernel time of a
# pointwise op. Same approach as linalg_matrix_rank._fast_launch: the cache
# key covers every property the compiled binary specializes on (constexprs,
# grid, tensor alignment, pointer dtype); N/seed/offset are marked
# do_not_specialize so they stay plain runtime arguments. Everything slow
# (compile lock, the TRITON_ALL_BLOCKS_PARALLEL pop, the device context)
# happens only on the cache-miss branch; a cached launch is just a stream
# query plus the C launcher call (~15 us).
_FAST_LAUNCH_CACHE = {}
_COMPILE_LOCK = threading.Lock()
_DRV = None  # driver.active is a LazyProxy; resolve it once


def _time_query(fn, arg, iters=200):
    """Wall-time one device/stream query candidate; None when it fails."""
    try:
        fn() if arg is None else fn(arg)
    except Exception:
        return None
    t0 = time.perf_counter()
    for _ in range(iters):
        fn() if arg is None else fn(arg)
    return time.perf_counter() - t0


def _pick_query(candidates, arg):
    """Keep the fastest candidate: the shims' cost differs wildly across
    torch_npu builds, so pick by measurement rather than by version."""
    scored = [(_time_query(fn, arg), fn) for fn in candidates]
    return min(scored, key=lambda t: float("inf") if t[0] is None else t[0])[1]


_DEVICE_QUERY = None
_STREAM_QUERY = None
_DEVICE_COUNT = None


def _device_count():
    global _DEVICE_COUNT
    if _DEVICE_COUNT is None:
        _DEVICE_COUNT = torch.npu.device_count()
    return _DEVICE_COUNT


def _current_device():
    global _DEVICE_QUERY
    if _DEVICE_QUERY is None:
        _DEVICE_QUERY = _pick_query(
            [torch.npu.current_device, torch_device_fn.current_device], None
        )
    return _DEVICE_QUERY()


def _current_stream(dev_idx):
    global _STREAM_QUERY, _DRV
    if _DRV is None:
        _DRV = driver.active
    if _STREAM_QUERY is None:
        candidates = [_DRV.get_current_stream]
        if _raw_stream is not None:
            candidates.insert(0, _raw_stream)
        _STREAM_QUERY = _pick_query(candidates, dev_idx)
    return _STREAM_QUERY(dev_idx)


def _compile_entry(kernel, grid, key, device, args, kwargs):
    global _DRV
    saved = _pop_all_blocks_parallel()
    try:
        with torch_device_fn.device(device):
            compiled = kernel.warmup(*args, grid=grid, **kwargs)
            compiled._init_handles()
    finally:
        _restore_all_blocks_parallel(saved)
    if hasattr(kernel, "non_constexpr_indices"):
        suffix = ()
    else:
        # Some frontends expect the constexpr values appended to the launch
        # arguments; resolve that once at compile time.
        suffix = tuple(kwargs[name] for name in kernel.arg_names[len(args) :])
    entry = (
        compiled.run,
        compiled.function,
        compiled.packed_metadata,
        compiled.launch_metadata,
        suffix,
        compiled,
    )
    _FAST_LAUNCH_CACHE[key] = entry
    _DRV = driver.active
    return entry


def _fast_launch(kernel, grid, key_extra, device, *args, **kwargs):
    # The cache key and the launch stream are both tied to the input's device:
    # a compiled function handle is only valid on the device it was loaded on,
    # and the launch must go to that device's current stream. The runtime
    # additionally rejects launches whose stream belongs to a device context
    # other than the thread's current one ("stream is not in current ctx"),
    # so cross-device calls must also switch the device context; the guard is
    # only entered when the index actually differs.
    # The grid is deliberately NOT part of the key: every runtime int argument
    # is do_not_specialize, so one compiled entry serves all grid sizes of the
    # same constexpr/alignment configuration and a new shape on an already-seen
    # configuration costs a plain dict lookup instead of a kernel.warmup call
    # (which is slow enough to poison benchmark iteration-count estimates).
    dev_idx = device.index
    if dev_idx is None:
        dev_idx = torch_device_fn.current_device()
    key = (
        id(kernel),
        dev_idx,
        key_extra,
        tuple(kwargs.values()),
        tuple(a.data_ptr() % 16 == 0 if torch.is_tensor(a) else None for a in args),
    )
    entry = _FAST_LAUNCH_CACHE.get(key)
    if entry is None:
        with _COMPILE_LOCK:
            entry = _FAST_LAUNCH_CACHE.get(key)
            if entry is None:
                entry = _compile_entry(kernel, grid, key, device, args, kwargs)
    run, function, md, launch_metadata, suffix, _ = entry
    launch_args = args + suffix
    saved = _pop_all_blocks_parallel()
    try:
        # Single-device hosts never need the query at all.
        if _device_count() == 1 or _current_device() == dev_idx:
            stream = _current_stream(dev_idx)
            lm = launch_metadata(grid, stream, *launch_args)
            run(grid[0], 1, 1, stream, function, md, lm, None, None, *launch_args)
        else:
            with torch_device_fn.device(device):
                stream = _current_stream(dev_idx)
                lm = launch_metadata(grid, stream, *launch_args)
                run(grid[0], 1, 1, stream, function, md, lm, None, None, *launch_args)
    finally:
        _restore_all_blocks_parallel(saved)


def _heur(name, N, dtype=None):
    cfg = _hcu.HEURISTICS_CONFIGS[name]
    args = {"N": N, "dtype": dtype}
    return {key: fn(args) for key, fn in cfg.items()}


# ---------------------------------------------------------------------------
# NPUGraph path for small tensors.
#
# Below _GRAPH_MAX_NUMEL (training) / _GRAPH_MAX_NUMEL_EVAL elements the fixed
# per-call overhead (arg checking, generator bookkeeping, kernel launch)
# dwarfs the kernel time, so in-place calls are replayed from a captured graph
# instead: pointer-stable callers (training loops, benchmarks) then pay only a
# graph replay (~10 us).  Eval graphs pay off up to much larger sizes: replay
# keeps the launch queue fed so the measured latency approaches the pure
# kernel time.  Out-of-place calls are excluded: replaying into a static
# output buffer and cloning it costs more than launching the kernel into a
# fresh output.
# Same capture pattern as linalg_matrix_rank.
# ---------------------------------------------------------------------------
_GRAPH_MAX_NUMEL = 1 << 20
_GRAPH_MAX_NUMEL_EVAL = 1 << 24
_GRAPH_MAX_ENTRIES = 64
_GRAPH_ENTRIES = {}
_GRAPH_LOCK = threading.Lock()
# Capture only after the same configuration has been called a few times:
# benchmarks probe the latency with a handful of calls before deciding the
# timed-iteration count, and an early capture (~0.4 s) inside that probe would
# poison the estimate. One-off callers never pay for capture either.
_GRAPH_CAPTURE_AFTER = 8
_GRAPH_SEEN = {}

# torch.npu.graph without an explicit stream captures on a process-wide
# default stream created on whichever device was current when the first graph
# context was built; that stream stays bound to that device for every later
# capture. Keep one capture stream per device so captures always run on the
# input's device, matching the stream _fast_launch picks for the kernels.
_CAPTURE_STREAMS = {}


def _device_index(device):
    if device.index is not None:
        return device.index
    return torch_device_fn.current_device()


def _capture_stream(dev_idx):
    stream = _CAPTURE_STREAMS.get(dev_idx)
    if stream is None:
        stream = torch.npu.Stream(device=dev_idx)
        _CAPTURE_STREAMS[dev_idx] = stream
    return stream


# Older torch_npu builds take no stream argument in torch.npu.graph; detect
# once. Without it the capture falls back to torch_npu's process-wide default
# capture stream, which stays correct for single-device callers (it is created
# lazily, inside our device context) and only multi-device captures lose the
# per-device binding.
try:
    _GRAPH_ACCEPTS_STREAM = "stream" in inspect.signature(torch.npu.graph).parameters
except (AttributeError, TypeError, ValueError):  # pragma: no cover
    _GRAPH_ACCEPTS_STREAM = False


def _graph_context(graph, device):
    kwargs = {}
    if _GRAPH_ACCEPTS_STREAM:
        kwargs["stream"] = _capture_stream(_device_index(device))
    return torch.npu.graph(graph, **kwargs)


# Configurations whose capture already failed once: retrying capture on every
# call costs ~0.2-0.3 ms each time (observed on a host whose torch_npu rejects
# the capture), which is far worse than never graphing at all.
_GRAPH_FAILED = set()


def _replay(ent, device):
    # Replay queues the captured kernels on the current stream, so it must run
    # under the graph's own device context; guard only when it differs.
    if _device_count() == 1 or _current_device() == _device_index(device):
        ent.graph.replay()
    else:
        with torch_device_fn.device(device):
            ent.graph.replay()


class _GraphEntry:
    __slots__ = ("graph", "out_s", "refs", "so", "expected", "inc")

    def __init__(self, graph, out_s, refs, so=None, expected=None, inc=0):
        self.graph = graph
        self.out_s = out_s
        self.refs = refs  # keep captured buffers alive
        self.so = so  # device [seed, offset] counter (training only)
        self.expected = expected  # expected host-side (seed, offset) (training)
        self.inc = inc  # per-replay offset increment (training)


def _graph_key(training, self, noise, lower, upper, out):
    return (
        training,
        out is not None,
        self.numel(),
        self.dtype,
        float(lower),
        float(upper),
        self.data_ptr(),
        noise.data_ptr(),
        self.device.index,
    )


def _graph_eval(self, noise, slope, out):
    key = _graph_key(False, self, noise, slope, 0.0, out)
    ent = _GRAPH_ENTRIES.get(key)
    if ent is not None:
        _replay(ent, self.device)
        return self

    if key in _GRAPH_FAILED or len(_GRAPH_ENTRIES) >= _GRAPH_MAX_ENTRIES:
        return None

    seen = _GRAPH_SEEN.get(key, 0) + 1
    if len(_GRAPH_SEEN) > 4 * _GRAPH_MAX_ENTRIES:
        _GRAPH_SEEN.clear()
        seen = 1
    _GRAPH_SEEN[key] = seen
    if seen < _GRAPH_CAPTURE_AFTER:
        return None
    N = self.numel()
    heur = _heur("rrelu_with_noise_eval", N, self.dtype)
    BLOCK = heur["BLOCK"]
    grid = (triton.cdiv(N, BLOCK * heur["UNROLL"]),)

    def launch():
        _fast_launch(
            rrelu_with_noise_eval_kernel,
            grid,
            (self.dtype, False),
            self.device,
            self,
            self,
            N,
            float(slope),
            BLOCK=BLOCK,
            WIDE=False,
            UNROLL=heur["UNROLL"],
            num_warps=heur["num_warps"],
        )

    launch()  # compiles (illegal during capture) and computes the first result
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    saved = _pop_all_blocks_parallel()
    try:
        with torch_device_fn.device(self.device):
            with _graph_context(graph, self.device):
                launch()
    except Exception:
        logger.warning(
            "NPUGraph capture failed for rrelu_with_noise eval; "
            "using direct launches",
            exc_info=True,
        )
        _GRAPH_FAILED.add(key)
        _restore_all_blocks_parallel(saved)
        return self
    _restore_all_blocks_parallel(saved)
    # The timing replays re-apply the in-place kernel, so snapshot and
    # restore the buffer around the check.
    snap = self.clone()
    if not _replay_beats_launch(graph, launch):
        _GRAPH_FAILED.add(key)
    else:
        _GRAPH_ENTRIES[key] = _GraphEntry(graph, self, (self, noise, snap))
    self.copy_(snap)
    return self


def _replay_beats_launch(graph, launch):
    """Keep the graph only when replay actually beats a direct launch.

    Measured once per configuration at capture time: some torch_npu builds
    replay slowly enough that the graph is a net loss (on one CANN 9.0 host a
    single-kernel replay costs ~50 us, on par with a direct launch, and
    back-to-back replays then run *slower* than launches).  The first replay
    pays graph instantiation, and single timings are noisy, so warm up first
    and average a few rounds.  The replays re-apply the (in-place) kernel, so
    callers snapshot the buffers beforehand and restore them afterwards.
    """
    torch.npu.synchronize()
    for _ in range(3):
        graph.replay()
    t0 = time.perf_counter()
    for _ in range(5):
        launch()
    torch.npu.synchronize()
    direct_s = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(5):
        graph.replay()
    torch.npu.synchronize()
    replay_s = (time.perf_counter() - t0) / 5
    if replay_s <= direct_s * 1.1:
        return True
    logger.warning(
        "NPUGraph replay is slower than a direct launch for rrelu_with_noise "
        "(%.0f us vs %.0f us); using direct launches",
        replay_s * 1e6,
        direct_s * 1e6,
    )
    return False


def _lean_seed_offset(increment, generator, device):
    # Ascend-local lean equivalent of philox_backend_seed_offset: the NPU
    # generator state is [seed, offset] int64, and get_offset/set_offset hit
    # the same fields with far less Python/Tensor overhead than manipulating
    # the state ByteTensor. The offset increment is rounded to a multiple of
    # 4 exactly like philox_backend_seed_offset.
    if generator is None:
        generator = torch_device_fn.default_generators[
            (
                device.index
                if device.index is not None
                else torch_device_fn.current_device()
            )
        ]
    seed = generator.initial_seed()
    offset = generator.get_offset()
    inc = (increment + 3) // 4 * 4
    generator.set_offset(offset + inc)
    return generator, seed, offset, inc


def _graph_train(self, noise, lower, upper, generator, out):
    N = self.numel()
    lower = float(lower)
    span = float(upper) - lower

    device = self.device
    key = _graph_key(True, self, noise, lower, upper, out)
    ent = _GRAPH_ENTRIES.get(key)
    if ent is not None:
        gen = generator
        if gen is None:
            gen = torch_device_fn.default_generators[
                (
                    device.index
                    if device.index is not None
                    else torch_device_fn.current_device()
                )
            ]
        seed = gen.initial_seed()
        offset = gen.get_offset()
        if (seed, offset) != ent.expected:
            # The generator was reseeded externally: resync the device counter.
            ent.so.copy_(torch.tensor([_u64_as_i64(seed), offset], dtype=torch.int64))
            ent.expected = (seed, offset)
        _replay(ent, device)
        ent.expected = (seed, offset + ent.inc)
        gen.set_offset(offset + ent.inc)
        return self

    if key in _GRAPH_FAILED or len(_GRAPH_ENTRIES) >= _GRAPH_MAX_ENTRIES:
        return None

    # Defer capture until the configuration recurs; in particular this keeps
    # the ~0.4 s capture out of benchmark latency probes.  The generator is
    # only advanced below, after the decision to capture.
    seen = _GRAPH_SEEN.get(key, 0) + 1
    if len(_GRAPH_SEEN) > 4 * _GRAPH_MAX_ENTRIES:
        _GRAPH_SEEN.clear()
        seen = 1
    _GRAPH_SEEN[key] = seen
    if seen < _GRAPH_CAPTURE_AFTER:
        return None

    gen, seed, offset, inc = _lean_seed_offset(N, generator, device)

    so = torch.empty(2, dtype=torch.int64, device=device)
    so.copy_(torch.tensor([_u64_as_i64(seed), offset], dtype=torch.int64))
    heur = _heur("rrelu_with_noise_train", N)
    BLOCK = heur["BLOCK"]
    num_warps = heur["num_warps"]
    grid = (triton.cdiv(N, BLOCK),)

    def launch_main():
        _fast_launch(
            fused_rrelu_with_noise_train_graph_kernel,
            grid,
            (self.dtype, False),
            device,
            self,
            self,
            noise,
            so,
            N,
            lower,
            span,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

    def launch_advance():
        _fast_launch(
            _rrelu_advance_offset_kernel,
            (1,),
            None,
            device,
            so,
            inc,
            num_warps=1,
        )

    launch_main()  # compiles and computes the first result
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    saved = _pop_all_blocks_parallel()
    captured = True
    try:
        with torch_device_fn.device(device):
            with _graph_context(graph, device):
                launch_main()
                launch_advance()
    except Exception:
        logger.warning(
            "NPUGraph capture failed for rrelu_with_noise train; "
            "using direct launches",
            exc_info=True,
        )
        _GRAPH_FAILED.add(key)
        captured = False
    finally:
        _restore_all_blocks_parallel(saved)
    if captured:
        # The timing replays re-apply the in-place kernel and advance the
        # device counter; snapshot/restore everything around the check.
        snap_self = self.clone()
        snap_noise = noise.clone()
        replay_ok = _replay_beats_launch(graph, launch_main)
        self.copy_(snap_self)
        noise.copy_(snap_noise)
        so.copy_(torch.tensor([_u64_as_i64(seed), offset], dtype=torch.int64))
        if replay_ok:
            launch_advance()  # bring the device counter in line with the host
            _GRAPH_ENTRIES[key] = _GraphEntry(
                graph,
                self,
                (self, noise, so, snap_self, snap_noise),
                so=so,
                expected=(seed, offset + inc),
                inc=inc,
            )
        else:
            _GRAPH_FAILED.add(key)
    return self


def _try_graph(self, noise, lower, upper, training, generator, out):
    # Graphs pay off only for the in-place variants: replay alone is ~9 us,
    # while an out-of-place replay must additionally clone the static output
    # buffer (~50 us), which is slower than launching the kernel directly
    # into a fresh output tensor.
    if out is None:
        return None
    limit = _GRAPH_MAX_NUMEL if training else _GRAPH_MAX_NUMEL_EVAL
    if self.numel() > limit:
        return None
    with _GRAPH_LOCK:
        if training:
            return _graph_train(self, noise, lower, upper, generator, out)
        return _graph_eval(self, noise, (float(lower) + float(upper)) * 0.5, out)


def _fused_rrelu_with_noise_train(self, noise, out, lower, upper, generator):
    N = self.numel()
    lower = float(lower)
    span = float(upper) - lower

    # One hash counter per element; advancing the offset by N keeps successive
    # calls on disjoint counter ranges, matching philox increment semantics.
    _, seed, offset, _ = _lean_seed_offset(N, generator, self.device)
    key, off_lo = _rng_key(seed, offset)

    wide = N > _WIDE_NUMEL_THRESHOLD
    heur = _heur("rrelu_with_noise_train", N)
    grid = (triton.cdiv(N, heur["BLOCK"]),)
    _fast_launch(
        fused_rrelu_with_noise_train_kernel,
        grid,
        (self.dtype, wide),
        self.device,
        self,
        out,
        noise,
        N,
        lower,
        span,
        key,
        off_lo,
        BLOCK=heur["BLOCK"],
        WIDE=wide,
        num_warps=heur["num_warps"],
    )
    return out


def _rrelu_with_noise_eval_ascend(self, out, slope):
    N = self.numel()
    wide = N > _WIDE_NUMEL_THRESHOLD
    heur = _heur("rrelu_with_noise_eval", N, self.dtype)
    BLOCK = heur["BLOCK"]
    grid = (triton.cdiv(N, BLOCK * heur["UNROLL"]),)
    _fast_launch(
        rrelu_with_noise_eval_kernel,
        grid,
        (self.dtype, wide),
        self.device,
        self,
        out,
        N,
        float(slope),
        BLOCK=BLOCK,
        WIDE=wide,
        UNROLL=heur["UNROLL"],
        num_warps=heur["num_warps"],
    )
    return out


def _rrelu_with_noise_ascend_impl(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
    out=None,
):
    _check_rrelu_with_noise_args(self, noise, lower, upper)

    if self.numel() == 0:
        # ATen returns the out-of-place result in legacy contiguous layout.
        if out is None:
            return torch.empty_like(self, memory_format=torch.contiguous_format)
        return out

    supported_dtype = self.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if not supported_dtype:
        return _rrelu_with_noise_impl(
            self, noise, lower, upper, training, generator, out
        )

    contiguous = (
        self.is_contiguous()
        and noise.is_contiguous()
        and (out is None or out.is_contiguous())
    )
    if not contiguous:
        # Bounce through contiguous buffers: for training this also keeps the
        # CPU/CUDA-aligned sampling semantics (x <= 0) on strided layouts,
        # and for eval it avoids the generic pointwise_dynamic path, whose
        # aliased strided out0 writes are miscompiled on triton-ascend. The
        # bounced out-of-place result comes back contiguous, as ATen's does.
        # Layout copies go through the FlagGems copy ops (pointwise_dynamic);
        # only allocations use torch directly.
        self_c = _gems_contiguous(self)
        if training:
            noise_c = torch.empty_like(self_c)
            out_c = torch.empty_like(self_c)
            _fused_rrelu_with_noise_train(
                self_c, noise_c, out_c, lower, upper, generator
            )
            _gems_copy_(noise, noise_c)
        else:
            slope = (float(lower) + float(upper)) * 0.5
            out_c = torch.empty_like(self_c)
            _rrelu_with_noise_eval_ascend(self_c, out_c, slope)
        if out is None:
            return out_c
        _gems_copy_(out, out_c)
        return out

    if training:
        graph_result = _try_graph(self, noise, lower, upper, True, generator, out)
        if graph_result is not None:
            return graph_result
        if out is None:
            out = torch.empty_like(self)
        return _fused_rrelu_with_noise_train(self, noise, out, lower, upper, generator)

    graph_result = _try_graph(self, noise, lower, upper, False, generator, out)
    if graph_result is not None:
        return graph_result
    slope = (float(lower) + float(upper)) * 0.5
    if out is None:
        out = torch.empty_like(self)
    return _rrelu_with_noise_eval_ascend(self, out, slope)


def rrelu_with_noise(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """Ascend implementation of aten.rrelu_with_noise.

    Like the generic implementation, no Python autograd wrapper is built here:
    once registered on the device dispatch key, ``aten::rrelu_with_noise``
    keeps the autograd kernel PyTorch generates from its derivative formula,
    which calls the separately registered ``aten::rrelu_with_noise_backward``.
    """
    logger.debug("GEMS_ASCEND RRELU_WITH_NOISE")
    return _rrelu_with_noise_ascend_impl(self, noise, lower, upper, training, generator)


def rrelu_with_noise_(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """Ascend implementation of aten.rrelu_with_noise_."""
    logger.debug("GEMS_ASCEND RRELU_WITH_NOISE_")
    _rrelu_with_noise_ascend_impl(self, noise, lower, upper, training, generator, self)
    return self
