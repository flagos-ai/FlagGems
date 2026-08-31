"""Performance benchmarks for causal_conv1d / causal_conv1d_update.

Append to benchmark/test_special_perf.py, or run standalone.

NOTE on why this does not subclass `Benchmark` like FFTBenchmark does:
causal_conv1d is *stateful* -- it writes conv_states in place. The Benchmark
base class replays the same input iterator against torch_op and gems_op, which
would let one run's state mutation leak into the next. There is also no torch
reference op to compare against. So we benchmark the two Triton kernels
(baseline vs TLE) against each other directly, restoring state each iteration.
"""

import pytest
import torch
import triton

import flag_gems
from flag_gems.ops import causal_conv1d as cc1d
from flag_gems.utils.triton_version_utils import HAS_TLE

device = flag_gems.device

WARMUP = 10
REP = 100


def _has_tle_hw():
    if not (HAS_TLE and torch.cuda.is_available()):
        return False
    return torch.cuda.get_device_capability()[0] >= 9


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _make_varlen_data(dim, total_seqlen, batch, width, dtype, seed=0):
    torch.manual_seed(seed)
    eos_pos = torch.randperm(total_seqlen - 1)[: batch - 1].sort().values
    seqlens = torch.diff(
        torch.cat(
            [
                torch.tensor([-1], dtype=torch.int32),
                eos_pos.to(dtype=torch.int32),
                torch.tensor([total_seqlen - 1], dtype=torch.int32),
            ]
        )
    )
    query_start_loc = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            torch.cumsum(seqlens, dim=0).to(torch.int32),
        ]
    ).to(device)

    x = torch.randn(dim, total_seqlen, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_states = torch.randn(batch, dim, width - 1, device=device, dtype=dtype)
    cache_indices = torch.arange(batch, dtype=torch.int32, device=device)
    has_initial_state = torch.ones(batch, dtype=torch.bool, device=device)
    return (
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
    )


def _make_update_data(dim, batch, width, dtype, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch, dim, 1, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_state = torch.randn(batch, dim, width - 1, device=device, dtype=dtype)
    conv_state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    return x, weight, bias, conv_state, conv_state_indices


# ---------------------------------------------------------------------------
# Roofline metrics
#
# causal_conv1d is depthwise: FLOPs are tiny relative to bytes moved, so this is
# a memory-bound op. Report achieved bandwidth as the headline metric; TFLOPS is
# included only for completeness.
# ---------------------------------------------------------------------------


def _fwd_gbps(dim, total_seqlen, width, dtype, ms):
    itemsize = torch.tensor([], dtype=dtype).element_size()
    # read x, write out; weight/bias/state are negligible at these sizes
    bytes_moved = 2 * dim * total_seqlen * itemsize
    return bytes_moved / (ms * 1e-3) / 1e9


def _fwd_tflops(dim, total_seqlen, width, ms):
    # one FMA per (channel, token, tap)
    flops = 2 * dim * total_seqlen * width
    return flops / (ms * 1e-3) / 1e12


def _update_gbps(dim, batch, width, dtype, ms):
    itemsize = torch.tensor([], dtype=dtype).element_size()
    state_len = width - 1
    # read x + conv_state, write out + conv_state
    bytes_moved = (2 * batch * dim + 2 * batch * dim * state_len) * itemsize
    return bytes_moved / (ms * 1e-3) / 1e9


# ---------------------------------------------------------------------------
# Timing helper -- restores mutated state before each timed replay
# ---------------------------------------------------------------------------


def _bench_stateful(fn, state, state_pristine):
    """do_bench a closure whose `state` tensor is written in place.

    triton.testing.do_bench calls fn many times; without restoring the state,
    later iterations run on data the earlier ones corrupted. That does not
    change the amount of work (the kernel is data-independent), but it does keep
    the benchmark honest and makes the correctness check at the end meaningful.
    """

    def wrapped():
        state.copy_(state_pristine)
        fn()

    # Subtract the cost of the copy_ so it does not pollute the measurement.
    copy_ms = triton.testing.do_bench(
        lambda: state.copy_(state_pristine), warmup=WARMUP, rep=REP
    )
    total_ms = triton.testing.do_bench(wrapped, warmup=WARMUP, rep=REP)
    return max(total_ms - copy_ms, 1e-6)


def _run_with(kernel_forced_tle, fn):
    """Run `fn` with the TLE dispatcher forced on/off."""
    orig = cc1d._tle_available
    try:
        if not kernel_forced_tle:
            cc1d._tle_available = lambda _x: False
        return fn()
    finally:
        cc1d._tle_available = orig


# ---------------------------------------------------------------------------
# Forward benchmark
# ---------------------------------------------------------------------------

FWD_CONFIGS = [
    # (dim, batch)
    (4096, 32),
    (8192, 128),
]
FWD_SEQLENS = [2048, 4096, 8192, 16384]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.causal_conv1d
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_perf_causal_conv1d(dtype):
    width = 4
    tle_on = _has_tle_hw()

    print(
        f"\n=== causal_conv1d fwd | width={width} {dtype} | "
        f"TLE={'yes' if tle_on else 'no (baseline only)'} ==="
    )

    for dim, batch in FWD_CONFIGS:
        print(f"\n--- dim={dim} batch={batch} ---")
        header = f"{'seqlen':<10} {'base ms':<10} {'base GB/s':<11} {'base TF':<9}"
        if tle_on:
            header += (
                f" {'tle ms':<10} {'tle GB/s':<11} {'tle TF':<9} "
                f"{'speedup':<9} {'match':<6}"
            )
        print(header)

        for seqlen in FWD_SEQLENS:
            x, w, b, cs, qsl, ci, his = _make_varlen_data(
                dim, seqlen, batch, width, dtype
            )
            cs_pristine = cs.clone()

            base_ms = _run_with(
                False,
                lambda: _bench_stateful(
                    lambda: cc1d.causal_conv1d_fn(
                        x, w, b, cs, qsl, ci, his, activation="silu"
                    ),
                    cs,
                    cs_pristine,
                ),
            )
            row = (
                f"{seqlen:<10} {base_ms:<10.4f} "
                f"{_fwd_gbps(dim, seqlen, width, dtype, base_ms):<11.1f} "
                f"{_fwd_tflops(dim, seqlen, width, base_ms):<9.2f}"
            )

            if tle_on:
                tle_ms = _run_with(
                    True,
                    lambda: _bench_stateful(
                        lambda: cc1d.causal_conv1d_fn(
                            x, w, b, cs, qsl, ci, his, activation="silu"
                        ),
                        cs,
                        cs_pristine,
                    ),
                )

                # correctness spot-check alongside the timing
                cs_a = cs_pristine.clone()
                out_a = _run_with(
                    False,
                    lambda: cc1d.causal_conv1d_fn(
                        x, w, b, cs_a, qsl, ci, his, activation="silu"
                    ),
                )
                cs_b = cs_pristine.clone()
                out_b = _run_with(
                    True,
                    lambda: cc1d.causal_conv1d_fn(
                        x, w, b, cs_b, qsl, ci, his, activation="silu"
                    ),
                )
                try:
                    torch.testing.assert_close(
                        out_a.float(), out_b.float(), rtol=1e-2, atol=1e-2
                    )
                    torch.testing.assert_close(
                        cs_a.float(), cs_b.float(), rtol=1e-2, atol=1e-2
                    )
                    match = "pass"
                except AssertionError:
                    match = "FAIL"

                speedup = base_ms / tle_ms if tle_ms > 0 else float("nan")
                row += (
                    f" {tle_ms:<10.4f} "
                    f"{_fwd_gbps(dim, seqlen, width, dtype, tle_ms):<11.1f} "
                    f"{_fwd_tflops(dim, seqlen, width, tle_ms):<9.2f} "
                    f"{speedup:<9.2f} {match:<6}"
                )

            print(row)


# ---------------------------------------------------------------------------
# Update (decode) benchmark
# ---------------------------------------------------------------------------

UPDATE_BATCHES = [256, 1024]
UPDATE_DIMS = [1024, 2048, 4096, 8192]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.causal_conv1d
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_perf_causal_conv1d_update(dtype):
    width = 4
    tle_on = _has_tle_hw()

    print(
        f"\n=== causal_conv1d update | width={width} {dtype} | "
        f"TLE={'yes' if tle_on else 'no (baseline only)'} ==="
    )

    for batch in UPDATE_BATCHES:
        print(f"\n--- batch={batch} ---")
        header = f"{'dim':<10} {'base ms':<10} {'base GB/s':<11}"
        if tle_on:
            header += f" {'tle ms':<10} {'tle GB/s':<11} {'speedup':<9} {'match':<6}"
        print(header)

        for dim in UPDATE_DIMS:
            x, w, b, cs, csi = _make_update_data(dim, batch, width, dtype)
            cs_pristine = cs.clone()

            base_ms = _run_with(
                False,
                lambda: _bench_stateful(
                    lambda: cc1d.causal_conv1d_update(
                        x, cs, w, b, activation="silu", conv_state_indices=csi
                    ),
                    cs,
                    cs_pristine,
                ),
            )
            row = (
                f"{dim:<10} {base_ms:<10.4f} "
                f"{_update_gbps(dim, batch, width, dtype, base_ms):<11.1f}"
            )

            if tle_on:
                tle_ms = _run_with(
                    True,
                    lambda: _bench_stateful(
                        lambda: cc1d.causal_conv1d_update(
                            x, cs, w, b, activation="silu", conv_state_indices=csi
                        ),
                        cs,
                        cs_pristine,
                    ),
                )

                cs_a = cs_pristine.clone()
                out_a = _run_with(
                    False,
                    lambda: cc1d.causal_conv1d_update(
                        x, cs_a, w, b, activation="silu", conv_state_indices=csi
                    ),
                )
                cs_b = cs_pristine.clone()
                out_b = _run_with(
                    True,
                    lambda: cc1d.causal_conv1d_update(
                        x, cs_b, w, b, activation="silu", conv_state_indices=csi
                    ),
                )
                try:
                    torch.testing.assert_close(
                        out_a.float(), out_b.float(), rtol=1e-2, atol=1e-2
                    )
                    torch.testing.assert_close(
                        cs_a.float(), cs_b.float(), rtol=1e-2, atol=1e-2
                    )
                    match = "pass"
                except AssertionError:
                    match = "FAIL"

                speedup = base_ms / tle_ms if tle_ms > 0 else float("nan")
                row += (
                    f" {tle_ms:<10.4f} "
                    f"{_update_gbps(dim, batch, width, dtype, tle_ms):<11.1f} "
                    f"{speedup:<9.2f} {match:<6}"
                )

            print(row)


# ---------------------------------------------------------------------------
# Kernel-width sweep -- extract_tile's benefit scales with width, since that is
# how many separate strided loads the baseline issues per token.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_tle_hw(), reason="requires TLE on Hopper+")
@pytest.mark.causal_conv1d
def test_perf_causal_conv1d_width_sweep():
    dtype = torch.bfloat16
    dim, seqlen, batch = 4096, 8192, 32

    print(
        f"\n=== causal_conv1d fwd width sweep | dim={dim} seqlen={seqlen} {dtype} ==="
    )
    print(f"{'width':<8} {'base ms':<10} {'tle ms':<10} {'speedup':<9}")

    for width in (2, 3, 4):
        x, w, b, cs, qsl, ci, his = _make_varlen_data(dim, seqlen, batch, width, dtype)
        cs_pristine = cs.clone()

        base_ms = _run_with(
            False,
            lambda: _bench_stateful(
                lambda: cc1d.causal_conv1d_fn(
                    x, w, b, cs, qsl, ci, his, activation="silu"
                ),
                cs,
                cs_pristine,
            ),
        )
        tle_ms = _run_with(
            True,
            lambda: _bench_stateful(
                lambda: cc1d.causal_conv1d_fn(
                    x, w, b, cs, qsl, ci, his, activation="silu"
                ),
                cs,
                cs_pristine,
            ),
        )
        print(
            f"{width:<8} {base_ms:<10.4f} {tle_ms:<10.4f} " f"{base_ms / tle_ms:<9.2f}"
        )
