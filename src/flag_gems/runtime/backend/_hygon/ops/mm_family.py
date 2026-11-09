"""Measure ordinary TL, cooperative, and reduction GEMM families."""

import logging
import statistics
import threading
from dataclasses import asdict
from functools import lru_cache
from math import prod

import torch
import triton

from flag_gems.runtime import torch_device_fn
from .mm_shared import (
    SharedMMConfig,
    ShortKMMConfig,
    launch_shared_mm,
    launch_short_k_mm,
    shared_mm_configs,
    short_k_mm_candidates,
)
from .mm_local_split import (
    LocalSplitMMConfig,
    local_split_mm_configs,
    launch_local_split_mm,
)
from .mm_cooperative import (
    CooperativeMMConfig,
    cooperative_mm_configs,
    launch_cooperative_mm,
)
from .mv import GroupedMVConfig, grouped_mv_candidates, launch_grouped_mv

logger = logging.getLogger(__name__)
_choices = {}
_choice_measurements = {}
_choice_lock = threading.RLock()
_SPLITK_LOCAL = (
    LocalSplitMMConfig((16, 16)),
    LocalSplitMMConfig((16, 16), num_stages=2),
)
_SPLITK_LOCAL_WIDE = LocalSplitMMConfig((32, 32), num_stages=2)
_SPLITK_SHARED = SharedMMConfig((32, 64, 64))


@lru_cache(maxsize=1)
def _shared_supported():
    try:
        from triton.backends.hcu.compiler import HIPOptions
        import triton.experimental.tle.language as tle
    except ImportError:
        return False
    return (
        callable(tle.load)
        and {"prefer_wide_async_copy", "mmac_warps_m", "lds_dma_alias_workaround"}
        <= HIPOptions.__dataclass_fields__.keys()
    )


def mm_family_candidates(a, b, c, *, splitk_pool=False):
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2 or a.device.type != "cuda":
        return ()
    with torch_device_fn.device(a.device):
        target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        return ()
    local_configs = local_split_mm_configs()
    shared_configs = shared_mm_configs() if _shared_supported() else ()
    if splitk_pool:
        m, n = c.shape
        compute_units = torch_device_fn.get_device_properties(a.device).multi_processor_count
        bm, bn = _SPLITK_LOCAL_WIDE.tile
        local_tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        local_shortlist = _SPLITK_LOCAL
        if (m >= bm and n >= bn) or local_tiles * 2 >= compute_units:
            local_shortlist += (_SPLITK_LOCAL_WIDE,)
        local_configs = tuple(config for config in local_shortlist if config in local_configs)
        # Consider the smallest Shared tile only when its CTA grid can
        # supply at least one wave per compute unit in aggregate.
        bm, bn, _ = _SPLITK_SHARED.tile
        shared_tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        shared_configs = (
            (_SPLITK_SHARED,)
            if _SPLITK_SHARED in shared_configs
            and shared_tiles * prod(_SPLITK_SHARED.warps) >= compute_units
            else ()
        )
    local = tuple(config for config in local_configs if config.supports(a, b, c))
    shared = tuple(config for config in shared_configs if config.supports(a, b, c))
    if splitk_pool:
        return local + shared + grouped_mv_candidates(a, b, compute_units)
    cooperative = tuple(
        config for config in cooperative_mm_configs() if config.supports(a, b, c)
    )
    short_k = short_k_mm_candidates(a, b, c) if shared_configs else ()
    return shared + cooperative + local + short_k


def _launch(a, b, c, candidate):
    if isinstance(candidate, SharedMMConfig):
        return launch_shared_mm(a, b, c, candidate)
    if isinstance(candidate, CooperativeMMConfig):
        return launch_cooperative_mm(a, b, c, candidate)
    if isinstance(candidate, GroupedMVConfig):
        return launch_grouped_mv(a, b, c, candidate)
    if isinstance(candidate, ShortKMMConfig):
        return launch_short_k_mm(a, b, c, candidate)
    return launch_local_split_mm(a, b, c, candidate)


def _label(candidate):
    if isinstance(candidate, CooperativeMMConfig):
        return dict(family="cooperative", **asdict(candidate))
    if isinstance(candidate, LocalSplitMMConfig):
        return dict(family="local_split", **asdict(candidate))
    if isinstance(candidate, GroupedMVConfig):
        return dict(family="grouped_mv", **asdict(candidate))
    if isinstance(candidate, ShortKMMConfig):
        return dict(family="short_k", **asdict(candidate))
    return asdict(candidate) if isinstance(candidate, SharedMMConfig) else candidate


def clear_mm_family_choices():
    with _choice_lock:
        _choices.clear()
        _choice_measurements.clear()


def try_mm_family(
    a, b, c, ordinary, *, tuning_key, ordinary_with_output=None, splitk_pool=False
):
    """Compare real candidates with the existing Default/Expanded TL tuner.

    Cache only the selected index, never tensors. A cold graph capture defers
    family selection to the incumbent path. In the Split-K region, the
    incumbent is the Split-K tuner and only a fixed shortlist challenges it.
    """
    if any(t.ndim != 2 for t in (a, b, c)) or a.device.type != "cuda":
        return False
    m, k = a.shape
    n = b.shape[1]
    begin, end = c.data_ptr(), c.data_ptr() + 2 * m * n
    storage = (
        tuple(t.data_ptr() % 16 == 0 for t in (a, b, c)),
        all(
            end <= t.data_ptr() or t.data_ptr() + 2 * count <= begin
            for t, count in ((a, m * k), (b, k * n))
        ),
        c.untyped_storage().data_ptr()
        not in (a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr()),
        c.requires_grad,
        c.is_inference(),
        torch.is_inference_mode_enabled(),
    )
    key = (
        a.device,
        b.device,
        c.device,
        a.dtype,
        b.dtype,
        c.dtype,
        tuple(a.shape),
        tuple(b.shape),
        tuple(c.shape),
        tuple(a.stride()),
        tuple(b.stride()),
        tuple(c.stride()),
        tuning_key,
        (
            "shared-family-v5-grouped-mv",
            storage,
            splitk_pool,
            ordinary_with_output is not None,
        ),
    )
    with torch_device_fn.device(a.device):
        cached = _choices.get(key)
        if cached is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            with _choice_lock:
                cached = _choices.get(key)
                if cached is None:
                    candidates = mm_family_candidates(a, b, c, splitk_pool=splitk_pool)
                    if not candidates:
                        return False
                    from triton.flagtune.runtime.graph_benchmark import (
                        do_bench_hcu_graph,
                    )

                    ordinary()
                    calls = (ordinary,) + tuple(
                        lambda candidate=candidate: _launch(a, b, c, candidate)
                        for candidate in candidates
                    )
                    for call in calls[1:]:
                        call()
                    samples = tuple([] for _ in calls)
                    for repeat in range(3):
                        order = range(len(calls))
                        if repeat % 2:
                            order = reversed(order)
                        for index in order:
                            samples[index].append(
                                float(
                                    do_bench_hcu_graph(
                                        calls[index],
                                        rep=10,
                                        n_retries=3,
                                        device_interface=torch.cuda,
                                    )
                                )
                            )
                    medians = tuple(statistics.median(x) for x in samples)
                    choice = min(range(len(calls)), key=medians.__getitem__)
                    finalists = tuple(
                        i
                        for i in sorted(range(len(calls)), key=medians.__getitem__)
                        if medians[i] <= medians[choice] * 1.05
                    )[:3]
                    refined = {}
                    if ordinary_with_output is not None and len(finalists) > 1:
                        refined_samples = {i: [] for i in finalists}
                        for repeat in range(3):
                            order = (
                                finalists if repeat % 2 == 0 else reversed(finalists)
                            )
                            for index in order:

                                def fresh_output():
                                    output = torch.empty_like(c)
                                    if index:
                                        _launch(a, b, output, candidates[index - 1])
                                    else:
                                        ordinary_with_output(output)
                                    return output

                                refined_samples[index].append(
                                    float(
                                        do_bench_hcu_graph(
                                            fresh_output,
                                            rep=100,
                                            n_retries=10,
                                            device_interface=torch.cuda,
                                        )
                                    )
                                )
                        refined = {
                            i: statistics.median(x) for i, x in refined_samples.items()
                        }
                        choice = min(finalists, key=refined.__getitem__)
                    _choice_measurements[key] = dict(
                        variants=(("splitk_pool" if splitk_pool else "ordinary_tl"),)
                        + tuple(_label(x) for x in candidates),
                        screened_ms=medians,
                        refined_allocating_ms=refined,
                        choice=choice,
                    )
                    cached = (choice, candidates)
                    _choices[key] = cached
                    logger.debug(
                        "Hygon MM family=%s screening=%s refined=%s ms",
                        choice,
                        medians,
                        refined,
                    )
        choice, candidates = cached
        if choice:
            _launch(a, b, c, candidates[choice - 1])
        else:
            ordinary()
    return True
