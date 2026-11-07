"""Measure ordinary shared/TL, cooperative, and CTA-local reduction GEMM."""

import logging
import statistics
import threading
from dataclasses import asdict
from functools import lru_cache

import torch
import triton

from flag_gems.runtime import torch_device_fn
from .mm_shared import SharedMMConfig, launch_shared_mm, shared_mm_configs
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

logger = logging.getLogger(__name__)
_choices = {}
_choice_measurements = {}
_choice_lock = threading.RLock()


@lru_cache(maxsize=1)
def _shared_supported():
    try:
        from triton.backends.hcu.compiler import HIPOptions
        from triton.experimental.gluon import language as gl
        from triton.experimental.gluon.language.hcu import matrix_load_to_shared
    except ImportError:
        return False
    return (
        hasattr(gl, "SharedLinearLayout")
        and callable(matrix_load_to_shared)
        and {"mma_phase_schedule", "post_mma_wait", "lds_dma_alias_workaround"}
        <= HIPOptions.__dataclass_fields__.keys()
    )


def mm_family_candidates(a, b, c, *, local_split_only=None):
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2 or a.device.type != "cuda":
        return ()
    with torch_device_fn.device(a.device):
        target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        return ()
    local = tuple(config for config in local_split_mm_configs() if config.supports(a, b, c))
    if local_split_only is not None:
        requested = tuple(local_split_only)
        return requested if requested and all(x in local for x in requested) else ()
    shared = (
        tuple(config for config in shared_mm_configs() if config.supports(a, b, c))
        if _shared_supported()
        else ()
    )
    cooperative = tuple(
        config for config in cooperative_mm_configs() if config.supports(a, b, c)
    )
    return shared + cooperative + local


def _launch(a, b, c, candidate):
    if isinstance(candidate, SharedMMConfig):
        return launch_shared_mm(a, b, c, candidate)
    if isinstance(candidate, CooperativeMMConfig):
        return launch_cooperative_mm(a, b, c, candidate)
    return launch_local_split_mm(a, b, c, candidate)


def _label(candidate):
    if isinstance(candidate, CooperativeMMConfig):
        return dict(family="cooperative", **asdict(candidate))
    if isinstance(candidate, LocalSplitMMConfig):
        return dict(family="local_split", **asdict(candidate))
    return asdict(candidate) if isinstance(candidate, SharedMMConfig) else candidate


def clear_mm_family_choices():
    with _choice_lock:
        _choices.clear()
        _choice_measurements.clear()


def try_mm_family(
    a, b, c, ordinary, *, tuning_key, ordinary_with_output=None, local_split_only=None
):
    """Compare real candidates with the existing Default/Expanded TL tuner.

    Cache only the selected index, never tensors. A cold graph capture defers
    family selection to the already established ordinary path. Input dtype,
    layout, tuning protocol and the actual candidate tuple enter cache identity.
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
            "shared-family-v3-local-config",
            storage,
            None if local_split_only is None else tuple(local_split_only),
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
                    candidates = mm_family_candidates(
                        a, b, c, local_split_only=local_split_only
                    )
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
                        variants=("ordinary_tl",)
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
