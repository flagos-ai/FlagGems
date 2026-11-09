"""Measured matrix-instruction candidate sharing the ordinary local-split GEMM.

The existing MV tuner keeps its complete configuration pool and protocol.
Only scalar family timings and indices are retained between public calls.
"""

import logging
import statistics
import threading
from functools import cache

import torch
import triton

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)
_choices = {}
_choice_measurements = {}
_choice_lock = threading.RLock()


@cache
def _mmac_config():
    from .mm_local_split import LocalSplitMMConfig

    return LocalSplitMMConfig((16, 16), num_stages=2)


def _eligible(inp, vec, out):
    from .mm_local_split import local_split_mm_configs

    n, k = inp.shape
    config = _mmac_config()
    if (
        vec.shape != (k,)
        or out.shape != (n,)
        or vec.stride() != (1,)
        or out.stride() != (1,)
        or config not in local_split_mm_configs()
    ):
        return False
    if not config.supports(vec.view(1, k), inp.t(), out.view(1, n)):
        return False
    with torch_device_fn.device(inp.device):
        target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx936"


def _launch_mmac(inp, vec, out):
    from .mm_local_split import launch_local_split_mm

    n, k = inp.shape
    return launch_local_split_mm(
        vec.view(1, k), inp.t(), out.view(1, n), _mmac_config()
    )


def try_mv_mmac(inp, vec, out, ordinary_with_output, *, tuning_key):
    if not _eligible(inp, vec, out):
        return False
    key = (
        inp.device,
        inp.dtype,
        vec.dtype,
        out.dtype,
        _mmac_config(),
        tuple(inp.shape),
        tuple(inp.stride()),
        tuple(vec.stride()),
        tuple(out.stride()),
        tuning_key,
        "mv-mmac-ordinary-tl-family-v3",
    )
    with torch_device_fn.device(inp.device):
        choice = _choices.get(key)
        if choice is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            with _choice_lock:
                choice = _choices.get(key)
                if choice is None:
                    from triton.flagtune.runtime.graph_benchmark import (
                        do_bench_hcu_graph,
                    )

                    # Resolve the unchanged Default/Expanded tuner first.
                    ordinary_with_output(out)
                    _launch_mmac(inp, vec, out)
                    samples = ([], [])
                    for repeat in range(3):
                        for index in (0, 1) if repeat % 2 == 0 else (1, 0):

                            def fresh_output(index=index):
                                output = torch.empty_like(out)
                                if index:
                                    _launch_mmac(inp, vec, output)
                                else:
                                    ordinary_with_output(output)
                                return output

                            samples[index].append(
                                float(
                                    do_bench_hcu_graph(
                                        fresh_output,
                                        rep=100,
                                        n_retries=10,
                                        device_interface=torch.cuda,
                                    )
                                )
                            )
                    medians = tuple(statistics.median(x) for x in samples)
                    choice = min(range(2), key=medians.__getitem__)
                    _choice_measurements[key] = {
                        "variants": ("ordinary", "mmac_k256_bn16_s2"),
                        "allocating_ms": medians,
                        "choice": choice,
                    }
                    _choices[key] = choice
                    logger.debug(
                        "Hygon MV family choice=%s allocating=%s ms", choice, medians
                    )
        if choice:
            _launch_mmac(inp, vec, out)
        else:
            ordinary_with_output(out)
    return True
