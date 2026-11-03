"""Measured matrix-instruction candidate for contiguous column-major BF16 MV.

The existing MV tuner keeps its complete configuration pool and protocol.
Only scalar family timings and indices are retained between public calls.
"""
import logging
import statistics
import threading

import torch
import triton

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)
_choices = {}
_choice_measurements = {}
_choice_lock = threading.RLock()


def _eligible(inp, vec, out):
    from .mm_packet import _loop_carried_refill_supported

    n, k = inp.shape
    if not (n > 1 and k >= 256 and k % 256 == 0
            and inp.dtype == vec.dtype == out.dtype == torch.bfloat16
            and inp.device == vec.device == out.device
            and inp.device.type == "cuda"
            and inp.stride() == (1, n) and vec.stride() == (1,)
            and out.stride() == (1,)
            and not out.requires_grad
            and (not out.is_inference() or torch.is_inference_mode_enabled())
            and out.untyped_storage().data_ptr()
            not in (inp.untyped_storage().data_ptr(), vec.untyped_storage().data_ptr())
            and all(t.data_ptr() % 16 == 0 for t in (inp, vec, out))
            and 2 * n * k < 2**31 and 2 * k < 2**31 and 2 * n < 2**31
            and _loop_carried_refill_supported()):
        return False
    with torch_device_fn.device(inp.device):
        target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx936"


def _launch_mmac(inp, vec, out):
    from .mm_splitu_nn_k256_kernel import _small_nn_splitu_k256

    n, k = inp.shape
    return _small_nn_splitu_k256[(1, triton.cdiv(n, 16), 1)](
        vec.view(1, k), inp.t(), out.view(1, n), 1, n, k, 16, 16,
        padded_slot_intervals=True, disable_llvm_loop_unroll=True,
        num_warps=4, num_stages=1, waves_per_eu=1,
        loop_unroll_stage="post-pipeline", wave_private_sync=True,
        loop_carried_store_refill=True,
        async_local_prefetch=False, mma_phase_schedule=False,
        packet_operand_overlap_diagnostic=False, packet_post_mma_wait=False,
        packet_wait_rotation=False, optimize_epilogue=False,
        kpack=2, mmac_transposed=False,
    )


def try_mv_mmac(inp, vec, out, ordinary_with_output, *, tuning_key):
    if not _eligible(inp, vec, out):
        return False
    key = (inp.device, tuple(inp.shape), tuple(inp.stride()), tuple(vec.stride()),
           tuple(out.stride()), tuning_key, "mv-mmac-allocating-family-v1")
    with torch_device_fn.device(inp.device):
        choice = _choices.get(key)
        if choice is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            with _choice_lock:
                choice = _choices.get(key)
                if choice is None:
                    from triton.flagtune.runtime.graph_benchmark import do_bench_hcu_graph

                    # Resolve the unchanged Default/Expanded tuner first.
                    ordinary_with_output(out)
                    _launch_mmac(inp, vec, out)
                    samples = ([], [])
                    for repeat in range(3):
                        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                            def fresh_output():
                                output = torch.empty_like(out)
                                if index:
                                    _launch_mmac(inp, vec, output)
                                else:
                                    ordinary_with_output(output)
                                return output

                            samples[index].append(float(do_bench_hcu_graph(
                                fresh_output, rep=100, n_retries=10,
                                device_interface=torch.cuda,
                            )))
                    medians = tuple(statistics.median(x) for x in samples)
                    choice = min(range(2), key=medians.__getitem__)
                    _choice_measurements[key] = dict(
                        variants=("ordinary", "mmac_k256_bn16"),
                        allocating_ms=medians, choice=choice,
                    )
                    _choices[key] = choice
                    logger.debug("Hygon MV family choice=%s allocating=%s ms", choice, medians)
        if choice:
            _launch_mmac(inp, vec, out)
        else:
            ordinary_with_output(out)
    return True
