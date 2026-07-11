"""Ascend index_fill benchmark with a direct ACLNN reference.

The reference intentionally covers only valid contiguous scalar floating-point cases.
Torch Eager is used for correctness only because torch_npu materializes NPU index
elements on the host before dispatching its index_fill implementation.
"""

import gc
import os
import statistics
import time

import flag_gems
import pytest
import torch

from . import base, consts
from .ascend_index_fill_reference import index_fill as aclnn_index_fill
from .ascend_index_fill_reference import index_fill_ as aclnn_index_fill_
from .test_index_fill import INDEX_FILL_DTYPES, _base_inputs, _scalar_value


_SAMPLES_ENV = "FLAGGEMS_INDEX_FILL_REFERENCE_SAMPLES"
_DEFAULT_SAMPLES = 5
_CORE_SHAPES = [
    (65536,),
    (4096, 256),
    (4096, 4096),
]
_COMPREHENSIVE_SHAPES = [
    (8192, 4096),
    (200, 40999, 3),
]


pytestmark = pytest.mark.skipif(
    base.vendor_name != "ascend" or base.device != "npu",
    reason="The direct ACLNN reference is only available on Ascend NPUs.",
)


def _sample_count():
    value = os.environ.get(_SAMPLES_ENV, str(_DEFAULT_SAMPLES))
    try:
        samples = int(value)
    except ValueError as error:
        raise ValueError(f"{_SAMPLES_ENV} must be a positive integer") from error
    if samples < 1:
        raise ValueError(f"{_SAMPLES_ENV} must be a positive integer")
    return samples


def _shapes():
    shapes = list(_CORE_SHAPES)
    if (
        base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
        and not base.Config.query
    ):
        shapes.extend(_COMPREHENSIVE_SHAPES)
    return shapes


def _call_args(args, is_inplace):
    if is_inplace:
        return (args[0].clone(), *args[1:])
    return args


def _invoke(op, args, is_inplace):
    return op(*_call_args(args, is_inplace))


def _prewarm(op, args, is_inplace):
    for _ in range(2):
        _invoke(op, args, is_inplace)
    base.torch_device_fn.synchronize()


def _p50_latency(op, args, is_inplace, samples):
    latencies = []
    for _ in range(samples):
        call_args = _call_args(args, is_inplace)
        base.torch_device_fn.synchronize()
        start = time.perf_counter()
        result = op(*call_args)
        base.torch_device_fn.synchronize()
        latencies.append((time.perf_counter() - start) * 1e3)
        del result
    return statistics.median(latencies)


def _assert_correct(torch_op, aclnn_op, args, is_inplace):
    expected = _invoke(torch_op, args, is_inplace)
    base.torch_device_fn.synchronize()

    reference = _invoke(aclnn_op, args, is_inplace)
    base.torch_device_fn.synchronize()
    torch.testing.assert_close(reference, expected)

    with flag_gems.use_gems(exclude=["zero_"]):
        actual = _invoke(torch_op, args, is_inplace)
    base.torch_device_fn.synchronize()
    torch.testing.assert_close(actual, expected)


def _print_header(op_name, samples):
    print(
        f"\n{op_name}: synchronized P50 of {samples} independent calls "
        "(direct ACLNN / FlagGems)"
    )
    print(
        f"{'dtype':<14} {'shape':<22} {'dim':>4} {'index':>8} "
        f"{'ACLNN ms':>12} {'Gems ms':>12} {'ACLNN/Gems':>12}"
    )


def _run(op_name, torch_op, aclnn_op, is_inplace):
    if base.Config.mode != consts.BenchMode.OPERATOR:
        pytest.skip("Direct ACLNN comparison requires --mode=operator")

    samples = _sample_count()
    dtypes = base.Config.user_desired_dtypes or INDEX_FILL_DTYPES
    unsupported_dtypes = [dtype for dtype in dtypes if dtype not in INDEX_FILL_DTYPES]
    if unsupported_dtypes:
        raise ValueError(
            "Direct ACLNN index_fill reference supports only "
            f"{INDEX_FILL_DTYPES}, got {unsupported_dtypes}"
        )
    _print_header(op_name, samples)

    for dtype in dtypes:
        for shape in _shapes():
            for inp, dim, index in _base_inputs(shape, dtype, base.device):
                args = (inp, dim, index, _scalar_value(dtype))
                _assert_correct(torch_op, aclnn_op, args, is_inplace)

                _prewarm(aclnn_op, args, is_inplace)
                aclnn_latency = _p50_latency(aclnn_op, args, is_inplace, samples)

                with flag_gems.use_gems(exclude=["zero_"]):
                    _prewarm(torch_op, args, is_inplace)
                    gems_latency = _p50_latency(
                        torch_op, args, is_inplace, samples
                    )

                print(
                    f"{str(dtype):<14} {str(tuple(shape)):<22} {dim:>4} "
                    f"{index.numel():>8} {aclnn_latency:>12.6f} "
                    f"{gems_latency:>12.6f} "
                    f"{aclnn_latency / gems_latency:>12.3f}"
                )
                del inp, index
                gc.collect()


@pytest.mark.index_fill
def test_index_fill_npu_reference():
    _run(
        op_name="index_fill",
        torch_op=torch.index_fill,
        aclnn_op=aclnn_index_fill,
        is_inplace=False,
    )


@pytest.mark.index_fill_
def test_index_fill_npu_reference_inplace():
    _run(
        op_name="index_fill_",
        torch_op=torch.Tensor.index_fill_,
        aclnn_op=aclnn_index_fill_,
        is_inplace=True,
    )
