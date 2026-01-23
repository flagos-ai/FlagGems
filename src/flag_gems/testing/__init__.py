import torch

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

if runtime.device.vendor_name == "kunlunxin":
    RESOLUTION = {
        torch.bool: 0,
        torch.uint8: 0,
        torch.int8: 0,
        torch.int16: 0,
        torch.int32: 0,
        torch.int64: 0,
        torch.float16: 1e-3,
        torch.float32: 1.3e-6,
        torch.bfloat16: 0.016,
        torch.float64: 1e-7,
        torch.complex32: 1e-3,
        torch.complex64: 1.3e-6,
    }
else:
    RESOLUTION = {
        torch.bool: 0,
        torch.uint8: 0,
        torch.int8: 0,
        torch.int16: 0,
        torch.int32: 0,
        torch.int64: 0,
        torch.float8_e4m3fn: 1e-3,
        torch.float8_e5m2: 1e-3,
        torch.float8_e4m3fnuz: 1e-3,
        torch.float8_e5m2fnuz: 1e-3,
        torch.float16: 1e-3,
        torch.float32: 1.3e-6,
        torch.bfloat16: 0.016,
        torch.float64: 1e-7,
        torch.complex32: 1e-3,
        torch.complex64: 1.3e-6,
    }


def _maybe_move_to_cpu(res, ref):
    if not (res.is_cuda and ref.is_cuda):
        return res, ref

    required = res.numel() * res.element_size()

    free_mem = None
    try:
        free_mem, _ = torch_device_fn.mem_get_info(res.device)
    except RuntimeError:
        pass

    # torch.isclose allocates an auxiliary tensor roughly the size of the inputs,
    # so ensure we have enough headroom; otherwise compare on CPU.
    HUGE_TENSOR_BYTES = 1 << 30  # 1 GiB
    if (free_mem is not None and required >= free_mem) or (
        required >= HUGE_TENSOR_BYTES
    ):
        return res.cpu(), ref.cpu()
    return res, ref


def assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1, atol=1e-4):
    if dtype is None:
        dtype = torch.float32
    assert res.dtype == dtype
    ref = ref.to(dtype)
    res, ref = _maybe_move_to_cpu(res, ref)
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(
        res, ref, atol=atol * reduce_dim, rtol=rtol, equal_nan=equal_nan
    )


def assert_equal(res, ref, equal_nan=False):
    torch.testing.assert_close(res, ref, atol=0, rtol=0, equal_nan=equal_nan)
