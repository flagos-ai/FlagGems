import torch

RESOLUTION = {
    torch.bool: 0,
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


def assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    assert res.dtype == dtype
    ref = ref.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol, equal_nan=equal_nan)


def assert_equal(res, ref, equal_nan=False):
    torch.testing.assert_close(res, ref, atol=0, rtol=0, equal_nan=equal_nan)
