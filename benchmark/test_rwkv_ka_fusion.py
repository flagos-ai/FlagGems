import pytest
import torch

import flag_gems

from . import base, consts


class RWKVBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def rwkv_ka_fusion_input_fn(shape, dtype, device):
    T = shape[0]
    H = 8
    N = 64
    C = H * N

    k = torch.randn(T, C, dtype=dtype, device=device)
    kk = torch.randn(C, dtype=dtype, device=device)
    a = torch.randn(T, C, dtype=dtype, device=device)
    ka = torch.randn(C, dtype=dtype, device=device)

    yield k, kk, a, ka, H, N


def torch_rwkv_ka(k, kk, a, ka, H, N):
    T, C = k.shape
    assert C == H * N and kk.shape == (C,) and a.shape == (T, C) and ka.shape == (C,)
    o_kk = torch.nn.functional.normalize((k * kk).view(T, H, N), dim=-1, p=2.0).view(
        T, H * N
    )
    o_k = k * (1 + (a - 1) * ka)
    o_kka = o_kk * a

    return o_k, o_kk, o_kka


@pytest.mark.rwkv_ka_fusion
def test_rwkv_ka_fusion():
    bench = RWKVBenchmark(
        input_fn=rwkv_ka_fusion_input_fn,
        op_name="rwkv_ka_fusion",
        torch_op=torch_rwkv_ka,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.rwkv_ka_fusion)

    bench.run()
