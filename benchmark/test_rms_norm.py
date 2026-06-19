import pytest
import torch

from flag_gems.ops.rms_norm import rms_norm_fp8_w8a16

from . import base


@pytest.mark.rms_norm
def test_rms_norm():
    def rms_norm_input_fn(shape, dtype, device):
        _, N = shape
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)
        yield inp, (N,), weight

    bench = base.GenericBenchmark2DOnly(
        op_name="rms_norm",
        input_fn=rms_norm_input_fn,
        torch_op=torch.nn.functional.rms_norm,
    )
    bench.run()


GROUP_SIZE = 128
FP8_DTYPE = torch.float8_e4m3fn


def _quantize_fp8_grouped(w, group_size=GROUP_SIZE):
    fp8_info = torch.finfo(FP8_DTYPE)
    if w.ndim == 1:
        n = w.shape[0]
        assert n % group_size == 0
        wg = w.reshape(n // group_size, group_size).float()
        scale = (wg.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(min=1e-8)
        q = (wg / scale).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)
        return q.reshape(n).contiguous(), scale.squeeze(-1).to(w.dtype).contiguous()
    m, n = w.shape
    assert n % group_size == 0
    wg = w.reshape(m, n // group_size, group_size).float()
    scale = (wg.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(min=1e-8)
    q = (wg / scale).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)
    return q.reshape(m, n).contiguous(), scale.squeeze(-1).to(w.dtype).contiguous()


def _torch_rms_norm_w8a16(x, normalized_shape, weight_fp8, weight_scale, weight_ref):
    return torch.nn.functional.rms_norm(x, normalized_shape, weight_ref)


def _gems_rms_norm_w8a16(x, normalized_shape, weight_fp8, weight_scale, weight_ref):
    return rms_norm_fp8_w8a16(x, normalized_shape, weight_fp8, weight_scale)




class RmsNormFp8Benchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 4096),
            (16, 4096),
            (64, 4096),
            (256, 4096),
            (1024, 4096),
            (1, 8192),
            (64, 8192),
            (256, 8192),
            (1, 16384),
            (64, 16384),
        ]


class RmsNormFp8W8A16Benchmark(RmsNormFp8Benchmark):
    def get_input_iter(self, dtype):
        for shape in self.shapes:
            _, n = shape
            x = torch.randn(shape, dtype=dtype, device=self.device)
            weight = torch.randn(n, dtype=dtype, device=self.device)
            weight_fp8, weight_scale = _quantize_fp8_grouped(weight)
            yield x, (n,), weight_fp8, weight_scale, weight



@pytest.mark.rms_norm
def test_rms_norm_fp8_w8a16():
    bench = RmsNormFp8W8A16Benchmark(
        op_name="rms_norm_fp8_w8a16",
        torch_op=_torch_rms_norm_w8a16,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_rms_norm_w8a16)
    bench.run()
