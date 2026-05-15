import math

import pytest
import torch

import flag_gems
from flag_gems.runtime.backend._nvidia.hopper.ops.fp8_einsum import fp8_einsum

from . import base

DEFAULT_BLOCK_SHAPE = [128, 128]


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()


try:
    import deep_gemm

    HAS_DEEPGEMM = hasattr(deep_gemm, "fp8_einsum")
except ImportError:
    HAS_DEEPGEMM = False


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 scales up to the nearest power-of-two (UE8M0 grid)."""
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool = True, gran_k: int = 128):
    assert x.dim() == 2
    m, n = x.shape
    padded_n = math.ceil(n / gran_k) * gran_k
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (
        (x_view * (1.0 / sf.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    return x_fp8, sf


def per_block_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool = True, gran_k: int = 128):
    assert x.dim() == 2
    m, n = x.shape
    padded_m = math.ceil(m / gran_k) * gran_k
    padded_n = math.ceil(n / gran_k) * gran_k
    x_padded = torch.zeros((padded_m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, gran_k, x_padded.size(1) // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return (
        x_scaled.view_as(x_padded)[:m, :n].contiguous(),
        sf.view(x_view.size(0), x_view.size(2)),
    )


def _make_fp8_einsum_inputs(b, h, r, d, block_shape, device, seed=0):
    """Build block-wise FP8 ``bhr,hdr->bhd`` inputs

    Returns (x_data, x_scale, y_data, y_scale):
      x_data:  (b, h, r) FP8           per-token scaled
      x_scale: (b, h, r // block_k) FP32
      y_data:  (h, d, r) FP8           per-block scaled
      y_scale: (h, d // block_n, r // block_k) FP32
    """
    block_n, block_k = block_shape
    torch.manual_seed(seed)
    x = torch.randn((b, h, r), device=device, dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device=device, dtype=torch.bfloat16)

    x_fp8 = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=True)
    x_data = x_fp8[0].view(b, h, r)
    x_scale = x_fp8[1].view(b, h, math.ceil(r / block_k))

    y_data = torch.empty_like(y, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty(
        (h, math.ceil(d / block_n), math.ceil(r / block_k)),
        device=device,
        dtype=torch.float32,
    )
    for i in range(h):
        y_data[i], y_scale[i] = per_block_cast_to_fp8(y[i], use_ue8m0=True)

    return x_data, x_scale, y_data, y_scale


class FP8EinsumBenchmark(base.Benchmark):
    """Benchmark for block-wise FP8 ``bhr,hdr->bhd`` einsum (FlagGems vs DeepGEMM)."""

    DEFAULT_METRICS = base.consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.block_shape = DEFAULT_BLOCK_SHAPE

    def set_shapes(self, shape_file_path=None):
        # (b, h, r, d)
        batches = (1, 4, 8, 16, 32, 64, 128, 4096, 8192, 16384, 32768)
        hrd_groups = {
            "flash": (8, 4096, 1024),
            "pro": (16, 7168, 1024),
        }
        self.shapes = [
            (b, h, r, d) for (h, r, d) in hrd_groups.values() for b in batches
        ]

    def get_input_iter(self, cur_dtype):
        del cur_dtype
        device = flag_gems.device
        for b, h, r, d in self.shapes:
            yield _make_fp8_einsum_inputs(b, h, r, d, self.block_shape, device)

    def get_tflops(self, op, *args, **kwargs):
        x_data, _, y_data, _ = args
        b, h, r = x_data.shape
        d = y_data.shape[1]
        return 2.0 * b * h * r * d


def _deepgemm_fp8_einsum_wrapper(x_data, x_scale, y_data, y_scale):
    """Reference: DeepGEMM block-wise FP8 ``fp8_einsum``."""
    b, h, _ = x_data.shape
    d = y_data.shape[1]
    z = torch.empty((b, h, d), device=x_data.device, dtype=torch.bfloat16)
    deep_gemm.fp8_einsum("bhr,hdr->bhd", (x_data, x_scale), (y_data, y_scale), z)
    return z


def _gems_fp8_einsum_wrapper(x_data, x_scale, y_data, y_scale):
    """FlagGems block-wise FP8 ``fp8_einsum`` (Gluon BMM kernel)."""
    return fp8_einsum(
        "bhr,hdr->bhd",
        x_data,
        x_scale,
        y_data,
        y_scale,
        block_size=DEFAULT_BLOCK_SHAPE,
    )


@pytest.mark.fp8_einsum
@pytest.mark.skipif(
    not (HAS_DEEPGEMM and CUDA_AVAILABLE),
    reason="requires DeepGEMM and NVIDIA Hopper architecture",
)
def test_perf_fp8_einsum_gems_vs_deepgemm():
    """Benchmark FlagGems vs DeepGEMM on block-wise FP8 ``bhr,hdr->bhd`` einsum."""
    bench = FP8EinsumBenchmark(
        op_name="fp8_einsum",
        torch_op=_deepgemm_fp8_einsum_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fp8_einsum_wrapper)
    bench.run()
