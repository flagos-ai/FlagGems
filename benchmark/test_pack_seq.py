import pytest
import torch

import flag_gems
from flag_gems.fused import pack_seq_triton

from . import base

# =============================================================================
# vLLM availability check
# =============================================================================

try:
    from vllm.v1.attention.ops.common import pack_seq_triton as vllm_pack_seq

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


# =============================================================================
# CUDA / Hopper check for FP8
# =============================================================================


def _is_cuda_hopper():
    if flag_gems.device != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 9


CUDA_HOPPER = _is_cuda_hopper()

FP8_DTYPE = torch.float8_e4m3fn if CUDA_HOPPER else None


# =============================================================================
# Wrapper functions — vllm
# =============================================================================


def _vllm_pack_seq(x, lengths):
    return vllm_pack_seq(x, lengths)


def _vllm_pack_seq_fp8(x, lengths):
    return vllm_pack_seq(x, lengths)


# =============================================================================
# Wrapper functions — FlagGems
# =============================================================================


def _gems_pack_seq(x, lengths):
    return pack_seq_triton(x, lengths)


def _gems_pack_seq_fp8(x, lengths):
    return pack_seq_triton(x, lengths)


# =============================================================================
# Benchmark shapes: (N, D, B, lengths_list)
# =============================================================================

PACK_BENCH_SHAPES = [
    (512, 64, 5, [64, 128, 64, 128, 128]),
    (4096, 128, 4, [1024, 1024, 1024, 1024]),
    (8192, 256, 5, [1024, 2048, 1024, 2048, 2048]),
    (2048, 512, 4, [512, 512, 512, 512]),
    (16384, 64, 8, [2048] * 8),
    (1024, 1024, 4, [256] * 4),
]

FP8_BENCH_SHAPES = [
    (512, 64, 5, [64, 128, 64, 128, 128]),
    (4096, 128, 4, [1024, 1024, 1024, 1024]),
    (2048, 512, 4, [512, 512, 512, 512]),
]


# =============================================================================
# Custom Benchmark class — pack_seq (float dtypes)
# =============================================================================


class PackSeqBenchmark(base.Benchmark):
    DEFAULT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = PACK_BENCH_SHAPES

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._pack_input_fn(config, cur_dtype)

    def _pack_input_fn(self, config, dtype):
        N, D, B, lengths_list = config
        device = flag_gems.device
        lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
        x = torch.randn(N, D, dtype=dtype, device=device)
        yield x, lengths


# =============================================================================
# Custom Benchmark class — pack_seq (FP8)
# =============================================================================


class PackSeqFP8Benchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = FP8_BENCH_SHAPES

    def get_input_iter(self, cur_dtype):
        del cur_dtype
        for config in self.shapes:
            yield from self._fp8_input_fn(config)

    def _fp8_input_fn(self, config):
        N, D, B, lengths_list = config
        device = flag_gems.device
        lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
        x = torch.randn(N, D, dtype=torch.float32, device=device) * 0.1
        x_fp8 = x.to(FP8_DTYPE)
        yield x_fp8, lengths


# =============================================================================
# Benchmark tests
# =============================================================================


@pytest.mark.pack_seq
@pytest.mark.skipif(
    not HAS_VLLM,
    reason="requires vLLM to be installed for reference comparison",
)
def test_pack_seq():
    bench = PackSeqBenchmark(
        op_name="pack_seq",
        torch_op=_vllm_pack_seq,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.set_gems(_gems_pack_seq)
    bench.run()


@pytest.mark.pack_seq
@pytest.mark.skipif(
    not (HAS_VLLM and CUDA_HOPPER),
    reason="requires vLLM and NVIDIA Hopper architecture for FP8",
)
def test_pack_seq_fp8():
    bench = PackSeqFP8Benchmark(
        op_name="pack_seq_fp8",
        torch_op=_vllm_pack_seq_fp8,
        dtypes=[FP8_DTYPE],
    )
    bench.set_gems(_gems_pack_seq_fp8)
    bench.run()
