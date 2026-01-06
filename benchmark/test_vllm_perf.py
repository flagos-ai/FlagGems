import random
from itertools import product
from math import ceil

import pytest
import torch

import flag_gems
from benchmark.performance_utils import Benchmark

random.seed(42)


def is_vllm_available():
    try:
        import vllm._custom_ops as ops  # noqa: F401

        return True
    except ImportError:
        return False


VLLM_AVAILABLE = is_vllm_available()


def is_hopper_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


HOPPER_AVAILABLE = is_hopper_available()


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


class CutlassScaledMMPerfKit:
    num_perf_cases = 4
    scalar_only_params = []
    vector_only_params = []
    scalar_and_vector_params = []
    block_params = []

    @staticmethod
    def _get_all_combinations():
        # these shapes come from the test file of op `cutlass_scaled_mm` of vLLM
        mnk = [
            (1, 256, 128),
            (1, 16384, 1024),
            (1, 24576, 496),
            (16, 256, 496),
            (16, 16384, 128),
            (16, 24576, 4096),
            (32, 8192, 4096),
            (32, 16384, 4096),
            (33, 1024, 1024),
            (33, 8192, 128),
            (64, 2048, 496),
            (64, 16384, 1024),
            (100, 8192, 496),
            (128, 32768, 4096),
            (256, 4096, 4096),
            (512, 256, 1024),
            (512, 8192, 4096),
            (512, 16384, 128),
            (512, 24576, 128),
        ]
        scale_shape_types = ["scalar", "vector", "matrix"]
        if_use_bias = [True, False]
        dtypes = [(torch.int8, torch.float16), (torch.float8_e4m3fn, torch.bfloat16)]

        combinations = product(
            mnk, scale_shape_types, scale_shape_types, if_use_bias, dtypes
        )
        return combinations

    @classmethod
    def _rand_sample(cls, all_params):
        random.shuffle(all_params)
        count = [0] * 4
        for param in all_params:
            a_scale_category = param["a_scale_category"]
            b_scale_category = param["b_scale_category"]
            if a_scale_category == "matrix" and count[0] < cls.num_perf_cases:
                count[0] += 1
                cls.block_params.append(param)
            elif (
                a_scale_category == "scalar"
                and b_scale_category == "scalar"
                and count[1] < cls.num_perf_cases
            ):
                count[1] += 1
                cls.scalar_only_params.append(param)
            elif (
                a_scale_category == "vector"
                and b_scale_category == "vector"
                and count[2] < cls.num_perf_cases
            ):
                count[2] += 1
                cls.vector_only_params.append(param)
            elif count[3] < cls.num_perf_cases:
                count[3] += 1
                cls.scalar_and_vector_params.append(param)
            else:
                continue

    @classmethod
    def init_perf_params(cls):
        combinations = cls._get_all_combinations()

        all_params = []
        for (
            (M, N, K),
            a_scale_category,
            b_scale_category,
            use_bias,
            (in_dtype, out_dtype),
        ) in combinations:
            is_scalar_or_vector_dequant = a_scale_category in [
                "scalar",
                "vector",
            ] and b_scale_category in ["scalar", "vector"]
            is_block_dequant = (
                a_scale_category == "matrix" and b_scale_category == "matrix"
            )

            if not (is_scalar_or_vector_dequant or is_block_dequant):
                continue

            if is_block_dequant and (use_bias or M % 4 != 0):
                continue

            param = {
                "M": M,
                "N": N,
                "K": K,
                "a_scale_category": a_scale_category,
                "b_scale_category": b_scale_category,
                "use_bias": use_bias,
                "in_dtype": in_dtype,
                "out_dtype": out_dtype,
            }
            all_params.append(param)

        cls._rand_sample(all_params)

    @staticmethod
    def get_scale_shape(M, N, K, category, is_a_scale=True):
        if category == "scalar":
            return (1,)
        elif category == "vector":
            if is_a_scale:
                return (M,)
            else:
                return (N,)
        else:
            if is_a_scale:
                return (M, ceil(K / 128))
            else:
                return (ceil(K / 128), ceil(N / 128))


class CutlassScaledMMBenchmark(Benchmark):
    def __init__(self):
        extended_dtypes = ["scalar_only", "vector_only", "scalar_and_vector", "block"]
        super().__init__(
            "cutlass_scaled_mm", torch.ops._C.cutlass_scaled_mm, extended_dtypes
        )
        self.set_gems(flag_gems.cutlass_scaled_mm)
        self.kit = CutlassScaledMMPerfKit
        self.kit.init_perf_params()

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        params = getattr(self.kit, f"{dtype}_params")

        for p in params:
            M, N, K = p["M"], p["N"], p["K"]
            in_dtype = p["in_dtype"]
            out_dtype = p["out_dtype"]
            a_scale_category = p["a_scale_category"]
            b_scale_category = p["b_scale_category"]

            if in_dtype == torch.int8:
                a = to_int8(torch.randn((M, K), device=flag_gems.device))
                b = to_int8(
                    torch.randn((K, N), device=flag_gems.device).t().contiguous().t()
                    * 5
                )
            else:
                a = to_fp8(torch.randn((M, K), device=flag_gems.device))
                b = to_fp8(
                    torch.randn((K, N), device=flag_gems.device).t().contiguous().t()
                )

            a_scale_shape = self.kit.get_scale_shape(M, N, K, a_scale_category)
            b_scale_shape = self.kit.get_scale_shape(M, N, K, b_scale_category, False)

            scale_a = torch.randn(
                a_scale_shape, device=flag_gems.device, dtype=torch.float32
            )
            scale_b = torch.randn(
                b_scale_shape, device=flag_gems.device, dtype=torch.float32
            )

            scale_a = scale_a.contiguous()
            # convert scale_b to col-major
            # (for scalar/vector scale_b, this's a identical transformation)
            scale_b = scale_b.t().contiguous().t()

            bias = None
            if p["use_bias"]:
                bias = torch.randn((N,), device=flag_gems.device, dtype=out_dtype)

            c = torch.empty((M, N), device=flag_gems.device, dtype=out_dtype)

            yield (c, a, b, scale_a, scale_b, bias)


@pytest.mark.skipif(
    not (VLLM_AVAILABLE and HOPPER_AVAILABLE),
    reason="requires vLLM and NVIDIA Hopper architecture",
)
@pytest.mark.cutlass_scaled_mm
@pytest.mark.performance
def test_cutlass_scaled_mm_benchmark():
    bench = CutlassScaledMMBenchmark()
    bench.run()
