"""
Competition Performance Benchmark Tests

Provides performance tests for the 20 competition operators defined in tasks.yaml.
Each test_perf_<op> function corresponds to a benchmark_tests entry in tasks.yaml.
"""

from typing import Generator

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input

device = flag_gems.device

# ============================================================
# 1. log10 — unary pointwise op
# ============================================================


class Log10Benchmark(Benchmark):
    def set_more_shapes(self):
        return [(1024, 2**i) for i in range(0, 20, 4)] + [
            (64, 64, 2**i) for i in range(0, 15, 4)
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = torch.rand(shape, dtype=cur_dtype, device=self.device) + 0.01
            yield inp,


@pytest.mark.competition
def test_perf_log10():
    bench = Log10Benchmark(
        op_name="log10",
        torch_op=torch.log10,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 2. logaddexp — binary pointwise op
# ============================================================


class LogaddexpBenchmark(Benchmark):
    def set_more_shapes(self):
        return [(1024, 2**i) for i in range(0, 20, 4)] + [
            (64, 64, 2**i) for i in range(0, 15, 4)
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            inp2 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp1, inp2


@pytest.mark.competition
def test_perf_logaddexp():
    bench = LogaddexpBenchmark(
        op_name="logaddexp",
        torch_op=torch.logaddexp,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 3. cosh — unary pointwise op
# ============================================================


class CoshBenchmark(Benchmark):
    def set_more_shapes(self):
        return [(1024, 2**i) for i in range(0, 20, 4)] + [
            (64, 64, 2**i) for i in range(0, 15, 4)
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp,


@pytest.mark.competition
def test_perf_cosh():
    bench = CoshBenchmark(
        op_name="cosh",
        torch_op=torch.cosh,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 4. gcd — binary integer op
# ============================================================


class GcdBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.int32]

    def set_more_shapes(self):
        return [(1024, 2**i) for i in range(0, 20, 4)]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = torch.randint(1, 1000, shape, dtype=cur_dtype, device="cpu").to(
                self.device
            )
            inp2 = torch.randint(1, 1000, shape, dtype=cur_dtype, device="cpu").to(
                self.device
            )
            yield inp1, inp2


@pytest.mark.competition
def test_perf_gcd():
    bench = GcdBenchmark(
        op_name="gcd",
        torch_op=torch.gcd,
        dtypes=[torch.int32],
    )
    bench.run()


# ============================================================
# 5. tril — lower triangular matrix
# ============================================================


class TrilBenchmark(Benchmark):
    def set_more_shapes(self):
        return [(256, 256), (1024, 1024), (4096, 4096)]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp,


@pytest.mark.competition
def test_perf_tril():
    bench = TrilBenchmark(
        op_name="tril",
        torch_op=torch.tril,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 6. roll — tensor roll
# ============================================================


class RollBenchmark(Benchmark):
    def set_more_shapes(self):
        return [(1024, 2**i) for i in range(0, 20, 4)] + [
            (64, 64, 2**i) for i in range(0, 15, 4)
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            shifts = tuple(s // 4 for s in shape)
            dims = tuple(range(len(shape)))
            yield inp, {"shifts": shifts, "dims": dims}


@pytest.mark.competition
def test_perf_roll():
    bench = RollBenchmark(
        op_name="roll",
        torch_op=torch.roll,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 7. leaky_relu — activation function
# ============================================================


class LeakyReluBenchmark(Benchmark):
    def set_more_shapes(self):
        return [(1024, 2**i) for i in range(0, 20, 4)] + [
            (64, 64, 2**i) for i in range(0, 15, 4)
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp, {"negative_slope": 0.01}


@pytest.mark.competition
def test_perf_leaky_relu():
    bench = LeakyReluBenchmark(
        op_name="leaky_relu",
        torch_op=torch.nn.functional.leaky_relu,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 8. asinh — unary pointwise op
# ============================================================


class AsinhBenchmark(Benchmark):
    def set_more_shapes(self):
        return [(1024, 2**i) for i in range(0, 20, 4)] + [
            (64, 64, 2**i) for i in range(0, 15, 4)
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp,


@pytest.mark.competition
def test_perf_asinh():
    bench = AsinhBenchmark(
        op_name="asinh",
        torch_op=torch.asinh,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 9. upsample_nearest2d — upsampling
# ============================================================


class UpsampleNearest2dCompBenchmark(Benchmark):
    def set_more_shapes(self):
        return None

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 3, 512, 512),
            (8, 16, 128, 128),
            (2, 3, 1024, 1024),
            (16, 16, 512, 512),
            (16, 16, 1024, 1024),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            batch, channel, height, width = shape
            inp = torch.randn(size=shape, device=self.device, dtype=cur_dtype)
            output_size = (height * 2, width * 2)
            yield {
                "input": inp,
                "output_size": output_size,
                "scales_h": None,
                "scales_w": None,
            },


@pytest.mark.competition
def test_perf_upsample_nearest2d():
    bench = UpsampleNearest2dCompBenchmark(
        op_name="upsample_nearest2d",
        torch_op=torch._C._nn.upsample_nearest2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 10. scatter_reduce — scatter reduction
# ============================================================


class ScatterReduceBenchmark(Benchmark):
    def set_more_shapes(self):
        return None

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1024, 1024),
            (4096, 4096),
            (256, 65536),
            (64, 512, 512),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            src = torch.randn(shape, dtype=cur_dtype, device=self.device)
            index = torch.randint(0, shape[-1], shape, device=self.device)
            inp = torch.zeros(shape, dtype=cur_dtype, device=self.device)
            yield inp, -1, index, src, {"reduce": "sum"}


@pytest.mark.competition
def test_perf_scatter_reduce():
    bench = ScatterReduceBenchmark(
        op_name="scatter_reduce",
        torch_op=torch.Tensor.scatter_reduce,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 11. median — reduction op
# ============================================================


class MedianBenchmark(Benchmark):
    def set_more_shapes(self):
        return [(1024, 2**i) for i in range(0, 20, 4)]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp.ndim > 1:
                yield inp, -1
            else:
                yield inp,


@pytest.mark.competition
def test_perf_median():
    bench = MedianBenchmark(
        op_name="median",
        torch_op=torch.median,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 12. smooth_l1_loss — loss function
# ============================================================


class SmoothL1LossBenchmark(Benchmark):
    def set_more_shapes(self):
        return [(1024, 2**i) for i in range(0, 20, 4)]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            target = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp, target


@pytest.mark.competition
def test_perf_smooth_l1_loss():
    bench = SmoothL1LossBenchmark(
        op_name="smooth_l1_loss",
        torch_op=torch.nn.functional.smooth_l1_loss,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 13. pixel_shuffle — pixel shuffle
# ============================================================


class PixelShuffleBenchmark(Benchmark):
    def set_more_shapes(self):
        return None

    def set_shapes(self, shape_file_path=None):
        # (N, C*r^2, H, W) with upscale_factor=r
        self.shapes = [
            (1, 64, 128, 128, 2),  # C=16 after shuffle
            (4, 36, 64, 64, 3),  # C=4 after shuffle
            (8, 16, 256, 256, 2),  # C=4 after shuffle
            (2, 144, 32, 32, 3),  # C=16 after shuffle
            (16, 64, 64, 64, 4),  # C=1 after shuffle
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            n, c, h, w, r = shape
            inp = torch.randn((n, c, h, w), dtype=cur_dtype, device=self.device)
            yield inp, r


@pytest.mark.competition
def test_perf_pixel_shuffle():
    bench = PixelShuffleBenchmark(
        op_name="pixel_shuffle",
        torch_op=torch.nn.functional.pixel_shuffle,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 14. conv_transpose2d — transposed convolution
# ============================================================


class ConvTranspose2dBenchmark(Benchmark):
    def set_more_shapes(self):
        return None

    def set_shapes(self, shape_file_path=None):
        # (N, C_in, H, W, C_out, K_h, K_w, stride, padding, groups)
        self.shapes = [
            (32, 64, 16, 16, 32, 3, 3, 1, 1, 1),
            (16, 128, 8, 8, 64, 4, 4, 2, 1, 1),
            (8, 256, 4, 4, 128, 3, 3, 2, 1, 1),
            (4, 512, 2, 2, 256, 4, 4, 2, 1, 1),
            (32, 32, 32, 32, 16, 3, 3, 1, 1, 2),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            n, c_in, h, w, c_out, kh, kw, stride, padding, groups = shape
            inp = torch.randn((n, c_in, h, w), dtype=cur_dtype, device=self.device)
            weight = torch.randn(
                (c_in, c_out // groups, kh, kw), dtype=cur_dtype, device=self.device
            )
            yield {
                "input": inp,
                "weight": weight,
                "bias": None,
                "stride": stride,
                "padding": padding,
                "groups": groups,
            },


@pytest.mark.competition
def test_perf_conv_transpose2d():
    bench = ConvTranspose2dBenchmark(
        op_name="conv_transpose2d",
        torch_op=torch.nn.functional.conv_transpose2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 15. avg_pool3d — 3D average pooling
# ============================================================


class AvgPool3dBenchmark(Benchmark):
    def set_more_shapes(self):
        return None

    def set_shapes(self, shape_file_path=None):
        # (N, C, D, H, W)
        self.shapes = [
            (4, 16, 16, 16, 16),
            (8, 32, 8, 32, 32),
            (2, 64, 16, 64, 64),
            (16, 8, 32, 32, 32),
            (4, 128, 8, 16, 16),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp, {"kernel_size": 3, "stride": 2, "padding": 1}


@pytest.mark.competition
def test_perf_avg_pool3d():
    bench = AvgPool3dBenchmark(
        op_name="avg_pool3d",
        torch_op=torch.nn.functional.avg_pool3d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 16. max_pool3d — 3D max pooling
# ============================================================


class MaxPool3dBenchmark(Benchmark):
    def set_more_shapes(self):
        return None

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4, 16, 16, 16, 16),
            (8, 32, 8, 32, 32),
            (2, 64, 16, 64, 64),
            (16, 8, 32, 32, 32),
            (4, 128, 8, 16, 16),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp, {"kernel_size": 3, "stride": 2, "padding": 1}


@pytest.mark.competition
def test_perf_max_pool3d():
    bench = MaxPool3dBenchmark(
        op_name="max_pool3d",
        torch_op=torch.nn.functional.max_pool3d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# 17. chunk_gated_delta_rule — FLA op
# ============================================================


class ChunkGatedDeltaRuleBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]

    def set_more_shapes(self):
        return None

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1,),
            (64,),
            (128,),
            (256,),
            (512,),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for (T,) in self.shapes:
            yield self._build_inputs(T, cur_dtype)

    def _build_inputs(self, T, dtype):
        B = 1
        H, HV, K, V = 16, 32, 128, 128
        tp_size = 4
        key_dim = H * K
        value_dim = HV * V

        mixed_qkv_dim = (2 * key_dim + value_dim) // tp_size
        total_tokens = B * T
        mixed_qkv = torch.randn(
            (total_tokens, mixed_qkv_dim), device=self.device, dtype=dtype
        )

        q, k, v = torch.split(
            mixed_qkv,
            [key_dim // tp_size, key_dim // tp_size, value_dim // tp_size],
            dim=-1,
        )
        q = q.view(1, q.shape[0], -1, K).contiguous()
        k = k.view(1, k.shape[0], -1, K).contiguous()
        v = v.view(1, v.shape[0], -1, V).contiguous()

        HV_local = v.shape[2]
        g = torch.nn.functional.logsigmoid(
            torch.randn((B, T, HV_local), device=self.device, dtype=dtype)
        )
        beta = torch.rand(B, T, HV_local, device=self.device, dtype=dtype).sigmoid()
        cu_seqlens = torch.arange(T + 1, device=self.device, dtype=torch.long)
        initial_state = torch.zeros(
            (1024, HV_local, K, V), device=self.device, dtype=dtype
        )
        ssm_state_indices = torch.zeros(T, device=self.device, dtype=torch.long)
        scale = 0.08838834764831845

        return (
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            True,
            cu_seqlens,
            ssm_state_indices,
            None,
            True,
        )


@pytest.mark.competition
@pytest.mark.skipif(flag_gems.device != "cuda", reason="requires CUDA")
def test_perf_chunk_gated_delta_rule():
    try:
        torch_op = flag_gems.fused_recurrent_gated_delta_rule_fwd
    except AttributeError:
        pytest.skip("fused_recurrent_gated_delta_rule_fwd not available")

    bench = ChunkGatedDeltaRuleBenchmark(
        op_name="chunk_gated_delta_rule",
        torch_op=torch_op,
    )
    bench.run()


# ============================================================
# 18. svd — singular value decomposition
# ============================================================


class SvdBenchmark(Benchmark):
    def set_more_shapes(self):
        return None

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (32, 32),
            (128, 128),
            (256, 256),
            (512, 512),
            (64, 256),
            (256, 64),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp,


@pytest.mark.competition
def test_perf_svd():
    bench = SvdBenchmark(
        op_name="svd",
        torch_op=torch.linalg.svd,
        dtypes=[torch.float32],
    )
    bench.run()


# ============================================================
# 19. ctc_loss — CTC loss
# ============================================================


class CtcLossBenchmark(Benchmark):
    def set_more_shapes(self):
        return None

    def set_shapes(self, shape_file_path=None):
        # (T, N, C) - time, batch, classes
        self.shapes = [
            (50, 16, 20),
            (100, 32, 28),
            (150, 64, 50),
            (200, 16, 100),
            (300, 8, 500),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            T, N, C = shape
            log_probs = torch.randn(
                T, N, C, dtype=cur_dtype, device=self.device
            ).log_softmax(2)
            targets = torch.randint(
                1, C, (N, T // 2), dtype=torch.long, device=self.device
            )
            input_lengths = torch.full((N,), T, dtype=torch.long, device=self.device)
            target_lengths = torch.full(
                (N,), T // 2, dtype=torch.long, device=self.device
            )
            yield log_probs, targets, input_lengths, target_lengths


@pytest.mark.competition
def test_perf_ctc_loss():
    bench = CtcLossBenchmark(
        op_name="ctc_loss",
        torch_op=torch.nn.functional.ctc_loss,
        dtypes=[torch.float32],
    )
    bench.run()


# ============================================================
# 20. grid_sample — grid sampling
# ============================================================


class GridSampleBenchmark(Benchmark):
    def set_more_shapes(self):
        return None

    def set_shapes(self, shape_file_path=None):
        # (N, C, H_in, W_in, H_out, W_out)
        self.shapes = [
            (4, 3, 64, 64, 128, 128),
            (8, 16, 32, 32, 64, 64),
            (2, 64, 128, 128, 256, 256),
            (16, 3, 256, 256, 128, 128),
            (4, 32, 64, 64, 64, 64),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            n, c, h_in, w_in, h_out, w_out = shape
            inp = torch.randn((n, c, h_in, w_in), dtype=cur_dtype, device=self.device)
            grid = (
                torch.rand((n, h_out, w_out, 2), dtype=cur_dtype, device=self.device)
                * 2
                - 1
            )
            yield inp, grid


@pytest.mark.competition
def test_perf_grid_sample():
    bench = GridSampleBenchmark(
        op_name="grid_sample",
        torch_op=torch.nn.functional.grid_sample,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
