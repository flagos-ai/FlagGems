import os
from typing import Generator

import pytest
import torch

import flag_gems
from benchmark.attri_util import (
    COMPLEX_DTYPES,
    DEFAULT_METRICS,
    FLOAT_DTYPES,
    BenchLevel,
    model_shapes,
)
from benchmark.conftest import Config
from benchmark.performance_utils import Benchmark, GenericBenchmark2DOnly

try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm as vllm_w8a8_triton_block_scaled_mm,
    )

    VLLM_W8A8_BLOCK_FP8_AVAILABLE = True
except Exception:
    vllm_w8a8_triton_block_scaled_mm = None
    VLLM_W8A8_BLOCK_FP8_AVAILABLE = False


class BlasBenchmark(Benchmark):
    """
    benchmark for blas
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, cur_dtype) -> Generator:
        for b, m, n, k in self.shapes:
            yield from self.input_fn(b, m, n, k, cur_dtype, self.device, False)

        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            for b, m, n, k in self.shapes:
                yield from self.input_fn(b, m, n, k, cur_dtype, self.device, True)

    def set_more_shapes(self):
        large_k_shapes = [
            (8, 1848, 1536, 151936),
            (8, 1848, 1536, 128256),
            (8, 1848, 1536, 152064),
            (8, 4096, 1, 152064),
        ]

        model_shaps = model_shapes()
        return large_k_shapes + model_shaps

    def get_tflops(self, op, *args, **kwargs):
        total_flops = 0
        # shape(m,k)(k,n)
        # total_flops mxnx2k
        if self.op_name == "mm":
            total_flops = args[0].shape[0] * args[0].shape[1] * args[1].shape[1] * 2
        # shape(m,n)(n,p)
        # total_flops mxpx(2n+1)
        elif self.op_name == "addmm":
            total_flops = (
                args[0].shape[0] * args[1].shape[1] * (args[1].shape[0] * 2 + 1)
            )
        # total_flops bxnxpx2m
        elif self.op_name == "bmm":
            total_flops = (
                args[0].shape[0]
                * args[0].shape[1]
                * args[1].shape[2]
                * 2
                * args[0].shape[2]
            )
        return total_flops


class BaddbmmBenchmark(BlasBenchmark):
    """
    benchmark for Baddbmm
    """

    def set_more_shapes(self):
        model_shapes_list = model_shapes()

        skip_shapes = [
            (4, 8192, 128256, 4096),
            (4, 8192, 152064, 3584),
        ]

        filtered = []
        for shape in model_shapes_list:
            if shape not in skip_shapes:
                filtered.append(shape)

        return filtered

    def get_tflops(self, op, *args, **kwargs):
        # shape(b,m,k)(b,k,n)
        # total_flops = b * m * n * (2 * k + 1)
        total_flops = (
            args[1].shape[0]
            * args[1].shape[1]
            * args[2].shape[2]
            * (args[1].shape[2] * 2 + 1)
        )
        return total_flops


def addmm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    bias = torch.randn([m, n], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield bias, inp1, inp2.t(),
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield bias, inp1, inp2,


def bmm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.transpose(1, 2)
    else:
        inp2 = torch.randn([b, k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


def baddbmm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=cur_dtype, device=device, requires_grad=True)

    if b_column_major:
        inp2 = torch.randn(
            [b, n, k], dtype=cur_dtype, device=device, requires_grad=True
        )
        inp2 = inp2.transpose(1, 2).contiguous()
    else:
        inp2 = torch.randn(
            [b, k, n], dtype=cur_dtype, device=device, requires_grad=True
        )

    bias = torch.randn([b, m, n], dtype=cur_dtype, device=device, requires_grad=True)

    yield bias, inp1, inp2


def mm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.t()
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


W8A8_BLOCK_FP8_BLOCK_SIZE = [128, 128]


def get_w8a8_block_fp8_dtype():
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return None

    major, _ = torch.cuda.get_device_capability()
    if major > 8 and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if major == 8 and hasattr(torch, "float8_e5m2"):
        return torch.float8_e5m2
    return None


def rand_fp8_tensor(shape, device, dtype):
    finfo = torch.finfo(dtype)
    return (
        torch.randn(shape, device=device, dtype=torch.float32)
        .clamp(min=finfo.min, max=finfo.max)
        .to(dtype)
    )


class W8A8BlockFP8MatmulBenchmark(Benchmark):
    """
    Benchmark for w8a8_block_fp8_matmul.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]
    SHAPE_CONFIG_KEYS = ("mm", "BlasBenchmark")

    def __init__(self, *args, block_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = (
            W8A8_BLOCK_FP8_BLOCK_SIZE[:] if block_size is None else list(block_size)
        )
        self.shape_desc = "M, N, K"

    def set_more_shapes(self):
        return BlasBenchmark.set_more_shapes(self)

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        normalized_shapes = []
        for shape in self.shapes:
            if len(shape) == 4:
                _, m, n, k = shape
                normalized_shapes.append((m, n, k))
            elif len(shape) == 3:
                normalized_shapes.append(shape)
            else:
                raise ValueError(
                    "w8a8_block_fp8_matmul benchmark expects shapes in (M, N, K) "
                    "or (B, M, N, K) format."
                )
        self.shapes = normalized_shapes
        self.shape_desc = "M, N, K"

    def get_input_iter(self, cur_dtype) -> Generator:
        fp8_dtype = get_w8a8_block_fp8_dtype()
        if fp8_dtype is None:
            raise RuntimeError(
                "w8a8_block_fp8_matmul benchmark requires CUDA device with FP8 support"
            )

        block_n, block_k = self.block_size
        for m, n, k in self.shapes:
            num_k_groups = (k + block_k - 1) // block_k
            num_n_groups = (n + block_n - 1) // block_n

            A = rand_fp8_tensor((m, k), self.device, fp8_dtype).contiguous()
            B = rand_fp8_tensor((n, k), self.device, fp8_dtype).contiguous()
            As = (
                0.01
                * torch.rand((m, num_k_groups), dtype=torch.float32, device=self.device)
                + 0.005
            ).contiguous()
            Bs = (
                0.01
                * torch.rand(
                    (num_n_groups, num_k_groups),
                    dtype=torch.float32,
                    device=self.device,
                )
                + 0.005
            ).contiguous()

            yield A, B, As, Bs, self.block_size[:], torch.float16

    def get_tflops(self, op, *args, **kwargs):
        A, B = args[0], args[1]
        m, k = A.shape
        n = B.shape[0]
        return 2 * m * n * k


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, bench_cls",
    [
        pytest.param(
            "addmm",
            torch.addmm,
            addmm_input_fn,
            BlasBenchmark,
            marks=pytest.mark.addmm,
        ),
        pytest.param(
            "bmm",
            torch.bmm,
            bmm_input_fn,
            BlasBenchmark,
            marks=pytest.mark.bmm,
        ),
        pytest.param(
            "mm",
            torch.Tensor.mm,
            mm_input_fn,
            BlasBenchmark,
            marks=pytest.mark.mm,
        ),
        pytest.param(
            "baddbmm",
            torch.baddbmm,
            baddbmm_input_fn,
            BaddbmmBenchmark,
            marks=pytest.mark.baddbmm,
        ),
    ],
)
def test_blas_benchmark(op_name, torch_op, input_fn, bench_cls):
    if flag_gems.vendor_name == "mthreads" and op_name != "baddbmm":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    bench = bench_cls(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=FLOAT_DTYPES
    )
    bench.run()

    if flag_gems.vendor_name == "mthreads" and op_name != "baddbmm":
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.w8a8_block_fp8_matmul
def test_perf_w8a8_block_fp8_matmul():
    if not VLLM_W8A8_BLOCK_FP8_AVAILABLE:
        pytest.skip("w8a8_block_fp8_matmul benchmark requires vLLM baseline operator")
    if get_w8a8_block_fp8_dtype() is None:
        pytest.skip(
            "w8a8_block_fp8_matmul benchmark requires CUDA device with FP8 support"
        )

    bench = W8A8BlockFP8MatmulBenchmark(
        op_name="w8a8_block_fp8_matmul",
        torch_op=vllm_w8a8_triton_block_scaled_mm,
        dtypes=["fp8"],
    )
    bench.set_gems(flag_gems.w8a8_block_fp8_matmul)
    bench.run()


class MvAndOuterBenchmark(GenericBenchmark2DOnly):
    """
    Benchmark for MV and Outer operations
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, cur_dtype, self.device)


def mv_input_fn(m, n, cur_dtype, device):
    inp1 = torch.randn([m, n], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


def outer_input_fn(m, n, cur_dtype, device):
    inp1 = torch.randn([m], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "mv",
            torch.Tensor.mv,
            mv_input_fn,
            marks=pytest.mark.mv,
        ),
        pytest.param(
            "outer",
            torch.Tensor.outer,
            outer_input_fn,
            marks=pytest.mark.outer,
        ),
    ],
)
def test_mv_and_outer_benchmark(op_name, torch_op, input_fn):
    bench = MvAndOuterBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class AddmvBenchmark(GenericBenchmark2DOnly):
    """
    Benchmark for addmv
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, cur_dtype, self.device)


def addmv_input_fn(m, n, cur_dtype, device):
    mat = torch.randn([m, n], dtype=cur_dtype, device=device)
    vec = torch.randn([n], dtype=cur_dtype, device=device)
    bias = torch.randn([m], dtype=cur_dtype, device=device)
    # torch.addmv(bias, mat, vec)
    yield bias, mat, vec


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "addmv",
            torch.addmv,
            addmv_input_fn,
            marks=pytest.mark.addmv,
        ),
    ],
)
def test_addmv_benchmark(op_name, torch_op, input_fn):
    bench = AddmvBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class VdotBenchmark(BlasBenchmark):
    """
    benchmark for vdot
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            m = shape[0]
            yield from self.input_fn(m, cur_dtype, self.device)


@pytest.mark.vdot
def test_vdot_benchmark():
    def vdot_input_fn(m, cur_dtype, device):
        inp1 = torch.randn([m], dtype=cur_dtype, device=device)
        inp2 = torch.randn([m], dtype=cur_dtype, device=device)
        yield inp1, inp2

    bench = VdotBenchmark(
        input_fn=vdot_input_fn,
        op_name="vdot",
        torch_op=torch.Tensor.vdot,
        dtypes=COMPLEX_DTYPES + FLOAT_DTYPES,
    )
    bench.run()


class AddrBenchmark(BlasBenchmark):
    """
    benchmark for addr
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            m, n = shape[0], shape[1]
            yield from self.input_fn(m, n, cur_dtype, self.device)


@pytest.mark.addr
def test_addr_benchmark():
    def addr_input_fn(m, n, cur_dtype, device):
        inp1 = torch.randn([m, n], dtype=cur_dtype, device=device)
        inp2 = torch.randn([m], dtype=cur_dtype, device=device)
        inp3 = torch.randn([n], dtype=cur_dtype, device=device)
        yield inp1, inp2, inp3, {"alpha": 0.5, "beta": 0.5}

    bench = AddrBenchmark(
        input_fn=addr_input_fn,
        op_name="addr",
        torch_op=torch.Tensor.addr,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# =============================================================================
# QC-GEM1 (GemLite v0.5.1) benchmarks
# =============================================================================
try:
    from flag_gems.ops.qcgem1 import (
        gemm,
        gemm_splitK,
        gemv,
        gemv_splitK,
        gemv_revsplitK,
        DType,
        forward_functional,
        GEMLITE_MATMUL_TYPES,
        GEMLITE_MATMUL_TYPES_MAPPING,
        get_matmul_type,
    )
    from flag_gems.ops.qcgem1.dtypes import DTYPE_TO_TORCH, TORCH_TO_DTYPE
    from flag_gems.ops.qcgem1.bitpack import pack_weights_over_cols_triton

    QCGEM1_AVAILABLE = True
except Exception:
    gemm = gemm_splitK = gemv = gemv_splitK = gemv_revsplitK = None
    DType = forward_functional = None
    GEMLITE_MATMUL_TYPES = []
    GEMLITE_MATMUL_TYPES_MAPPING = {}
    get_matmul_type = None
    DTYPE_TO_TORCH = {}
    TORCH_TO_DTYPE = {}
    QCGEM1_AVAILABLE = False


PRECISION_CONFIGS = {
    # name: (W_nbits, input_dtype, activation_scaling, description)
    "W8A16": (8, DType.FP16, False, "W8/FP16 (INT8 weight + FP16 activation)"),
    "W4A16": (4, DType.FP16, False, "W4/FP16 (INT4 weight + FP16 activation)"),
    "W8A8_FP8": (8, DType.FP8, True, "W8/FP8 (INT8 weight + FP8 activation)"),
}


class QCGem1Benchmark(Benchmark):
    """
    Generic benchmark for qcgem1 (GemLite v0.5.1) quantized GEMM kernels.

    Supports:
      - W8A16:  INT8 weights + FP16 activations
      - W4A16:  INT4 weights + FP16 activations
      - W8A8_FP8: INT8 weights + FP8 activations
      - W4A4_MXFP4: MXFP4 weight+activation
      - W4A4_NVFP4: NVFP4 weight+activation
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]
    SHAPE_CONFIG_KEYS = ("mm", "BlasBenchmark")

    def __init__(
        self,
        *args,
        precision="W4A16",
        group_size=128,
        matmul_type=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.group_size = group_size
        self.matmul_type = matmul_type  # None = auto-select based on M
        self.shape_desc = "M, N, K"

        if precision not in PRECISION_CONFIGS:
            raise ValueError(
                f"Unsupported precision '{precision}'. "
                f"Supported: {list(PRECISION_CONFIGS.keys())}"
            )
        w_nbits, self.input_dtype_enum, self.activation_scaling, self._desc = PRECISION_CONFIGS[precision]
        self.w_nbits = w_nbits

    def set_more_shapes(self):
        return BlasBenchmark.set_more_shapes(self)

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        normalized_shapes = []
        for shape in self.shapes:
            if len(shape) == 4:
                _, m, n, k = shape
                normalized_shapes.append((m, n, k))
            elif len(shape) == 3:
                normalized_shapes.append(shape)
            else:
                raise ValueError(
                    "qcgem1 benchmark expects shapes in (M, N, K) "
                    "or (B, M, N, K) format."
                )
        self.shapes = normalized_shapes
        self.shape_desc = "M, N, K"

    def _quantize_weights(self, W_fp, m_type):
        """Quantize float weights to qcgem1 format.

        W_fp: (N, K) float tensor
        Returns: W_q_packed (K, N_packed), scales (N, n_groups), zeros (N, n_groups)
        """
        n, k = W_fp.shape  # (N, K)
        n_groups = k // self.group_size

        # Group-wise quantization
        W_view = W_fp.view(n, n_groups, self.group_size)  # (N, n_groups, G)
        scales = W_view.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scales = scales.squeeze(-1)  # (N, n_groups)

        if self.w_nbits == 8:
            max_val = 127.0
        else:
            max_val = 7.0

        W_normalized = W_view / scales.unsqueeze(-1)
        W_q = W_normalized.round().clamp(-max_val, max_val).to(torch.int8)
        zeros = torch.zeros_like(scales)

        # Pack: 4-bit packs into uint8 bytes, 8-bit stays as-is
        # W_q is (N, K), transpose to (K, N) for pack
        if self.w_nbits == 4:
            W_q_T = W_q.t()  # (K, N)
            W_q_packed, _ = pack_weights_over_cols_triton(
                W_q_T, self.w_nbits, self.group_size, True
            )
        else:
            W_q_T = W_q.t()
            W_q_packed, _ = pack_weights_over_cols_triton(
                W_q_T, self.w_nbits, self.group_size, True
            )

        return W_q_packed, scales, zeros

    def _select_kernel(self, m):
        """Auto-select kernel based on batch size."""
        if self.matmul_type:
            return self.matmul_type
        return get_matmul_type(m, self.w_nbits, mx_dtype=False)

    def get_input_iter(self, cur_dtype) -> Generator:
        if not QCGEM1_AVAILABLE:
            raise RuntimeError("qcgem1 kernels are not available")

        input_dtype_torch = cur_dtype
        input_dtype_enum = self._get_input_dtype_enum(input_dtype_torch)

        for m, n, k in self.shapes:
            # Generate float weights (N, K) and quantize
            W_fp = torch.randn(n, k, dtype=input_dtype_torch, device=self.device)
            W_q_packed, scales, zeros = self._quantize_weights(W_fp, self._select_kernel(m))

            # Input: (M, K)
            x = torch.randn(m, k, dtype=input_dtype_torch, device=self.device)

            # qcgem1 forward args
            tensor_args = [W_q_packed, scales, zeros]
            meta_args = [
                int(self.activation_scaling),  # scaled_activations
                self.w_nbits,
                self.group_size,
                0,  # unpack_mask
                1,  # elements_per_sample
                input_dtype_enum.value,  # input_dtype
                input_dtype_enum.value,  # output_dtype
                0,  # acc_dtype (0 = auto)
                0,  # meta_dtype (0 = auto)
                0,  # channel_scale_mode (0 = auto)
                0,  # W_group_mode (0 = auto)
                1,  # data_contiguous
                -1,  # type_id (-1 = auto)
            ]
            yield x, tensor_args, meta_args, None

    @staticmethod
    def _get_input_dtype_enum(torch_dtype):
        """Map torch dtype to qcgem1 DType enum."""
        if torch_dtype == torch.float16:
            return DType.FP16
        elif torch_dtype == torch.bfloat16:
            return DType.BF16
        elif torch_dtype == torch.float32:
            return DType.FP32
        return DType.FP16

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        W_q_packed = args[1]
        m, k = x.shape
        n = W_q_packed.shape[1]
        return 2 * m * n * k

    def _build_metric_from_input(self, input_item):
        from benchmark.attri_util import BenchmarkMetrics

        x, tensor_args, meta_args, _ = input_item
        m = x.shape[0]
        matmul_type_str = self._select_kernel(m)
        matmul_type_id = GEMLITE_MATMUL_TYPES_MAPPING.get(matmul_type_str, -1)

        metric = BenchmarkMetrics()
        metric.shape_detail = [x.shape, tensor_args[0].shape]

        def qcgem1_fn():
            return forward_functional(x, None, tensor_args, meta_args, matmul_type_id)

        def fp16_ref_fn():
            W_q = tensor_args[0]
            scales = tensor_args[1]
            W_deq = W_q.float() * scales.float().T
            return torch.mm(x, W_deq.T)

        if "latency_base" in self.to_bench_metrics:
            metric.latency_base = self.get_latency(fp16_ref_fn)
        if "latency" in self.to_bench_metrics:
            metric.latency = self.get_latency(qcgem1_fn)
        if "speedup" in self.to_bench_metrics:
            metric.speedup = metric.latency_base / metric.latency
        if "gbps" in self.to_bench_metrics:
            metric.gbps_base = 0
            metric.gbps = 0
        if "tflops" in self.to_bench_metrics:
            metric.tflops = self.get_tflops(None, x, tensor_args[0])
        return metric


# Convenience factory for common precisions
def make_qcgem1_benchmark(precision, op_name=None, group_size=128):
    """Create a QCGem1Benchmark for the given precision."""
    if op_name is None:
        op_name = f"qcgem1_{precision.lower()}"
    bench = QCGem1Benchmark(
        op_name=op_name,
        torch_op=None,
        precision=precision,
        group_size=group_size,
        dtypes=[torch.float16, torch.bfloat16],
    )
    return bench


@pytest.mark.qcgem1
@pytest.mark.parametrize("precision", list(PRECISION_CONFIGS.keys()))
def test_perf_qcgem1_gemm(precision):
    """Benchmark qcgem1 GEMM kernels for different weight precisions."""
    if not QCGEM1_AVAILABLE:
        pytest.skip("qcgem1 kernels not available")

    bench = make_qcgem1_benchmark(
        precision=precision,
        op_name=f"qcgem1_gemm_{precision.lower()}",
    )
    bench.run()


@pytest.mark.qcgem1
def test_perf_qcgem1_gemm_w8a16():
    """Benchmark qcgem1 GEMM W8A16 (INT8 weights + FP16 activations)."""
    if not QCGEM1_AVAILABLE:
        pytest.skip("qcgem1 kernels not available")

    bench = make_qcgem1_benchmark(
        precision="W8A16",
        op_name="qcgem1_gemm_w8a16",
    )
    bench.run()


@pytest.mark.qcgem1
def test_perf_qcgem1_gemm_w4a16():
    """Benchmark qcgem1 GEMM W4A16 (INT4 weights + FP16 activations)."""
    if not QCGEM1_AVAILABLE:
        pytest.skip("qcgem1 kernels not available")

    bench = make_qcgem1_benchmark(
        precision="W4A16",
        op_name="qcgem1_gemm_w4a16",
    )
    bench.run()
