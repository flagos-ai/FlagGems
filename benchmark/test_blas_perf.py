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
from benchmark.performance_utils import Benchmark, GenericBenchmark2DOnly, SkipVersion

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


class GroupmmBenchmark(BlasBenchmark):
    """
    benchmark for Groupmm
    """

    def get_input_iter(self, cur_dtype) -> Generator:
        for groups, n, k in self.shapes:
            yield from self.input_fn(groups, n, k, cur_dtype, self.device)

    def set_more_shapes(self):
        return None

    def get_tflops(self, op, *args, **kwargs):
        groups, N, K = args[1].shape
        size_per_group = torch.diff(
            args[2], prepend=torch.zeros(1, device="cuda", dtype=torch.int32)
        )
        total_flops = 0
        for i in range(groups):
            total_flops += size_per_group[i].item() * N * K * 2
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


def group_mm_input_fn(groups, N, K, cur_dtype, device):
    assert cur_dtype == torch.bfloat16
    import random

    group_A_list = []
    group_B_list = []
    A_offs = 0
    B_offs = 0
    M_list = []
    for i in range(groups):
        M_g = random.randint(1, 16384)
        N_g = N
        K_g = K
        A_g = torch.rand([M_g, K_g], device="cuda", dtype=cur_dtype)
        B_g = torch.rand([K_g, N_g], device="cuda", dtype=cur_dtype)
        group_A_list.append(A_g)
        group_B_list.append(B_g)
        M_list.append(M_g)
        A_offs += M_g * K_g
        B_offs += K_g * N_g

    mat_a = torch.cat([x for x in group_A_list], dim=0)
    mat_b = torch.stack([x for x in group_B_list], dim=0)
    offs = torch.tensor(
        [sum(M_list[: i + 1]) for i in range(groups)], dtype=torch.int32, device="cuda"
    )

    yield mat_a, mat_b, offs


W8A8_BLOCK_FP8_MNK_SHAPES = [
    (64, 128, 128),
    (128, 256, 512),
    (1, 4096, 7168),
    (16, 4096, 7168),
    (64, 4096, 7168),
    (83, 7748, 3884),
    (84, 7168, 3884),
]
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

    def __init__(self, *args, block_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = (
            W8A8_BLOCK_FP8_BLOCK_SIZE[:] if block_size is None else list(block_size)
        )
        self.shape_desc = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = W8A8_BLOCK_FP8_MNK_SHAPES[:]
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
        pytest.param(
            "groupmm",
            None if SkipVersion("torch", "<2.8") else torch._grouped_mm,  # torch 2.8.0
            group_mm_input_fn,
            GroupmmBenchmark,
            marks=[
                pytest.mark.skipif(
                    SkipVersion("torch", "<2.8"),
                    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
                ),
                pytest.mark.groupmm,
            ],
        ),
    ],
)
def test_blas_benchmark(op_name, torch_op, input_fn, bench_cls):
    if flag_gems.vendor_name == "mthreads" and op_name not in ("mm", "baddbmm"):
        os.environ["MUSA_ENABLE_SQMMA"] = "1"
    if op_name == "groupmm":
        FLOAT_DTYPES = [torch.bfloat16]

    bench = bench_cls(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=FLOAT_DTYPES
    )

    if op_name == "groupmm":
        gems_op = flag_gems.group_mm
        bench.set_gems(gems_op)

    bench.run()

    if flag_gems.vendor_name == "mthreads" and op_name not in ("mm", "baddbmm"):
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
