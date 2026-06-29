import pytest
import torch

from flag_gems.ops.bmm import bmm as triton_bmm
from flag_gems.ops.bmm import (
    bmm_fp8_w8a8,
    bmm_fp8_w8a8_block_scale,
    bmm_fp8_w8a8_packed_scale,
    bmm_fp8_w8a16,
)
from flag_gems.ops.bmm import bmm_out as triton_bmm_out

from . import base, consts


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device)
        yield inp1, inp2.transpose(1, 2)
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device)
        yield inp1, inp2


@pytest.mark.bmm
def test_bmm(monkeypatch):
    bench = base.BlasBenchmark(
        op_name="bmm",
        input_fn=_input_fn,
        torch_op=torch.bmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device)
        inp2 = inp2.transpose(1, 2)
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device)
    out = torch.empty([b, m, n], dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.bmm_out
def test_bmm_out(monkeypatch):
    bench = base.BlasBenchmark(
        op_name="bmm_out",
        input_fn=_input_fn_out,
        torch_op=torch.bmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


def _is_fp8e4nv_supported():
    if FP8_DTYPE is None or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major + minor / 10 >= 8.9


def _quantize_b_fp8_per_k_block(B, block_size=128):
    fp8_info = torch.finfo(FP8_DTYPE)
    batch, K, N = B.shape
    num_blocks = (K + block_size - 1) // block_size
    padded_k = num_blocks * block_size
    if padded_k != K:
        B_for_scale = torch.cat(
            [
                B,
                torch.zeros((batch, padded_k - K, N), dtype=B.dtype, device=B.device),
            ],
            dim=1,
        )
    else:
        B_for_scale = B
    B_blocked = B_for_scale.reshape(batch, num_blocks, block_size, N).float()
    scale = (B_blocked.abs().amax(dim=2) / fp8_info.max).clamp(min=1e-8)
    B_fp8 = (
        (B_blocked / scale[:, :, None, :])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
    )
    B_fp8 = B_fp8.reshape(batch, padded_k, N)[:, :K, :].contiguous()
    return B_fp8, scale.to(B.dtype).contiguous()


def _dequant_b_fp8_per_k_block(B_fp8, B_scale, block_size=128):
    K = B_fp8.shape[1]
    block_ids = torch.arange(K, device=B_fp8.device) // block_size
    return B_fp8.float() * B_scale.index_select(1, block_ids).float()


def _quantize_a_fp8_per_k_block(A, block_k=128):
    fp8_info = torch.finfo(FP8_DTYPE)
    batch, M, K = A.shape
    num_k_blocks = (K + block_k - 1) // block_k
    padded_k = num_k_blocks * block_k
    if padded_k != K:
        A_for_scale = torch.cat(
            [
                A,
                torch.zeros((batch, M, padded_k - K), dtype=A.dtype, device=A.device),
            ],
            dim=2,
        )
    else:
        A_for_scale = A
    A_blocked = A_for_scale.reshape(batch, M, num_k_blocks, block_k).float()
    scale = (A_blocked.abs().amax(dim=3) / fp8_info.max).clamp(min=1e-8)
    A_fp8 = (
        (A_blocked / scale[:, :, :, None])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
    )
    A_fp8 = A_fp8.reshape(batch, M, padded_k)[:, :, :K].contiguous()
    return A_fp8, scale.float().contiguous()


def _dequant_a_fp8_per_k_block(A_fp8, A_scale, block_k=128):
    K = A_fp8.shape[2]
    block_ids = torch.arange(K, device=A_fp8.device) // block_k
    return A_fp8.float() * A_scale.index_select(2, block_ids).float()


def _quantize_a_fp8_per_mk_block(A, block_m=128, block_k=128):
    fp8_info = torch.finfo(FP8_DTYPE)
    batch, M, K = A.shape
    num_m_blocks = (M + block_m - 1) // block_m
    num_k_blocks = (K + block_k - 1) // block_k
    padded_m = num_m_blocks * block_m
    padded_k = num_k_blocks * block_k
    A_for_scale = A
    if padded_m != M:
        A_for_scale = torch.cat(
            [
                A_for_scale,
                torch.zeros((batch, padded_m - M, K), dtype=A.dtype, device=A.device),
            ],
            dim=1,
        )
    if padded_k != K:
        A_for_scale = torch.cat(
            [
                A_for_scale,
                torch.zeros(
                    (batch, padded_m, padded_k - K),
                    dtype=A.dtype,
                    device=A.device,
                ),
            ],
            dim=2,
        )
    A_blocked = A_for_scale.reshape(
        batch, num_m_blocks, block_m, num_k_blocks, block_k
    ).float()
    scale = (A_blocked.abs().amax(dim=(2, 4)) / fp8_info.max).clamp(min=1e-8)
    A_fp8 = (
        (A_blocked / scale[:, :, None, :, None])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
    )
    A_fp8 = A_fp8.reshape(batch, padded_m, padded_k)[:, :M, :K].contiguous()
    return A_fp8, scale.float().contiguous()


def _dequant_a_fp8_per_mk_block(A_fp8, A_scale, block_m=128, block_k=128):
    M = A_fp8.shape[1]
    K = A_fp8.shape[2]
    m_ids = torch.arange(M, device=A_fp8.device) // block_m
    k_ids = torch.arange(K, device=A_fp8.device) // block_k
    scale = A_scale.index_select(1, m_ids).index_select(2, k_ids)
    return A_fp8.float() * scale.float()


def _quantize_b_fp8_per_nk_block(B, block_n=128, block_k=128):
    fp8_info = torch.finfo(FP8_DTYPE)
    batch, K, N = B.shape
    num_k_blocks = (K + block_k - 1) // block_k
    num_n_blocks = (N + block_n - 1) // block_n
    padded_k = num_k_blocks * block_k
    padded_n = num_n_blocks * block_n
    B_for_scale = B
    if padded_k != K:
        B_for_scale = torch.cat(
            [
                B_for_scale,
                torch.zeros((batch, padded_k - K, N), dtype=B.dtype, device=B.device),
            ],
            dim=1,
        )
    if padded_n != N:
        B_for_scale = torch.cat(
            [
                B_for_scale,
                torch.zeros(
                    (batch, padded_k, padded_n - N),
                    dtype=B.dtype,
                    device=B.device,
                ),
            ],
            dim=2,
        )
    B_blocked = B_for_scale.reshape(
        batch, num_k_blocks, block_k, num_n_blocks, block_n
    ).float()
    scale = (B_blocked.abs().amax(dim=(2, 4)) / fp8_info.max).clamp(min=1e-8)
    B_fp8 = (
        (B_blocked / scale[:, :, None, :, None])
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
    )
    B_fp8 = B_fp8.reshape(batch, padded_k, padded_n)[:, :K, :N].contiguous()
    return B_fp8, scale.float().contiguous()


def _dequant_b_fp8_per_nk_block(B_fp8, B_scale, block_n=128, block_k=128):
    K = B_fp8.shape[1]
    N = B_fp8.shape[2]
    k_ids = torch.arange(K, device=B_fp8.device) // block_k
    n_ids = torch.arange(N, device=B_fp8.device) // block_n
    scale = B_scale.index_select(1, k_ids).index_select(2, n_ids)
    return B_fp8.float() * scale.float()


def _pack_w8a8_scale(A_scale, B_scale):
    return (A_scale[:, :, :, None] * B_scale[:, None, :, :]).contiguous()


def _torch_bmm_fp8_w8a16_baseline(A, B_fp8, B_scale, block_size=128):
    B_dequant = _dequant_b_fp8_per_k_block(B_fp8, B_scale, block_size).to(A.dtype)
    return torch.bmm(A, B_dequant)


def _gems_bmm_fp8_w8a16(A, B_fp8, B_scale, block_size=128):
    return bmm_fp8_w8a16(A, B_fp8, B_scale, block_size=block_size)


def _triton_bmm_bf16_baseline(A, B, B_fp8, B_scale, block_size=128):
    return triton_bmm(A, B)


def _gems_bmm_fp8_w8a16_vs_bf16(A, B, B_fp8, B_scale, block_size=128):
    return bmm_fp8_w8a16(A, B_fp8, B_scale, block_size=block_size)


def _torch_bmm_fp8_w8a8_baseline(
    A_fp8, B_fp8, A_scale, B_scale, block_n=128, block_k=128
):
    A_dequant = _dequant_a_fp8_per_k_block(A_fp8, A_scale, block_k)
    B_dequant = _dequant_b_fp8_per_nk_block(B_fp8, B_scale, block_n, block_k)
    return torch.bmm(A_dequant, B_dequant)


def _gems_bmm_fp8_w8a8(A_fp8, B_fp8, A_scale, B_scale, block_n=128, block_k=128):
    return bmm_fp8_w8a8(A_fp8, B_fp8, A_scale, B_scale, block_size=(block_n, block_k))


def _triton_bmm_bf16_baseline_w8a8(
    A, B, A_fp8, B_fp8, A_scale, B_scale, block_n=128, block_k=128
):
    return triton_bmm(A, B)


def _gems_bmm_fp8_w8a8_vs_bf16(
    A, B, A_fp8, B_fp8, A_scale, B_scale, block_n=128, block_k=128
):
    return bmm_fp8_w8a8(A_fp8, B_fp8, A_scale, B_scale, block_size=(block_n, block_k))


def _triton_bmm_bf16_out_baseline_w8a8(
    A, B, A_fp8, B_fp8, A_scale, B_scale, out, block_n=128, block_k=128
):
    return triton_bmm_out(A, B, out)


def _triton_bmm_bf16_out_baseline_packed_scale(
    A, B, A_fp8, B_fp8, scale, out, block_n=128, block_k=128
):
    return triton_bmm_out(A, B, out)


def _triton_bmm_bf16_out_baseline_block_scale(
    A, B, A_fp8, B_fp8, A_scale, B_scale, out, block_m=128, block_n=128, block_k=128
):
    return triton_bmm_out(A, B, out)


def _gems_bmm_fp8_w8a8_out_vs_bf16(
    A, B, A_fp8, B_fp8, A_scale, B_scale, out, block_n=128, block_k=128
):
    return bmm_fp8_w8a8(
        A_fp8,
        B_fp8,
        A_scale,
        B_scale,
        block_size=(block_n, block_k),
        out=out,
    )


def _gems_bmm_fp8_w8a8_packed_scale_vs_bf16(
    A, B, A_fp8, B_fp8, scale, out, block_n=128, block_k=128
):
    return bmm_fp8_w8a8_packed_scale(
        A_fp8,
        B_fp8,
        scale,
        block_size=(block_n, block_k),
        out=out,
    )


def _gems_bmm_fp8_w8a8_block_scale_vs_bf16(
    A,
    B,
    A_fp8,
    B_fp8,
    A_scale,
    B_scale,
    out,
    block_m=128,
    block_n=128,
    block_k=128,
):
    return bmm_fp8_w8a8_block_scale(
        A_fp8,
        B_fp8,
        A_scale,
        B_scale,
        block_size=(block_m, block_n, block_k),
        out=out,
    )


class BmmFp8W8A16Benchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "B, M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (2, 16, 32, 64),
            (4, 64, 64, 128),
            (2, 128, 128, 256),
            (4, 256, 256, 512),
            (8, 512, 512, 1024),
        ]

    def get_input_iter(self, dtype):
        for batch, m, n, k in self.shapes:
            A = torch.randn((batch, m, k), device=self.device, dtype=dtype)
            B = torch.randn((batch, k, n), device=self.device, dtype=dtype)
            B_fp8, B_scale = _quantize_b_fp8_per_k_block(B)
            yield A, B_fp8, B_scale, 128


class BmmFp8W8A16VsBf16Benchmark(BmmFp8W8A16Benchmark):
    def get_input_iter(self, dtype):
        for batch, m, n, k in self.shapes:
            A = torch.randn((batch, m, k), device=self.device, dtype=dtype)
            B = torch.randn((batch, k, n), device=self.device, dtype=dtype)
            B_fp8, B_scale = _quantize_b_fp8_per_k_block(B)
            yield A, B, B_fp8, B_scale, 128


class BmmFp8W8A8Benchmark(BmmFp8W8A16Benchmark):
    def get_input_iter(self, dtype):
        for batch, m, n, k in self.shapes:
            A = torch.randn((batch, m, k), device=self.device, dtype=dtype)
            B = torch.randn((batch, k, n), device=self.device, dtype=dtype)
            A_fp8, A_scale = _quantize_a_fp8_per_k_block(A)
            B_fp8, B_scale = _quantize_b_fp8_per_nk_block(B)
            yield A_fp8, B_fp8, A_scale, B_scale, 128, 128


class BmmFp8W8A8VsBf16Benchmark(BmmFp8W8A16Benchmark):
    def get_input_iter(self, dtype):
        for batch, m, n, k in self.shapes:
            A = torch.randn((batch, m, k), device=self.device, dtype=dtype)
            B = torch.randn((batch, k, n), device=self.device, dtype=dtype)
            A_fp8, A_scale = _quantize_a_fp8_per_k_block(A)
            B_fp8, B_scale = _quantize_b_fp8_per_nk_block(B)
            yield A, B, A_fp8, B_fp8, A_scale, B_scale, 128, 128


class BmmFp8W8A8OutVsBf16Benchmark(BmmFp8W8A16Benchmark):
    def get_input_iter(self, dtype):
        for batch, m, n, k in self.shapes:
            A = torch.randn((batch, m, k), device=self.device, dtype=dtype)
            B = torch.randn((batch, k, n), device=self.device, dtype=dtype)
            A_fp8, A_scale = _quantize_a_fp8_per_k_block(A)
            B_fp8, B_scale = _quantize_b_fp8_per_nk_block(B)
            out = torch.empty((batch, m, n), device=self.device, dtype=dtype)
            yield A, B, A_fp8, B_fp8, A_scale, B_scale, out, 128, 128


class BmmFp8W8A8PackedScaleVsBf16Benchmark(BmmFp8W8A16Benchmark):
    def get_input_iter(self, dtype):
        for batch, m, n, k in self.shapes:
            A = torch.randn((batch, m, k), device=self.device, dtype=dtype)
            B = torch.randn((batch, k, n), device=self.device, dtype=dtype)
            A_fp8, A_scale = _quantize_a_fp8_per_k_block(A)
            B_fp8, B_scale = _quantize_b_fp8_per_nk_block(B)
            scale = _pack_w8a8_scale(A_scale, B_scale)
            out = torch.empty((batch, m, n), device=self.device, dtype=dtype)
            yield A, B, A_fp8, B_fp8, scale, out, 128, 128


class BmmFp8W8A8BlockScaleVsBf16Benchmark(BmmFp8W8A16Benchmark):
    def get_input_iter(self, dtype):
        for batch, m, n, k in self.shapes:
            A = torch.randn((batch, m, k), device=self.device, dtype=dtype)
            B = torch.randn((batch, k, n), device=self.device, dtype=dtype)
            A_fp8, A_scale = _quantize_a_fp8_per_mk_block(A)
            B_fp8, B_scale = _quantize_b_fp8_per_nk_block(B)
            out = torch.empty((batch, m, n), device=self.device, dtype=dtype)
            yield A, B, A_fp8, B_fp8, A_scale, B_scale, out, 128, 128, 128


@pytest.mark.bmm
@pytest.mark.skipif(
    not _is_fp8e4nv_supported(),
    reason="FP8 BMM W8A16 requires CUDA fp8e4nv support",
)
def test_bmm_fp8_w8a16():
    bench = BmmFp8W8A16Benchmark(
        op_name="bmm_fp8_w8a16",
        torch_op=_torch_bmm_fp8_w8a16_baseline,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_bmm_fp8_w8a16)
    bench.run()


@pytest.mark.bmm
@pytest.mark.skipif(
    not _is_fp8e4nv_supported(),
    reason="FP8 BMM W8A16 requires CUDA fp8e4nv support",
)
def test_bmm_fp8_w8a16_vs_triton_bf16():
    bench = BmmFp8W8A16VsBf16Benchmark(
        op_name="bmm_fp8_w8a16_vs_triton_bf16",
        torch_op=_triton_bmm_bf16_baseline,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_bmm_fp8_w8a16_vs_bf16)
    bench.run()


@pytest.mark.bmm
@pytest.mark.skipif(
    not _is_fp8e4nv_supported(),
    reason="FP8 BMM W8A8 requires CUDA fp8e4nv support",
)
def test_bmm_fp8_w8a8():
    bench = BmmFp8W8A8Benchmark(
        op_name="bmm_fp8_w8a8",
        torch_op=_torch_bmm_fp8_w8a8_baseline,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_bmm_fp8_w8a8)
    bench.run()


@pytest.mark.bmm
@pytest.mark.skipif(
    not _is_fp8e4nv_supported(),
    reason="FP8 BMM W8A8 requires CUDA fp8e4nv support",
)
def test_bmm_fp8_w8a8_vs_triton_bf16():
    bench = BmmFp8W8A8VsBf16Benchmark(
        op_name="bmm_fp8_w8a8_vs_triton_bf16",
        torch_op=_triton_bmm_bf16_baseline_w8a8,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_bmm_fp8_w8a8_vs_bf16)
    bench.run()


@pytest.mark.bmm
@pytest.mark.skipif(
    not _is_fp8e4nv_supported(),
    reason="FP8 BMM W8A8 requires CUDA fp8e4nv support",
)
def test_bmm_fp8_w8a8_out_vs_triton_bf16():
    bench = BmmFp8W8A8OutVsBf16Benchmark(
        op_name="bmm_fp8_w8a8_out_vs_triton_bf16",
        torch_op=_triton_bmm_bf16_out_baseline_w8a8,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_bmm_fp8_w8a8_out_vs_bf16)
    bench.run()


@pytest.mark.bmm
@pytest.mark.skipif(
    not _is_fp8e4nv_supported(),
    reason="FP8 BMM W8A8 requires CUDA fp8e4nv support",
)
def test_bmm_fp8_w8a8_packed_scale_vs_triton_bf16():
    bench = BmmFp8W8A8PackedScaleVsBf16Benchmark(
        op_name="bmm_fp8_w8a8_packed_scale_vs_triton_bf16",
        torch_op=_triton_bmm_bf16_out_baseline_packed_scale,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_bmm_fp8_w8a8_packed_scale_vs_bf16)
    bench.run()


@pytest.mark.bmm
@pytest.mark.skipif(
    not _is_fp8e4nv_supported(),
    reason="FP8 BMM W8A8 requires CUDA fp8e4nv support",
)
def test_bmm_fp8_w8a8_block_scale_vs_triton_bf16():
    bench = BmmFp8W8A8BlockScaleVsBf16Benchmark(
        op_name="bmm_fp8_w8a8_block_scale_vs_triton_bf16",
        torch_op=_triton_bmm_bf16_out_baseline_block_scale,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_bmm_fp8_w8a8_block_scale_vs_bf16)
    bench.run()
