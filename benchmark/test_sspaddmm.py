import pytest
import torch

import flag_gems

from . import base, consts


def _make_sparse_coo(shape, dtype, device, sparsity=0.9):
    m, n = shape
    dense = torch.randn((m, n), dtype=dtype, device=device)
    mask = torch.rand((m, n), device=device) > sparsity
    dense = (dense * mask).contiguous()
    return dense.to_sparse().coalesce()


def _torch_sspaddmm(input, mat1, mat2, *, alpha=1.0, beta=1.0, out=None):
    # torch.sspaddmm has no CUDA kernel and no low-precision CPU kernel, so run
    # the reference on CPU in fp32 and cast/move the result back. This mirrors
    # what a native implementation would produce.
    dtype = input.dtype
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

    inp_c = input.cpu()
    mat1_c = mat1.cpu()
    mat2_c = mat2.cpu()
    if compute_dtype != dtype:
        inp_c = torch.sparse_coo_tensor(
            inp_c.indices(), inp_c.values().to(compute_dtype), size=inp_c.shape
        )
        mat1_c = torch.sparse_coo_tensor(
            mat1_c.indices(), mat1_c.values().to(compute_dtype), size=mat1_c.shape
        )
        mat2_c = mat2_c.to(compute_dtype)

    res_cpu = torch.sspaddmm(inp_c, mat1_c, mat2_c, alpha=alpha, beta=beta)
    res_cpu = res_cpu.coalesce()
    res_cpu = torch.sparse_coo_tensor(
        res_cpu.indices(), res_cpu.values().to(dtype), size=res_cpu.shape
    )
    return res_cpu.to(input.device)


def _input_fn(b, m, n, k, dtype, device, b_column_major, sparsity=0.9):
    mat1 = _make_sparse_coo((m, k), dtype, device, sparsity)
    mat2 = torch.randn((k, n), dtype=dtype, device=device)
    sparse_input = _make_sparse_coo((m, n), dtype, device, sparsity)
    yield sparse_input, mat1, mat2


# torch.sspaddmm has no CUDA kernel, so the reference path runs a sparse COO
# matmul on CPU whose cost grows super-linearly with the matrix dims (a single
# 2048x2048x2048 call already takes ~10s). The benchmark harness additionally
# probes each shape ~5x to size its iteration count, so the huge BlasBenchmark
# shapes (up to 16x4096x4096x4096) would run for many minutes each. Cap the
# benchmark at moderate square shapes that still exercise the kernel while
# keeping the CPU reference fast enough to finish quickly.
SSPADDMM_SHAPES = [
    (1, 128, 128, 128),
    (1, 256, 256, 256),
    (1, 384, 384, 384),
    (1, 512, 512, 512),
    (1, 1024, 1024, 1024),
]


class SspaddmmBenchmark(base.BlasBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Ignore the shared BlasBenchmark shapes from core_shapes.yaml; the CPU
        # reference for sspaddmm cannot handle them in reasonable time.
        self.shapes = list(SSPADDMM_SHAPES)

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for b, m, n, k in self.shapes:
            yield from self.input_fn(b, m, n, k, dtype, self.device, False)

    def get_tflops(self, op, *args, **kwargs):
        sparse_input, mat1 = args[0], args[1]
        nnz = mat1._nnz()
        n = sparse_input.shape[1]
        # each mat1 nonzero contributes a length-n scaled-add (2 flops per elem)
        return nnz * n * 2


@pytest.mark.sspaddmm
def test_sspaddmm(monkeypatch):
    bench = SspaddmmBenchmark(
        op_name="sspaddmm",
        input_fn=_input_fn,
        torch_op=_torch_sspaddmm,
        gems_op=flag_gems.sspaddmm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _input_fn_out(b, m, n, k, dtype, device, b_column_major, sparsity=0.9):
    mat1 = _make_sparse_coo((m, k), dtype, device, sparsity)
    mat2 = torch.randn((k, n), dtype=dtype, device=device)
    sparse_input = _make_sparse_coo((m, n), dtype, device, sparsity)
    out = torch.empty((m, n), dtype=dtype, device=device).to_sparse()
    yield sparse_input, mat1, mat2, {"out": out}


@pytest.mark.sspaddmm_out
def test_sspaddmm_out(monkeypatch):
    bench = SspaddmmBenchmark(
        op_name="sspaddmm_out",
        input_fn=_input_fn_out,
        torch_op=_torch_sspaddmm,
        gems_op=flag_gems.sspaddmm_out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
