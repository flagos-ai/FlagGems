import math

import pytest
import torch

import flag_gems

from . import base, consts


def _make_sparse_input(shape, dtype, device, sparsity=0.9):
    m, n = shape[-2:]
    dense = torch.randn(shape, dtype=dtype, device=device)
    dense = torch.where(dense == 0, 1.0, dense)
    mask = torch.rand((m, n), device=device) > sparsity
    mask = mask.expand(shape)
    dense = (dense * mask).contiguous()

    if torch.device(device).type == "cuda":
        csr = dense.to_sparse_csr()
    else:
        csr_cpu = dense.cpu().to_sparse_csr()
        csr = torch.sparse_csr_tensor(
            csr_cpu.crow_indices().to(device),
            csr_cpu.col_indices().to(device),
            csr_cpu.values().to(device),
            size=csr_cpu.shape,
            dtype=csr_cpu.dtype,
            device=device,
        )

    csr._dense_twin = dense
    csr._dense_mask = dense != 0
    csr._dense_flat_idx = (dense != 0).reshape(-1).nonzero(as_tuple=False).reshape(-1)
    return csr


def _torch_sampled_addmm(input, mat1, mat2, *, alpha=1.0, beta=1.0, out=None):
    if input.device.type not in ("cuda", "cpu"):
        dense = getattr(input, "_dense_twin", None)
        mask = getattr(input, "_dense_mask", None)
        flat_idx = getattr(input, "_dense_flat_idx", None)
        if dense is None or mask is None or flat_idx is None:
            dense = input.cpu().to_dense().to(input.device)
            mask = dense != 0
            flat_idx = mask.reshape(-1).nonzero(as_tuple=False).reshape(-1)
        res = alpha * (mat1 @ mat2) * mask + beta * dense

        sampled = (
            res.reshape(-1).index_select(0, flat_idx).reshape(input.values().shape)
        )
        if out is not None:
            out.values().copy_(sampled)
            return out
        return torch.sparse_csr_tensor(
            input.crow_indices(),
            input.col_indices(),
            sampled,
            size=input.shape,
            dtype=input.dtype,
            device=input.device,
        )
    if mat1.dtype not in (torch.float16, torch.bfloat16):
        return torch.sparse.sampled_addmm(
            input, mat1, mat2, alpha=alpha, beta=beta, out=out
        )
    input32 = torch.sparse_csr_tensor(
        input.crow_indices(),
        input.col_indices(),
        input.values().to(torch.float32),
        size=input.shape,
        device=input.device,
    )
    result = torch.sparse.sampled_addmm(
        input32,
        mat1.to(torch.float32),
        mat2.to(torch.float32),
        alpha=alpha,
        beta=beta,
    )
    new_values = result.values().to(mat1.dtype)
    if out is not None:
        out.values().copy_(new_values)
        return out
    return torch.sparse_csr_tensor(
        result.crow_indices(),
        result.col_indices(),
        new_values,
        size=result.shape,
        device=result.device,
    )


def _input_fn(b, m, n, k, dtype, device, b_column_major, sparsity=0.9):
    mat1 = torch.randn([b, m, k], dtype=dtype, device=device)
    mat2 = torch.randn([b, k, n], dtype=dtype, device=device)
    sparse_input = _make_sparse_input([b, m, n], dtype, device, sparsity)
    yield sparse_input, mat1, mat2


class SparseSampledAddmmBenchmark(base.BlasBenchmark):

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for b, m, n, k in self.shapes:
            yield from self.input_fn(b, m, n, k, dtype, self.device, False)

    def get_tflops(self, op, *args, **kwargs):
        sparse_input = args[0]
        mat1 = args[1]

        batch = math.prod(sparse_input.shape[:-2]) if sparse_input.dim() > 2 else 1
        nnz_per_batch = sparse_input._nnz()
        k = mat1.shape[-1]
        flops = batch * nnz_per_batch * k * 2
        return flops


@pytest.mark.sparse_sampled_addmm
def test_sparse_sampled_addmm(monkeypatch):
    bench = SparseSampledAddmmBenchmark(
        op_name="sparse_sampled_addmm",
        input_fn=_input_fn,
        torch_op=_torch_sampled_addmm,
        gems_op=flag_gems.sparse_sampled_addmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(b, m, n, k, dtype, device, b_column_major, sparsity=0.9):
    mat1 = torch.randn([b, m, k], dtype=dtype, device=device)
    mat2 = torch.randn([b, k, n], dtype=dtype, device=device)
    sparse_input = _make_sparse_input([b, m, n], dtype, device, sparsity)

    out = torch.sparse_csr_tensor(
        sparse_input.crow_indices().clone(),
        sparse_input.col_indices().clone(),
        sparse_input.values().clone(),
        size=sparse_input.shape,
        dtype=sparse_input.dtype,
        device=sparse_input.device,
    )

    yield sparse_input, mat1, mat2, {"out": out}


@pytest.mark.sparse_sampled_addmm_out
def test_sparse_sampled_addmm_out(monkeypatch):
    bench = SparseSampledAddmmBenchmark(
        op_name="sparse_sampled_addmm_out",
        input_fn=_input_fn_out,
        torch_op=_torch_sampled_addmm,
        gems_op=flag_gems.sparse_sampled_addmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
