import pytest
import torch

import flag_gems

from . import base, consts


def _build_2_4_weight(N, K, dtype, device):
    """Build a dense 2:4 weight and its bool meta (N, K//4).

    Each row carries an independent 2:4 pattern, matching the kernel semantics.
    """
    K4 = K // 4
    base_w = torch.randn(N, K, dtype=dtype, device=device)
    choice = torch.randint(0, 2, (N, K4), device=device).bool()
    w = torch.zeros(N, K, dtype=dtype, device=device)
    for g in range(K4):
        keep = choice[:, g]
        w[:, 4 * g] = torch.where(
            keep, base_w[:, 4 * g], torch.zeros_like(keep, dtype=dtype)
        )
        w[:, 4 * g + 1] = torch.where(
            keep, base_w[:, 4 * g + 1], torch.zeros_like(keep, dtype=dtype)
        )
        w[:, 4 * g + 2] = torch.where(
            ~keep, base_w[:, 4 * g + 2], torch.zeros_like(keep, dtype=dtype)
        )
        w[:, 4 * g + 3] = torch.where(
            ~keep, base_w[:, 4 * g + 3], torch.zeros_like(keep, dtype=dtype)
        )
    return w, choice.to(torch.int8)


class SparseSemiStructuredLinearBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, op_name, torch_op, use_bias=False, **kwargs):
        super().__init__(op_name, torch_op, dtypes=consts.FLOAT_DTYPES, **kwargs)
        self.use_bias = use_bias

    def set_shapes(self, shape_file_path=None):
        # Representative (M, K) shapes from small to medium; K must be a multiple of 4.
        self.shapes = [
            (16, 32),
            (64, 128),
            (256, 512),
        ]
        self.shape_desc = "M, K"

    def set_more_shapes(self):
        return [
            (128, 1024),
            (512, 2048),
        ]

    def get_input_iter(self, dtype):
        for M, K in self.shapes:
            N = K  # output features equal to input features
            input = torch.randn(M, K, dtype=dtype, device=self.device)
            weight, meta = _build_2_4_weight(N, K, dtype, self.device)
            if self.use_bias:
                bias = torch.randn(N, dtype=dtype, device=self.device)
                yield input, weight, meta, bias
            else:
                yield input, weight, meta

    def get_tflops(self, op, *args, **kwargs):
        M, K = args[0].shape
        N = K
        # 2:4 sparsity: 2 non-zeros per group of 4 -> 1/2 of the dense FLOPs.
        return 2 * M * N * K // 2


def _torch_ref(input, weight, meta):
    """Reference: dense matmul with the 2:4 meta applied to the weight."""
    K4 = weight.shape[1] // 4
    choice = meta.to(torch.bool)
    w = weight.view(weight.shape[0], K4, 4)
    keep = torch.cat(
        [
            choice.unsqueeze(2),
            choice.unsqueeze(2),
            (~choice).unsqueeze(2),
            (~choice).unsqueeze(2),
        ],
        dim=2,
    )
    masked = torch.where(keep, w, torch.zeros_like(w)).reshape(weight.shape[0], -1)
    return torch.matmul(input, masked.t())


def _torch_ref_with_bias(input, weight, meta, bias):
    return _torch_ref(input, weight, meta) + bias


@pytest.mark.sparse_semi_structured_linear
def test_sparse_semi_structured_linear():
    bench = SparseSemiStructuredLinearBenchmark(
        op_name="sparse_semi_structured_linear",
        torch_op=_torch_ref,
        gems_op=flag_gems._sparse_semi_structured_linear,
        use_bias=False,
    )
    bench.run()


@pytest.mark.sparse_semi_structured_linear
def test_sparse_semi_structured_linear_with_bias():
    bench = SparseSemiStructuredLinearBenchmark(
        op_name="sparse_semi_structured_linear",
        torch_op=_torch_ref_with_bias,
        gems_op=flag_gems._sparse_semi_structured_linear,
        use_bias=True,
    )
    bench.run()
