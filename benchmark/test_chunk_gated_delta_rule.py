from typing import Generator

import pytest
import torch

from . import base, consts


class ChunkGatedDeltaRuleBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        # Fully override the default shape iteration: each "shape" encodes
        # ``(B, T, H, K, V)`` for chunk_gated_delta_rule and is not 4D.
        shapes = [
            (1, 256, 4, 64, 64),
            (2, 1024, 4, 64, 64),
            (1, 4096, 8, 128, 128),
            (4, 512, 16, 128, 128),
        ]
        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


def _input_fn(shape, dtype, device):
    B, T, H, K, V = shape
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.3
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = -torch.rand(B, T, H, dtype=torch.float32, device=device) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    yield q, k, v, g, beta


def _torch_ref(q, k, v, g, beta):
    """Eager reference (chunk-parallel torch) — bandwidth-bound floor."""
    from flag_gems.ops.chunk_gated_delta_rule import _eager_chunk_gated_delta_rule

    o, _ = _eager_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=None,
        output_final_state=False,
    )
    return o


@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule():
    bench = ChunkGatedDeltaRuleBenchmark(
        op_name="chunk_gated_delta_rule",
        input_fn=_input_fn,
        torch_op=_torch_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
