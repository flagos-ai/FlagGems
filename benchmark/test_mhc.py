import pytest
import torch

from flag_gems.fused.mhc.hc_split_sinkhorn import (
    hc_split_sinkhorn,
    mhc_split_sinkhorn_torch_ref,
)
from flag_gems.fused.mhc.mhc_bwd import mhc_bwd, mhc_bwd_ref, sinkhorn_forward
from flag_gems.fused.mhc.mhc_post import mhc_post, mhc_post_ref
from flag_gems.fused.mhc.mhc_pre import mhc_pre, mhc_pre_ref

from . import base


class MHCPostBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, H"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4096, 1280),
            (4096, 2560),
            (4096, 7168),
        ]

    def get_input_iter(self, dtype):
        for n, h in self.shapes:
            hc_mult = 4
            x = torch.randn((n, h), dtype=torch.bfloat16, device=self.device)
            residual = torch.randn(
                (n, hc_mult, h), dtype=torch.bfloat16, device=self.device
            )
            post_layer_mix = torch.randn(
                (n, hc_mult, 1), dtype=torch.float32, device=self.device
            )
            comb_res_mix = torch.randn(
                (n, hc_mult, hc_mult), dtype=torch.float32, device=self.device
            )
            yield x, residual, post_layer_mix, comb_res_mix


@pytest.mark.mhc_post
def test_mhc_post():
    bench = MHCPostBenchmark(
        op_name="mhc_post",
        torch_op=mhc_post_ref,
        gems_op=mhc_post,
        dtypes=[torch.bfloat16],
    )
    bench.run()


class MHCPreBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, hidden_size"

    def __init__(self, *args, hc_mult=4, sinkhorn_repeat=10, **kwargs):
        self.hc_mult = hc_mult
        self.sinkhorn_repeat = sinkhorn_repeat
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (512, 1280),
            (512, 2560),
            (512, 4096),
            (1024, 1280),
            (1024, 2560),
            (1024, 4096),
            (2048, 1280),
            (2048, 2560),
            (2048, 4096),
            (8192, 1280),
            (8192, 2560),
            (8192, 4096),
        ]

    def get_input_iter(self, dtype):
        for n, hidden_size in self.shapes:
            hc_mult = self.hc_mult
            hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
            device = self.device

            torch.manual_seed(42)
            residual = (
                torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device)
                .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
                .bfloat16()
            )
            fn = (
                torch.randn(
                    (hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device
                )
                * 1e-4
                * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
            ).flatten(1, 2)
            hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
            hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

            yield (
                residual,
                fn,
                hc_scale,
                hc_base,
                1e-6,  # rms_eps
                1e-6,  # hc_pre_eps
                1e-6,  # hc_sinkhorn_eps
                1.0,  # hc_post_mult_value
                self.sinkhorn_repeat,
            )


@pytest.mark.mhc_pre
def test_mhc_pre():
    bench = MHCPreBenchmark(
        op_name="mhc_pre",
        torch_op=mhc_pre_ref,
        gems_op=mhc_pre,
        dtypes=[torch.bfloat16],
    )
    bench.run()


class MHCSplitSinkhornBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "batch, seqlen"

    def __init__(self, *args, hc_mult=4, sinkhorn_iters=20, eps=1e-6, **kwargs):
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (8, 16, 4),
            (32, 64, 4),
            (128, 128, 4),
            (256, 256, 4),
            (8, 16, 2),
            (32, 64, 2),
            (128, 128, 2),
            (256, 256, 2),
        ]

    def get_input_iter(self, dtype):
        for batch, seqlen, hc_mult in self.shapes:
            mix_hc = (2 + hc_mult) * hc_mult
            device = self.device

            torch.manual_seed(42)
            mixes = torch.randn(
                (batch, seqlen, mix_hc), dtype=torch.float32, device=device
            )
            hc_scale = torch.randn((3,), dtype=torch.float32, device=device) * 0.1
            hc_base = torch.randn((mix_hc,), dtype=torch.float32, device=device) * 0.1

            yield mixes, hc_scale, hc_base, hc_mult, self.sinkhorn_iters, self.eps


# TODO(Qiming): Find out where this is implemented
@pytest.mark.hc_split_sinkhorn_forward
def test_hc_split_sinkhorn_forward():
    bench = MHCSplitSinkhornBenchmark(
        op_name="hc_split_sinkhorn",
        torch_op=mhc_split_sinkhorn_torch_ref,
        dtypes=[torch.float32],
    )
    bench.set_gems(hc_split_sinkhorn)
    bench.run()


class MHCBwdBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "seqlen, n_stream"

    def __init__(self, *args, n_stream=4, sinkhorn_iters=20, **kwargs):
        self.n_stream = n_stream
        self.sinkhorn_iters = sinkhorn_iters
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (256, 4),
            (1024, 4),
            (4096, 4),
            (8192, 4),
            (16384, 4),
            (65536, 4),
        ]

    def get_input_iter(self, dtype):
        for seqlen, n_stream in self.shapes:
            device = self.device
            torch.manual_seed(42)

            dist = torch.distributions.uniform.Uniform(0.0, 4.0)
            M = dist.sample((seqlen, n_stream, n_stream)).to(device)
            R, _P = sinkhorn_forward(M, iters=self.sinkhorn_iters)
            dR = torch.randn_like(R)

            yield R.detach(), dR


@pytest.mark.mhc_bwd
def test_mhc_bwd():
    bench = MHCBwdBenchmark(
        op_name="mhc_bwd",
        torch_op=mhc_bwd_ref,
        gems_op=mhc_bwd,
        dtypes=[torch.float32],
    )
    bench.run()
