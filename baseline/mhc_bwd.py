# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVIDIA native baseline for `mhc_bwd`.

Mirrors benchmark/test_mhc.py (MHCBwdBenchmark, a base.Benchmark subclass):
    op_name  = "mhc_bwd"
    torch_op = flag_gems.mhc_bwd_ref
    dtypes   = [torch.float32]

The class overrides set_shapes with its own (seqlen, n_stream) list and
declares shape_desc "seqlen, n_stream", so core_shapes.yaml is not consulted.
n_stream is 4 for every entry and sinkhorn_iters is 20.

DEVIATION: this operator's "native" reference is not an ATen operator at all --
it is `flag_gems.mhc_bwd_ref`, a pure-PyTorch implementation of the Sinkhorn
backward via implicit conjugate gradient that lives in the FlagGems tree
(src/flag_gems/deprecated/mhc/mhc_bwd.py). A baseline module cannot import
flag_gems, so that reference and the `sinkhorn_forward` input helper are
reproduced verbatim below from the same source. They use only torch
primitives, so the measured work is identical; the duplication is the cost of
keeping this module self-contained. Re-check both against
src/flag_gems/deprecated/mhc/mhc_bwd.py if that file changes. See Q&A.md.

The benchmark seeds torch with 42 and draws M from Uniform(0, 4) before
running the Sinkhorn forward, so build_inputs does the same -- which also makes
the collector's two build_inputs calls agree.
"""

import torch

OP_NAME = "mhc_bwd"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.distributions.uniform.Uniform is unavailable on this torch build"
)

# MHCBwdBenchmark.set_shapes -- (seqlen, n_stream)
_SHAPES = [
    (256, 4),
    (1024, 4),
    (4096, 4),
    (8192, 4),
    (16384, 4),
    (65536, 4),
]

_SINKHORN_ITERS = 20
_SEED = 42
# src/flag_gems/deprecated/mhc/mhc_bwd.py::EPS
_EPS = 1e-10


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent.

    torch.ops.aten.<name>.<overload> raises for an operator this build does not
    carry, so the lookup is guarded.
    """
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def _sinkhorn_forward(m, iters=_SINKHORN_ITERS):
    """flag_gems.sinkhorn_forward, reproduced from mhc_bwd.py."""
    p = torch.exp(m)
    r = p.clone()
    for _ in range(iters):
        r = r / r.sum(-2, keepdim=True)
        r = r / r.sum(-1, keepdim=True)
    return r, p


def _mhc_bwd_ref(out, dout, cg_iters=None):
    """flag_gems.mhc_bwd_ref, reproduced from mhc_bwd.py.

    PyTorch reference for the Sinkhorn backward via implicit conjugate
    gradient. cg_iters defaults to 2 * n_stream.
    """
    _, n_stream, _ = out.shape
    if cg_iters is None:
        cg_iters = 2 * n_stream

    r_mat = out.float()
    d_r = dout.float()
    r_dr = r_mat * d_r
    b1 = r_dr.sum(dim=-1)
    b2 = r_dr.sum(dim=-2)
    x1 = torch.zeros_like(b1)
    x2 = torch.zeros_like(b2)

    def matvec(r, x1_in, x2_in):
        y1 = (r * x2_in.unsqueeze(-2)).sum(dim=-1) + x1_in
        y2 = (r * x1_in.unsqueeze(-1)).sum(dim=-2) + x2_in
        return y1, y2

    r1, r2 = b1.clone(), b2.clone()
    p1, p2 = r1.clone(), r2.clone()
    r_normsq = (r1 * r1 + r2 * r2).sum(dim=-1)

    for _ in range(cg_iters):
        ap1, ap2 = matvec(r_mat, p1, p2)
        p_ap = (p1 * ap1 + p2 * ap2).sum(dim=-1)
        alpha = (r_normsq / (p_ap + _EPS)).unsqueeze(-1)
        x1 = x1 + alpha * p1
        x2 = x2 + alpha * p2
        r1 = r1 - alpha * ap1
        r2 = r2 - alpha * ap2
        r_new_normsq = (r1 * r1 + r2 * r2).sum(dim=-1)
        beta = (r_new_normsq / (r_normsq + _EPS)).unsqueeze(-1)
        p1 = r1 + beta * p1
        p2 = r2 + beta * p2
        r_normsq = r_new_normsq

    return (d_r - x1.unsqueeze(-1) - x2.unsqueeze(-2)) * r_mat


def native():
    if _resolve("torch.distributions.uniform.Uniform") is None:
        return None
    return _mhc_bwd_ref


def grid():
    return [{"seqlen": seqlen, "n_stream": n} for seqlen, n in _SHAPES]


def build_inputs(binding, dtype, device):
    seqlen, n_stream = binding["seqlen"], binding["n_stream"]
    torch.manual_seed(_SEED)
    dist = torch.distributions.uniform.Uniform(0.0, 4.0)
    m = dist.sample((seqlen, n_stream, n_stream)).to(device)
    r_mat, _ = _sinkhorn_forward(m, iters=_SINKHORN_ITERS)
    d_r = torch.randn_like(r_mat)
    return (r_mat.detach(), d_r), {}


def key_shape(binding):
    return [binding["seqlen"], binding["n_stream"]]


def config(binding, dtype):
    seqlen, n_stream = binding["seqlen"], binding["n_stream"]
    dt = str(dtype)
    shape = [seqlen, n_stream, n_stream]
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "sinkhorn output R"},
                {"shape": shape, "dtype": dt, "note": "upstream grad dR"},
            ],
            "kwargs": {},
        },
        "outputs": {"grad_input": {"shape": shape, "dtype": dt}},
        "dims": {
            "seqlen": seqlen,
            "n_stream": n_stream,
            "sinkhorn_iters": _SINKHORN_ITERS,
            "cg_iters": 2 * n_stream,
        },
    }
