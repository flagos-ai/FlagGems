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

"""NVIDIA native baseline for `convert_indices_from_csr_to_coo_out`.

Mirrors benchmark/test_convert_indices_from_csr_to_coo.py
(CsrToCooOutBenchmark, a base.Benchmark subclass):
    op_name  = "convert_indices_from_csr_to_coo_out"
    torch_op = torch.ops.aten._convert_indices_from_csr_to_coo.out
    dtypes   = consts.INT_DTYPES + consts.EXTRA_INT_DTYPES

The parent class overrides set_shapes with CSR_TO_COO_CASES -- (lead, rows,
nnz) descriptors rather than tensor shapes -- so core_shapes.yaml is not
consulted.

Only integral index dtypes are accepted; on NVIDIA int64_enabled is True, so
the pool is [int16, int32, int8, uint8, int64]. A descriptor whose `nnz` or
`rows` exceeds the dtype's maximum is skipped rather than built from wrapped
pointer values -- unsupported_reason() reproduces that filter so the collector
records an explicit row instead of silently dropping the workload.
"""

import torch

OP_NAME = "convert_indices_from_csr_to_coo_out"
DTYPES = [torch.int16, torch.int32, torch.int8, torch.uint8, torch.int64]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._convert_indices_from_csr_to_coo.out is unavailable on this build"
)

# test_convert_indices_from_csr_to_coo.py::CSR_TO_COO_CASES -- (lead, rows, nnz)
_SHAPES = [
    ((), 8, 8),
    ((2, 2), 4, 4),
    ((), 1024, 65536),
    ((8,), 512, 8192),
    ((1024,), 256, 4096),
    ((64, 16), 256, 4096),
    ((16, 8, 8), 128, 2048),
]

_CSR_SEED = 0


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


def _make_csr(lead, rows, nnz, dtype, device):
    """benchmark::_make_csr -- monotonic pointers ending at nnz per block."""
    lead = tuple(lead)
    gen = torch.Generator(device="cpu").manual_seed(_CSR_SEED)
    cuts = torch.sort(
        torch.randint(0, nnz + 1, lead + (rows,), generator=gen, dtype=torch.int64),
        dim=-1,
    ).values
    crow = torch.cat([torch.zeros(lead + (1,), dtype=torch.int64), cuts], dim=-1)
    crow[..., -1] = nnz
    col = torch.randint(0, rows, lead + (nnz,), generator=gen, dtype=torch.int64)
    return crow.to(dtype).to(device), col.to(dtype).to(device)


def native():
    op = _resolve("torch.ops.aten._convert_indices_from_csr_to_coo.out")
    return op if callable(op) else None


def grid():
    return [
        {"lead": list(lead), "rows": rows, "nnz": nnz} for lead, rows, nnz in _SHAPES
    ]


def unsupported_reason(binding, dtype):
    max_index = int(torch.iinfo(dtype).max)
    if binding["nnz"] > max_index or binding["rows"] > max_index:
        return (
            f"{dtype} cannot address rows={binding['rows']} / "
            f"nnz={binding['nnz']} (max {max_index}); the benchmark skips it"
        )
    return None


def build_inputs(binding, dtype, device):
    lead = tuple(binding["lead"])
    crow, col = _make_csr(lead, binding["rows"], binding["nnz"], dtype, device)
    nd = crow.dim() - 1
    total = crow.numel() // crow.shape[-1] * col.shape[-1]
    out = torch.empty((nd + 2, total), dtype=torch.int64, device=device)
    return (crow, col), {"out": out}


def key_shape(binding):
    lead = "x".join(str(d) for d in binding["lead"]) or "scalar"
    return f"lead{lead}_rows{binding['rows']}_nnz{binding['nnz']}"


def config(binding, dtype):
    lead = list(binding["lead"])
    rows, nnz = binding["rows"], binding["nnz"]
    dt = str(dtype)
    crow_shape = lead + [rows + 1]
    col_shape = lead + [nnz]
    blocks = 1
    for d in lead:
        blocks *= d
    total = blocks * nnz
    return {
        "inputs": {
            "args": [
                {"shape": crow_shape, "dtype": dt},
                {"shape": col_shape, "dtype": dt},
            ],
            "kwargs": {
                "out": {
                    "shape": [len(lead) + 2, total],
                    "dtype": "torch.int64",
                }
            },
        },
        "outputs": {"out": {"shape": [len(lead) + 2, total], "dtype": "torch.int64"}},
        "dims": {"lead": lead, "rows": rows, "nnz": nnz},
    }
