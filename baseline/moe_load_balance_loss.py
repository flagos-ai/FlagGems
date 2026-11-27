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

"""NVIDIA native baseline for `moe_load_balance_loss`.

Mirrors benchmark/test_moe_load_balance_loss.py (MoeLoadBalanceLossBenchmark, a
base.Benchmark subclass):
    op_name  = "moe_load_balance_loss" (masked=False; the masked=True
               parametrization records under "moe_load_balance_loss_mask")
    torch_op = _torch_reference (the module-local reference below)
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:moe_load_balance_loss (benchmark/core_shapes.yaml),
               shape_desc "T, N_e, K"

This operator has no ATen equivalent; the benchmark's baseline is an eager
PyTorch composition (softmax + topk + scatter_add), reproduced verbatim below.
The masked branch of that reference is kept so the signature matches, but with
masked=False the attention_mask is always None for this operator name.
"""

import torch

OP_NAME = "moe_load_balance_loss"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.softmax is unavailable on this torch build"

# yaml:moe_load_balance_loss -- shape_desc "T, N_e, K"
_SHAPES = [
    (256, 8, 2),
    (1024, 8, 2),
    (4096, 16, 2),
    (4096, 64, 4),
    (16384, 64, 4),
    (32768, 128, 8),
]


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


def _torch_reference(gate_logits, top_k, attention_mask):
    """benchmark::_torch_reference -- same-device eager PyTorch baseline."""
    num_tokens, num_experts = gate_logits.shape
    probabilities = torch.softmax(gate_logits.float(), dim=-1)
    selected_experts = torch.topk(gate_logits, top_k, dim=-1).indices

    if attention_mask is None:
        token_weights = torch.ones(
            num_tokens, dtype=torch.float32, device=gate_logits.device
        )
        valid = num_tokens
        probability_sum = probabilities.sum(dim=0)
    else:
        flat_mask = attention_mask.reshape(-1).float()
        token_weights = flat_mask
        valid = flat_mask.sum()
        probability_sum = (probabilities * flat_mask.unsqueeze(-1)).sum(dim=0)

    # Each top-k rank gets its own column; expand_as is a stride-0 view, which
    # avoids the materialization repeat_interleave would pay for.
    counts = (
        torch.zeros(num_experts, top_k, dtype=torch.float32, device=gate_logits.device)
        .scatter_add_(
            0,
            selected_experts,
            token_weights.unsqueeze(-1).expand_as(selected_experts),
        )
        .sum(dim=1)
    )
    return num_experts * torch.sum(counts * probability_sum) / (valid * valid)


def native():
    if _resolve("torch.softmax") is None:
        return None
    return _torch_reference


def grid():
    return [{"num_tokens": t, "num_experts": e, "top_k": k} for t, e, k in _SHAPES]


def build_inputs(binding, dtype, device):
    logits = torch.randn(
        binding["num_tokens"],
        binding["num_experts"],
        dtype=dtype,
        device=device,
    )
    # masked=False for this operator name, so no attention mask.
    return (logits, binding["top_k"], None), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["num_experts"], binding["top_k"]]


def config(binding, dtype):
    t, e, k = binding["num_tokens"], binding["num_experts"], binding["top_k"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [t, e], "dtype": dt},
                {"scalar": k},
                None,
            ],
            "kwargs": {},
        },
        "outputs": {"loss": {"shape": [], "dtype": "torch.float32"}},
        "dims": {"T": t, "N_e": e, "K": k},
    }
