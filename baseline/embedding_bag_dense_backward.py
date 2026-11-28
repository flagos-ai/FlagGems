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

"""NVIDIA native baseline for `embedding_bag_dense_backward`.

Mirrors benchmark/test_embedding_bag_dense_backward.py
(EmbeddingBagDenseBackwardBenchmark, a base.Benchmark subclass):
    op_name  = "embedding_bag_dense_backward"
    torch_op = torch.ops.aten._embedding_bag_dense_backward
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with EMBEDDING_BAG_BACKWARD_SHAPES --
(num_bags, embedding_dim, num_weights, samples_per_bag) -- so core_shapes.yaml
is not consulted.

offset2bag / bag_size / maximum_indices come from a real
torch.ops.aten._embedding_bag forward, which build_inputs runs outside the
timed region exactly as the benchmark's get_input_iter does. mode 0 is sum,
scale_grad_by_freq is False, per_sample_weights is None and padding_idx is -1.

Note the benchmark slices the offsets with [:num_bags] after building them with
arange(0, num_samples + 1, samples_per_bag), which is reproduced here.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "embedding_bag_dense_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._embedding_bag_dense_backward is unavailable on this build"
)

# test_embedding_bag_dense_backward.py::EMBEDDING_BAG_BACKWARD_SHAPES
# (num_bags, embedding_dim, num_weights, num_samples_per_bag_avg)
_SHAPES = [
    (8, 16, 50, 4),
    (16, 32, 100, 4),
    (32, 64, 100, 4),
    (64, 128, 200, 4),
    (128, 256, 500, 4),
    (256, 128, 500, 8),
    (512, 256, 1000, 8),
]

_MODE = 0  # sum
_SCALE_GRAD_BY_FREQ = False
_PADDING_IDX = -1


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


def native():
    op = _resolve("torch.ops.aten._embedding_bag_dense_backward")
    return op if callable(op) else None


def grid():
    return [
        {
            "num_bags": num_bags,
            "embedding_dim": embedding_dim,
            "num_weights": num_weights,
            "samples_per_bag": samples_per_bag,
        }
        for num_bags, embedding_dim, num_weights, samples_per_bag in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    num_bags = binding["num_bags"]
    embedding_dim = binding["embedding_dim"]
    num_weights = binding["num_weights"]
    samples_per_bag = binding["samples_per_bag"]
    num_samples = num_bags * samples_per_bag
    weight = torch.randn(num_weights, embedding_dim, dtype=dtype, device=device)
    indices = torch.randint(
        0, num_weights, (num_samples,), dtype=torch.long, device=device
    )
    offsets = torch.arange(
        0, num_samples + 1, samples_per_bag, dtype=torch.long, device=device
    )[:num_bags]
    (
        output,
        offset2bag,
        bag_size,
        maximum_indices,
    ) = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, _MODE, False, None, False, _PADDING_IDX
    )
    grad = torch.randn_like(output)
    return (
        grad,
        indices,
        offset2bag,
        bag_size,
        maximum_indices,
        num_weights,
        _SCALE_GRAD_BY_FREQ,
        _MODE,
        None,  # per_sample_weights
        _PADDING_IDX,
    ), {}


def key_shape(binding):
    return (
        f"bags{binding['num_bags']}_dim{binding['embedding_dim']}"
        f"_weights{binding['num_weights']}_spb{binding['samples_per_bag']}"
    )


def config(binding, dtype):
    num_bags = binding["num_bags"]
    embedding_dim = binding["embedding_dim"]
    num_weights = binding["num_weights"]
    samples_per_bag = binding["samples_per_bag"]
    num_samples = num_bags * samples_per_bag
    dt = str(dtype)
    i64 = "torch.int64"
    return {
        "inputs": {
            "args": [
                {"shape": [num_bags, embedding_dim], "dtype": dt},
                {"shape": [num_samples], "dtype": i64},
                {"shape": [num_samples], "dtype": i64},
                {"shape": [num_bags], "dtype": i64},
                {"dtype": i64, "note": "maximum_indices from the forward"},
                {"scalar": num_weights},
                {"scalar": _SCALE_GRAD_BY_FREQ},
                {"scalar": _MODE},
                None,
                {"scalar": _PADDING_IDX},
            ],
            "kwargs": {},
        },
        "outputs": {
            "grad_weight": {"shape": [num_weights, embedding_dim], "dtype": dt}
        },
        "dims": {
            "num_bags": num_bags,
            "embedding_dim": embedding_dim,
            "num_weights": num_weights,
            "samples_per_bag": samples_per_bag,
            "num_samples": num_samples,
            "mode": _MODE,
        },
    }
