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

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

from . import base, consts

# ``_transformer_encoder_layer_fwd`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._transformer_encoder_layer_fwd``
# and ``-m _transformer_encoder_layer_fwd`` both work.
setattr(
    pytest.mark,
    "_transformer_encoder_layer_fwd",
    MarkDecorator(
        Mark("_transformer_encoder_layer_fwd", (), {}, _ispytest=True), _ispytest=True
    ),
)


def input_fn(shape, dtype, device):
    """Generate inputs for transformer encoder layer benchmark.

    Args:
        shape: Tuple specifying tensor size (used as reference for dimensions)
        dtype: Data type for tensors
        device: Device to create tensors on
    """
    # Use fixed configuration for transformer encoder layer
    # These are typical model sizes
    batch_size = 8
    seq_len = 128
    embed_dim = 512
    num_heads = 8

    # Input source tensor
    src = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device)

    # QKV projection weights
    qkv_weight = torch.randn(3 * embed_dim, embed_dim, dtype=dtype, device=device)
    qkv_bias = torch.randn(3 * embed_dim, dtype=dtype, device=device)

    # Output projection weights
    proj_weight = torch.randn(embed_dim, embed_dim, dtype=dtype, device=device)
    proj_bias = torch.randn(embed_dim, dtype=dtype, device=device)

    # Layer norm weights (first)
    norm_weight_1 = torch.randn(embed_dim, dtype=dtype, device=device)
    norm_bias_1 = torch.randn(embed_dim, dtype=dtype, device=device)

    # Layer norm weights (second)
    norm_weight_2 = torch.randn(embed_dim, dtype=dtype, device=device)
    norm_bias_2 = torch.randn(embed_dim, dtype=dtype, device=device)

    # FFN weights (4x expansion factor)
    ffn_weight_1 = torch.randn(4 * embed_dim, embed_dim, dtype=dtype, device=device)
    ffn_bias_1 = torch.randn(4 * embed_dim, dtype=dtype, device=device)
    ffn_weight_2 = torch.randn(embed_dim, 4 * embed_dim, dtype=dtype, device=device)
    ffn_bias_2 = torch.randn(embed_dim, dtype=dtype, device=device)

    # Configuration parameters
    use_gelu = True
    norm_first = False
    eps = 1e-5

    yield (
        src,
        embed_dim,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        use_gelu,
        norm_first,
        eps,
        norm_weight_1,
        norm_bias_1,
        norm_weight_2,
        norm_bias_2,
        ffn_weight_1,
        ffn_bias_1,
        ffn_weight_2,
        ffn_bias_2,
    )


@pytest.mark._transformer_encoder_layer_fwd
def test__transformer_encoder_layer_fwd():
    bench = base.GenericBenchmark(
        op_name="_transformer_encoder_layer_fwd",
        input_fn=input_fn,
        torch_op=torch._transformer_encoder_layer_fwd,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
