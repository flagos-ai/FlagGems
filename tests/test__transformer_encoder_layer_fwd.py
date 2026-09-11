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

import flag_gems

from . import accuracy_utils as utils

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


@pytest.mark._transformer_encoder_layer_fwd
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("embed_dim", [64])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("use_gelu", [True, False])
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test__transformer_encoder_layer_fwd(
    batch_size, seq_len, embed_dim, num_heads, use_gelu, norm_first, dtype
):
    # Skip invalid configurations
    if embed_dim % num_heads != 0:
        pytest.skip("embed_dim must be divisible by num_heads")

    # Create input tensor
    src = torch.randn(
        batch_size, seq_len, embed_dim, dtype=dtype, device=flag_gems.device
    )

    # Create weights and biases
    qkv_weight = torch.randn(
        3 * embed_dim, embed_dim, dtype=dtype, device=flag_gems.device
    )
    qkv_bias = torch.randn(3 * embed_dim, dtype=dtype, device=flag_gems.device)
    proj_weight = torch.randn(
        embed_dim, embed_dim, dtype=dtype, device=flag_gems.device
    )
    proj_bias = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device)
    norm_weight_1 = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device)
    norm_bias_1 = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device)
    norm_weight_2 = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device)
    norm_bias_2 = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device)
    ffn_weight_1 = torch.randn(
        4 * embed_dim, embed_dim, dtype=dtype, device=flag_gems.device
    )
    ffn_bias_1 = torch.randn(4 * embed_dim, dtype=dtype, device=flag_gems.device)
    ffn_weight_2 = torch.randn(
        embed_dim, 4 * embed_dim, dtype=dtype, device=flag_gems.device
    )
    ffn_bias_2 = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device)

    eps = 1e-5

    # Convert to reference (CPU/higher precision)
    ref_src = utils.to_reference(src, True)
    ref_qkv_weight = utils.to_reference(qkv_weight, True)
    ref_qkv_bias = utils.to_reference(qkv_bias, True)
    ref_proj_weight = utils.to_reference(proj_weight, True)
    ref_proj_bias = utils.to_reference(proj_bias, True)
    ref_norm_weight_1 = utils.to_reference(norm_weight_1, True)
    ref_norm_bias_1 = utils.to_reference(norm_bias_1, True)
    ref_norm_weight_2 = utils.to_reference(norm_weight_2, True)
    ref_norm_bias_2 = utils.to_reference(norm_bias_2, True)
    ref_ffn_weight_1 = utils.to_reference(ffn_weight_1, True)
    ref_ffn_bias_1 = utils.to_reference(ffn_bias_1, True)
    ref_ffn_weight_2 = utils.to_reference(ffn_weight_2, True)
    ref_ffn_bias_2 = utils.to_reference(ffn_bias_2, True)

    # Call reference (PyTorch) implementation
    ref_out = torch._transformer_encoder_layer_fwd(
        ref_src,
        embed_dim,
        num_heads,
        ref_qkv_weight,
        ref_qkv_bias,
        ref_proj_weight,
        ref_proj_bias,
        use_gelu,
        norm_first,
        eps,
        ref_norm_weight_1,
        ref_norm_bias_1,
        ref_norm_weight_2,
        ref_norm_bias_2,
        ref_ffn_weight_1,
        ref_ffn_bias_1,
        ref_ffn_weight_2,
        ref_ffn_bias_2,
    )

    # Call FlagGems implementation
    res_out = flag_gems._transformer_encoder_layer_fwd(
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

    # Compare results with appropriate tolerance
    # Note: This is a complex composite operator with multiple operations,
    # so tolerance needs to be higher than simple pointwise ops
    if dtype == torch.float16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-1)
    elif dtype == torch.bfloat16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=2e-1)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-1)
