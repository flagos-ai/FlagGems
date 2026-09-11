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

# Register marker for operator name starting with underscore
setattr(
    pytest.mark,
    "_transform_bias_rescale_qkv",
    MarkDecorator(
        Mark("_transform_bias_rescale_qkv", (), {}, _ispytest=True), _ispytest=True
    ),
)


def _transform_bias_rescale_qkv_input_fn(shape, dtype, device):
    """Generate inputs for _transform_bias_rescale_qkv benchmark.

    shape is expected to be (batch, seq_len, num_heads, head_dim)
    """
    # Only process 4D shapes (batch, seq_len, num_heads, head_dim)
    if len(shape) != 4:
        return

    batch, seq_len, num_heads, head_dim = shape
    hidden_dim = num_heads * head_dim
    total_dim = 3 * hidden_dim

    qkv = torch.randn(batch, seq_len, total_dim, dtype=dtype, device=device)
    bias = torch.randn(total_dim, dtype=dtype, device=device)

    yield qkv, bias, num_heads


class TransformBiasRescaleQKVBenchmark(base.GenericBenchmark):
    """Custom benchmark that preserves custom shapes."""

    def __init__(self, *args, custom_shapes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_shapes = custom_shapes or self.shapes

    def set_shapes(self, shape_file_path=None):
        """Override to preserve custom shapes."""
        # Don't load from file, use our custom shapes
        self.shapes = self._custom_shapes


@pytest.mark._transform_bias_rescale_qkv
def test__transform_bias_rescale_qkv():
    def torch_op(qkv, bias, num_heads):
        return torch._transform_bias_rescale_qkv(qkv, bias, num_heads)

    # Define test shapes as (batch, seq_len, num_heads, head_dim)
    custom_shapes = [
        (2, 128, 8, 64),
        (4, 256, 8, 64),
        (2, 512, 8, 64),
        (4, 1024, 8, 64),
        (2, 128, 12, 64),
        (4, 256, 12, 64),
    ]

    bench = TransformBiasRescaleQKVBenchmark(
        op_name="_transform_bias_rescale_qkv",
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=_transform_bias_rescale_qkv_input_fn,
        custom_shapes=custom_shapes,
        shape_desc="batch, seq_len, num_heads, head_dim",
    )

    bench.run()
