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

"""
Test suite for the underscore marker auto-registration plugin.

This test verifies that the plugin in conftest.py correctly registers
underscore-prefixed markers from operators.yaml.
"""

import pytest


def test_underscore_markers_are_registered():
    """Test that underscore-prefixed markers are automatically registered."""
    # These operators are defined in operators.yaml with underscore prefixes
    underscore_operators = [
        "_adaptive_avg_pool3d_backward",
        "_flash_attention_forward",
        "_reshape_alias",
        "_weight_norm",
    ]

    for op_id in underscore_operators:
        # Verify the marker is accessible via pytest.mark
        assert hasattr(pytest.mark, op_id), f"Marker {op_id} should be registered"

        # Verify we can access it without AttributeError
        marker = getattr(pytest.mark, op_id)
        assert marker is not None, f"Marker {op_id} should not be None"


def test_underscore_marker_can_be_used():
    """Test that we can actually use an underscore-prefixed marker."""

    # This should not raise an AttributeError
    @pytest.mark._reshape_alias
    def dummy_test():
        pass

    # Verify the marker was applied
    markers = [m.name for m in dummy_test.pytestmark]
    assert "_reshape_alias" in markers, "Marker should be applied to function"
