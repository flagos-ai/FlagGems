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

import flag_gems

from . import accuracy_utils as utils


def _function_from(module):
    def fn():
        pass

    fn.__module__ = module
    return fn


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("flag_gems.ops.add", "GEMS"),
        ("_test_vendor.ops.add", "GEMS_TEST_VENDOR"),
        (
            "flag_gems.runtime.backend._test_vendor.ops.add",
            "GEMS_TEST_VENDOR",
        ),
        ("_test_vendor.fused.nested.add", "GEMS_TEST_VENDOR"),
        ("_test_vendor_extra.ops.add", "GEMS"),
    ],
)
def test_gems_log_prefix(monkeypatch, module, expected):
    monkeypatch.setattr(flag_gems, "vendor_name", "test_vendor")
    assert utils.gems_log_prefix(_function_from(module)) == expected


def test_gems_log_logger():
    module = "_test_vendor.ops.nested.add"
    assert utils.gems_log_logger(_function_from(module)) == "flag_gems.ops.add"
