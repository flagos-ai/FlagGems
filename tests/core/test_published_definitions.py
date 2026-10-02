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

"""Published metadata stays portable and colocated with its source tests."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = sorted((ROOT / "definitions").glob("*.json"))


@pytest.mark.parametrize("path", DEFINITIONS, ids=lambda path: path.stem)
def test_published_definition_contract(path):
    definition = json.loads(path.read_text(encoding="utf-8"))
    assert definition["api_version"] == "v6.0"
    assert definition["name"] == path.stem
    assert isinstance(definition["description"], str)
    parameters = definition["parameters"]
    assert len({item["name"] for item in parameters}) == len(parameters)
    for item in parameters:
        assert isinstance(item["required"], bool)
        if item["required"]:
            assert "default" not in item
    # Void-returning APIs legitimately have no logical outputs.
    assert isinstance(definition["outputs"], list)
    assert (ROOT / "tests" / f"test_{path.stem}.py").is_file()
    assert (ROOT / "benchmark" / f"test_{path.stem}.py").is_file()


def test_definition_collection_is_not_empty():
    assert DEFINITIONS
