# Copyright 2026, The FlagOS Contributors.
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

"""Check the real foreach test resolvers without importing an accelerator runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def _resolver(filename, names, **namespace):
    path = Path(__file__).resolve().parents[1] / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _original(*args):
    return "original"


def _candidate(*args):
    raise RuntimeError("injected candidate was called")


@pytest.mark.parametrize(
    "key,public,private",
    [
        ("_foreach_max", "foreach_max", "_foreach_max"),
        ("_foreach_add.List", "foreach_add_list", "_foreach_add_List"),
        ("_foreach_div_.Scalar", "foreach_div_scalar_", "_foreach_div__Scalar"),
    ],
)
def test_binary_resolver_uses_live_candidate(key, public, private):
    module = SimpleNamespace(**{private: _original})
    namespace = _resolver(
        "test_foreach_ops.py",
        {"_key_parts", "_marks_for", "_gems"},
        flag_gems=module,
        ALL_WRAPPERS={key: _original},
    )
    assert namespace["_gems"](key)() == "original"
    setattr(module, public, _candidate)
    with pytest.raises(RuntimeError, match="injected candidate was called"):
        namespace["_gems"](key)()
    delattr(module, public)
    setattr(module, private, _candidate)
    with pytest.raises(RuntimeError, match="injected candidate was called"):
        namespace["_gems"](key)()


@pytest.mark.parametrize("inplace", [False, True])
def test_unary_resolver_uses_live_candidate(inplace):
    suffix = "_" if inplace else ""
    module = SimpleNamespace(**{f"_foreach_sin{suffix}": _original})
    namespace = _resolver("test_foreach_unary.py", {"_gems_op"}, flag_gems=module)
    assert namespace["_gems_op"]("sin", inplace)() == "original"
    setattr(module, f"foreach_sin{suffix}", _candidate)
    with pytest.raises(RuntimeError, match="injected candidate was called"):
        namespace["_gems_op"]("sin", inplace)()
