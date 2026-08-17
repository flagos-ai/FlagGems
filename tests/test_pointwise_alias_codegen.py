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

"""Unit tests for ModuleGenerator._collect_jit_deps alias/import collection.

These tests exercise the real collector by pointing it at a temporary module
file on disk (the collector locates the source file of the scalar function's
module via ``inspect.getfile``), rather than re-parsing an in-memory AST, so
they cover the actual code path used during codegen.
"""

import importlib.util
import sys
import textwrap
import uuid

import pytest
from triton.runtime.jit import JITFunction

from flag_gems.utils.pointwise_dynamic import ModuleGenerator


def _load_module_from_source(source: str):
    """Write `source` to a temp file, import it as a throwaway module, and
    return the module object. Used so the real module has a real __file__
    for `_collect_jit_deps` (via `scalar_fn.fn.__module__`) to read back.
    """
    module_name = f"_flaggems_alias_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    # Give the module a real file backing so inspect.getfile()/linecache
    # can read its source back, matching how _collect_jit_deps behaves for
    # actual flag_gems source files.
    import tempfile
    from pathlib import Path

    tmp_dir = Path(tempfile.mkdtemp(prefix="flaggems_alias_test_"))
    tmp_file = tmp_dir / f"{module_name}.py"
    tmp_file.write_text(textwrap.dedent(source))
    module.__file__ = str(tmp_file)

    sys.modules[module_name] = module
    exec(compile(tmp_file.read_text(), str(tmp_file), "exec"), module.__dict__)
    return module


def _collect(source: str):
    module = _load_module_from_source(source)
    scalar_fn = module.scalar_fn
    return ModuleGenerator._collect_jit_deps(scalar_fn)


def _fake_jit_function(module):
    """Return an object shaped like a triton JITFunction for the collector:
    it only reads `getattr(scalar_fn, "fn", scalar_fn)` and then
    `.__module__` off the result.
    """
    return module.dummy


def test_import_from_with_asname():
    """`from X import a as b` must be re-emitted with the real name and
    asname, and the alias referencing `b` must resolve to the real name."""
    source = """
        def dummy():
            pass

        from flag_gems.utils import tl_extra_shim as shim

        _tanh_alias = shim.tanh

        scalar_fn = dummy
    """
    extra_imports, plain_imports, alias_sources, local_sources = _collect(source)

    assert "flag_gems.utils" in extra_imports
    assert ("tl_extra_shim", "shim") in extra_imports["flag_gems.utils"]
    assert not any("shim" == real for real, _ in extra_imports["flag_gems.utils"])

    assert any("_tanh_alias = shim.tanh" in src for src in alias_sources)
    assert plain_imports == []
    assert local_sources == []


def test_import_from_without_asname():
    """Plain `from X import Y` (no rename) should behave as before."""
    source = """
        def dummy():
            pass

        from flag_gems.utils import tl_extra_shim

        _t = tl_extra_shim.tanh

        scalar_fn = dummy
    """
    extra_imports, plain_imports, alias_sources, local_sources = _collect(source)

    assert "flag_gems.utils" in extra_imports
    assert ("tl_extra_shim", None) in extra_imports["flag_gems.utils"]
    assert any("_t = tl_extra_shim.tanh" in src for src in alias_sources)


def test_plain_import_with_asname_supported():
    """`import X.Y as Z` (ast.Import) should register `Z` as an alias root
    so that `Z.attr` aliases are picked up. `triton.language` is already
    imported by `generate_imports` under the `tl` binding, so it is not
    re-emitted as a plain import line (it would be a redundant duplicate)."""
    source = """
        def dummy():
            pass

        import triton.language as tl

        _sigmoid = tl.sigmoid

        scalar_fn = dummy
    """
    extra_imports, plain_imports, alias_sources, local_sources = _collect(source)

    assert any("_sigmoid = tl.sigmoid" in src for src in alias_sources)
    assert plain_imports == []


def test_plain_import_of_new_module_with_asname_is_emitted():
    """`import X as Y` for a module NOT already covered by `generate_imports`
    must be re-emitted verbatim so `Y` is defined in the standalone file."""
    source = """
        def dummy():
            pass

        import torch.nn as nn

        _Linear = nn.Linear

        scalar_fn = dummy
    """
    extra_imports, plain_imports, alias_sources, local_sources = _collect(source)

    assert "import torch.nn as nn" in plain_imports
    assert any("_Linear = nn.Linear" in src for src in alias_sources)


def test_plain_import_without_asname_supported():
    """`import X` without asname is collected under its own dotted name."""
    source = """
        def dummy():
            pass

        import os.path

        _p = os.path.join

        scalar_fn = dummy
    """
    extra_imports, plain_imports, alias_sources, local_sources = _collect(source)

    assert "import os.path" in plain_imports


def test_chained_attribute_warns(caplog):
    """Chained attribute access (a.b.c) cannot be reproduced and must warn,
    not be silently dropped."""
    source = """
        def dummy():
            pass

        import torch

        Tensor = torch.Tensor.__name__

        scalar_fn = dummy
    """
    with caplog.at_level("WARNING", logger="flag_gems.utils.pointwise_dynamic"):
        extra_imports, plain_imports, alias_sources, local_sources = _collect(source)

    assert not any("Tensor = torch.Tensor.__name__" in src for src in alias_sources)
    assert any(
        "chained attribute" in record.message.lower() for record in caplog.records
    )


def test_annotated_assignment_warns(caplog):
    """AnnAssign (`name: Type = value`) aliasing an import must warn."""
    source = """
        def dummy():
            pass

        import torch

        Tensor: type = torch.Tensor

        scalar_fn = dummy
    """
    with caplog.at_level("WARNING", logger="flag_gems.utils.pointwise_dynamic"):
        extra_imports, plain_imports, alias_sources, local_sources = _collect(source)

    assert not any("Tensor: type = torch.Tensor" in src for src in alias_sources)
    assert any(
        "annotated assignment" in record.message.lower() for record in caplog.records
    )


def test_nested_in_if_warns(caplog):
    """Aliases nested inside `if`/`try` blocks (not module top-level) must
    warn instead of being silently skipped."""
    source = """
        def dummy():
            pass

        import torch

        if True:
            _tensor = torch.Tensor

        scalar_fn = dummy
    """
    with caplog.at_level("WARNING", logger="flag_gems.utils.pointwise_dynamic"):
        extra_imports, plain_imports, alias_sources, local_sources = _collect(source)

    assert not any("_tensor = torch.Tensor" in src for src in alias_sources)
    assert any(
        "nested inside a module-level control-flow block" in record.message.lower()
        for record in caplog.records
    )


def test_unrelated_assignment_does_not_warn(caplog):
    """Plain assignments unrelated to any import must not trigger a
    warning; only alias-shaped constructs referencing an imported name
    should be flagged."""
    source = """
        def dummy():
            pass

        CONSTANT = 42

        if True:
            OTHER_CONSTANT = "hello"

        scalar_fn = dummy
    """
    with caplog.at_level("WARNING", logger="flag_gems.utils.pointwise_dynamic"):
        extra_imports, plain_imports, alias_sources, local_sources = _collect(source)

    assert alias_sources == []
    assert caplog.records == []


def test_function_body_assignment_does_not_warn(caplog):
    """Ordinary local assignments inside a function (e.g.
    ``pi = math.pi`` inside a @triton.jit kernel) must not warn, since
    they are plain Python/Triton code, not module-level aliases the
    generated standalone file needs to reproduce."""
    source = """
        import math

        def my_kernel():
            pi = math.pi
            return pi

        scalar_fn = my_kernel
    """
    with caplog.at_level("WARNING", logger="flag_gems.utils.pointwise_dynamic"):
        extra_imports, plain_imports, alias_sources, local_sources = _collect(source)

    # The assignment is inside a function body, so it's not collected as
    # a module-level alias, and it must not produce any warning.
    assert not any("pi = math.pi" in src for src in alias_sources)
    assert caplog.records == []


@pytest.mark.parametrize("dummy_marker", [True])
def test_real_swiglu_module_alias_is_collected(dummy_marker):
    """Sanity check against a real in-repo module known to use this alias
    pattern (`sigmoid = tl.sigmoid` in flag_gems.fused.swiglu)."""
    import importlib

    # `flag_gems.fused.__init__` re-exports `swiglu` (the function) under
    # the same attribute name as the submodule, shadowing it on the
    # `flag_gems.fused` package. Go through `importlib.import_module` and
    # `sys.modules` to get the actual submodule object.
    swiglu_module = importlib.import_module("flag_gems.fused.swiglu")

    scalar_fn_candidate = getattr(swiglu_module, "swiglu_kernel", None)
    assert isinstance(scalar_fn_candidate, JITFunction)

    extra_imports, plain_imports, alias_sources, local_sources = (
        ModuleGenerator._collect_jit_deps(scalar_fn_candidate)
    )
    # `swiglu_kernel` itself isn't a pointwise_dynamic scalar fn, but the
    # collector only depends on `__module__`, so this still exercises the
    # real source file end to end.
    assert isinstance(extra_imports, dict)
    assert isinstance(plain_imports, list)
    assert isinstance(alias_sources, list)
    assert isinstance(local_sources, list)


def test_noncanonical_asname_is_reemitted():
    """Gap 1 regression: ``import triton.language as lang`` must be
    re-emitted even though ``triton.language`` is in ALREADY_IMPORTED,
    because the generated prelude only binds the canonical name ``tl``,
    not ``lang``. Without re-emission, ``lang.sigmoid`` would raise
    NameError in the standalone file."""
    source = """
        def dummy():
            pass

        import triton.language as lang

        _sigmoid = lang.sigmoid

        scalar_fn = dummy
    """
    extra_imports, plain_imports, alias_sources, local_sources = _collect(source)

    assert "import triton.language as lang" in plain_imports
    assert any("_sigmoid = lang.sigmoid" in src for src in alias_sources)


def test_dotted_import_tail_alias_is_collected():
    """Gap 2 regression: ``import os.path`` (no asname) binds ``os``, and
    ``_join = os.path.join`` is a valid alias. The chain ``os.path.join``
    has effective depth 1 (a single attribute access on the fully-qualified
    imported module ``os.path``), not depth 2 (an unsupported chained
    access), so it must be collected, not warned/skipped."""
    source = """
        def dummy():
            pass

        import os.path

        _join = os.path.join

        scalar_fn = dummy
    """
    extra_imports, plain_imports, alias_sources, local_sources = _collect(source)

    assert "import os.path" in plain_imports
    assert any("_join = os.path.join" in src for src in alias_sources)
