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

import ctypes

import pytest
import torch

import flag_gems

from . import test_utils as tu

# aten::_print(str s) -> () takes no tensor operand and has no optional parameter, so
# the value-range/shape grid, the broadcast pair and the backward check have no
# expression here: each needs a tensor, and the native schema rejects one ("Expected a
# value of type 'str' for argument 's' but instead found type 'Tensor'"). The payload
# is the only workload dimension, so content classes and lengths stand in for dtypes
# and shapes. The probed native contract is that the payload reaches stdout verbatim
# followed by exactly one "\n"; any other value, argument count or keyword raises.

CALL_FORMS = ["positional", "keyword"]

DEFAULT_MESSAGE_CASES = [
    pytest.param("", id="empty"),
    pytest.param(" ", id="space"),
    pytest.param(" " * 8, id="spaces_8"),
    pytest.param("\t", id="tab"),
    pytest.param(" \t ", id="tab_and_spaces"),
    pytest.param("\n", id="newline_only"),
    pytest.param("\n\n", id="two_newlines"),
    pytest.param("line1\nline2", id="embedded_newline"),
    pytest.param("end\n", id="trailing_newline"),
    pytest.param("\nstart", id="leading_newline"),
    pytest.param("a\r\nb", id="crlf"),
    pytest.param("a\rb", id="carriage_return"),
    pytest.param("a\x00b", id="interior_nul"),
    pytest.param("\x07\x1b[31mred\x1b[0m", id="control_and_ansi"),
    pytest.param('he said "hi"', id="quotes"),
    pytest.param("C:\\tmp\\payload", id="backslashes"),
    pytest.param("{0} %s %%s", id="format_like"),
    pytest.param("flag_gems _print payload", id="ascii_word"),
    pytest.param("[INFO] _print payload", id="log_like"),
    pytest.param('{"op": "_print", "chars": 21}', id="json_like"),
    pytest.param("/tmp/flag_gems/print.txt", id="path_like"),
    pytest.param("trailing   ", id="trailing_spaces"),
    pytest.param("   leading", id="leading_spaces"),
    pytest.param("\t\t\t", id="tabs_only"),
    pytest.param("你好，世界", id="cjk"),
    pytest.param("🚀 rocket", id="emoji"),
    pytest.param("éàüñ", id="accents"),
    # Decomposed "e" + U+0301, not the precomposed single code point U+00E9.
    pytest.param("e\u0301", id="combining_mark"),
    pytest.param("שלום", id="rtl"),
    pytest.param("abc 你好 123", id="mixed_unicode_ascii"),
    pytest.param("a", id="len_1"),
    pytest.param("b" * 2, id="len_2"),
    pytest.param("c" * 15, id="len_15"),
    pytest.param("d" * 16, id="len_16"),
    pytest.param("e" * 31, id="len_31"),
    pytest.param("f" * 32, id="len_32"),
    pytest.param("g" * 63, id="len_63"),
    pytest.param("h" * 64, id="len_64"),
    pytest.param("i" * 255, id="len_255"),
    pytest.param("j" * 256, id="len_256"),
    pytest.param("k" * 1023, id="len_1023"),
    pytest.param("l" * 1024, id="len_1024"),
    pytest.param("m" * 4096, id="len_4096"),
    pytest.param("n" * 65536, id="len_65536"),
    pytest.param("o" * 1048576, id="len_1048576"),
    pytest.param("print line\n" * 32, id="multiline_block"),
    pytest.param("-" * 64, id="dashes_64"),
    pytest.param("x" * 200, id="long_word"),
    pytest.param("w " * 256, id="words_repeated"),
    # The native dispatcher coerces bytes/bytearray and writes them raw.
    pytest.param(b"bytes payload", id="bytes_ascii"),
    pytest.param(b"", id="bytes_empty"),
    pytest.param(b"bytes\nline", id="bytes_newline"),
    pytest.param(bytearray(b"bytearray payload"), id="bytearray_ascii"),
]

# Quick keeps every accepted argument form (str, bytes, bytearray) and every cheap
# content boundary; only the large payloads are default-only, because emitting them
# is a cost concern and not a semantic one.
QUICK_MESSAGE_CASES = [
    case
    for case in DEFAULT_MESSAGE_CASES
    if case.id not in ("len_65536", "len_1048576")
]


INVALID_LITERALS = {
    "int": 1,
    "float": 1.0,
    "bool": True,
    "none": None,
    "list": ["msg"],
    "tuple": ("msg",),
    "dict": {"msg": 1},
    "set": {"msg"},
}

INVALID_VALUE_KINDS = [
    pytest.param("int", id="int"),
    pytest.param("float", id="float"),
    pytest.param("bool", id="bool"),
    pytest.param("none", id="none"),
    pytest.param("list", id="list"),
    pytest.param("tuple", id="tuple"),
    pytest.param("dict", id="dict"),
    pytest.param("set", id="set"),
    pytest.param("tensor_1d", id="tensor_1d"),
    pytest.param("tensor_0d", id="tensor_0d"),
]

INVALID_CALL_FORMS = [
    pytest.param((), {}, id="no_argument"),
    pytest.param(("a", "b"), {}, id="two_arguments"),
    pytest.param((), {"unknown": "a"}, id="unknown_keyword"),
]

# The negative rows stay in both modes: an argument the schema cannot accept is
# rejected regardless of payload size.
MESSAGE_CASES = tu.selected_cases(DEFAULT_MESSAGE_CASES, quick=QUICK_MESSAGE_CASES)


def _invoke(op, message, form):
    return op(s=message) if form == "keyword" else op(message)


def _payload_bytes(message):
    return message.encode("utf-8") if isinstance(message, str) else bytes(message)


def _invalid_value(kind):
    # The operator launches no kernel, so device resolution stays in the test body
    # rather than running while the module is imported.
    if kind == "tensor_1d":
        return torch.zeros(2, dtype=torch.float32, device=flag_gems.device)
    if kind == "tensor_0d":
        return torch.zeros((), dtype=torch.float32, device=flag_gems.device)
    return INVALID_LITERALS[kind]


def _capture(op, message, form, capfdbinary):
    """Return ``(result, bytes written to stdout)`` for one call.

    The printer writes through the C stdout stream, whose buffer pytest's
    file-descriptor capture cannot see; a payload without a trailing newline would
    otherwise stay in that buffer forever, so the C stream is flushed to the
    captured descriptor before the bytes are read back.
    """
    capfdbinary.readouterr()  # discard anything left over from the previous call
    result = _invoke(op, message, form)
    ctypes.CDLL(None).fflush(None)
    return result, capfdbinary.readouterr().out


@pytest.mark.print
@pytest.mark.parametrize("form", CALL_FORMS)
@pytest.mark.parametrize("message", MESSAGE_CASES)
def test__print_writes_payload_to_stdout(message, form, capfdbinary):
    expected = _payload_bytes(message) + b"\n"
    _, ref_bytes = _capture(torch.ops.aten._print, message, form, capfdbinary)
    # Check the measurement before comparing anything, otherwise a broken capture
    # would compare two empty buffers and pass.
    assert ref_bytes == expected

    res, res_bytes = _capture(flag_gems._print, message, form, capfdbinary)
    assert res is None
    assert res_bytes == ref_bytes


@pytest.mark.print
@pytest.mark.parametrize("form", CALL_FORMS)
@pytest.mark.parametrize("kind", INVALID_VALUE_KINDS)
def test__print_rejects_non_string(kind, form):
    with pytest.raises((RuntimeError, TypeError)):
        _invoke(flag_gems._print, _invalid_value(kind), form)


@pytest.mark.print
@pytest.mark.parametrize("args,kwargs", INVALID_CALL_FORMS)
def test__print_rejects_invalid_call_form(args, kwargs):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._print(*args, **kwargs)
