#!/usr/bin/env python3
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

"""Check Python source for non-ASCII characters in code.

Non-ASCII characters in Python *code* (operators, numbers, identifiers) are
almost always IME artifacts — fullwidth punctuation (，), fullwidth digits
(１２３), fullwidth operators (＋－), ideographic spaces (U+3000), zero-width
characters — that the tokenizer either rejects outright or silently miscompiles.
A fullwidth comma in a function call, for example, makes Black's AST safety
check raise a ParseError that is hard to read.

Rules:
  1. Scan code tokens only — skip comments and strings, so Chinese prose in
     comments and docstrings is left untouched.
  2. Flag any codepoint > 127 appearing in a NAME, OP, or NUMBER token. These
     token categories must be pure ASCII; PEP 3131 Unicode identifiers are not
     used in this codebase (verified).

Exit codes:
  0 - no non-ASCII characters found in code
  1 - non-ASCII characters found in code
  2 - script internal error
"""

import io
import sys
import tokenize

# Token types that carry code; comments and strings are intentionally excluded
# so Chinese prose in docstrings/comments is not flagged.
CODE_TOKEN_TYPES = {
    tokenize.NAME,
    tokenize.OP,
    tokenize.NUMBER,
}


def check_file(path):
    """Return a list of (row, col, char) for non-ASCII chars in code."""
    errors = []
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except (OSError, UnicodeDecodeError) as exc:
        print(f"{path}: cannot read: {exc}", file=sys.stderr)
        return errors

    try:
        tokens = tokenize.generate_tokens(io.StringIO(src).readline)
        for tok_type, tok_str, (row, col), _, _ in tokens:
            if tok_type not in CODE_TOKEN_TYPES:
                continue
            for ch in tok_str:
                if ord(ch) > 127:
                    errors.append((row, col, ch))
                    break
    except tokenize.TokenError as exc:
        print(f"{path}: tokenize error: {exc}", file=sys.stderr)
        return errors
    return errors


def main():
    if len(sys.argv) < 2:
        print("usage: check_fullwidth_punct.py <file>...", file=sys.stderr)
        return 2

    failed = False
    for path in sys.argv[1:]:
        for row, col, ch in check_file(path):
            print(
                f"{path}:{row}:{col}: non-ASCII char U+{ord(ch):04X} '{ch}' "
                "in code (switch IME to ASCII)"
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
