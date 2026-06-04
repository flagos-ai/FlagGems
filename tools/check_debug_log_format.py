#!/usr/bin/env python3
"""Pre-commit hook to check debug log format in backend ops.

Ensures that logger.debug calls in backend op files follow the
GEMS_{VENDOR} {OP_NAME} pattern, where VENDOR is derived from the
directory name (e.g., _ascend -> ASCEND).
"""

import re
import sys
from pathlib import Path

BACKEND_OPS_PATTERN = re.compile(r"src/flag_gems/runtime/backend/_([^/]+)/ops/.*\.py$")
DEBUG_LOG_PATTERN = re.compile(r'logger\.debug\(["\']GEMS (.+?)["\']')


def check_file(filepath: str) -> list[str]:
    match = BACKEND_OPS_PATTERN.search(filepath)
    if not match:
        return []

    vendor = match.group(1).upper()
    expected_prefix = f"GEMS_{vendor} "
    errors = []

    try:
        lines = Path(filepath).read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return []

    for lineno, line in enumerate(lines, start=1):
        m = DEBUG_LOG_PATTERN.search(line)
        if not m:
            continue
        log_content = "GEMS " + m.group(1)
        if not log_content.startswith(expected_prefix):
            errors.append(
                f"{filepath}:{lineno}: expected '{expected_prefix}...' "
                f"but found '{log_content}'"
            )

    return errors


def main() -> int:
    errors = []
    for filepath in sys.argv[1:]:
        errors.extend(check_file(filepath))

    if errors:
        print("Debug log format errors found:")
        for err in errors:
            print(f"  {err}")
        print('\nExpected format: logger.debug("GEMS_{VENDOR} {OP_NAME}")')
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
