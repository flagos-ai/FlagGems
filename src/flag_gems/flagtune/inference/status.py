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

"""Human-readable status output, separate from structured stdout results."""

import os
import re
import sys
from datetime import datetime


def _clean(value):
    text = str(value).replace("\n", " ").replace("\r", " ")
    # Diagnostic exceptions can contain authenticated URLs. Never echo credentials
    # or query parameters (including signed download tokens).
    text = re.sub(r"(https?://)[^/\s@]+@", r"\1<redacted>@", text)
    return re.sub(r"(https?://[^\s?#]+)[?#][^\s]*", r"\1?<redacted>", text)


def exception_reason(exc):
    """Format explicit causes and unsuppressed contexts without retaining frames."""
    parts, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        parts.append(f"{type(exc).__name__}: {_clean(exc)}")
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return " caused_by=".join(parts)


def print_status(event, **fields):
    """Status reporting must not turn a successful operation into a failure."""
    try:
        details = " ".join(f"{key}={_clean(value)}" for key, value in fields.items())
        print(
            f"[FlagTune][{datetime.now():%Y-%m-%d %H:%M:%S}][PID={os.getpid()}] "
            f"{event}: {details}",
            file=sys.stderr,
            flush=True,
        )
    except (OSError, ValueError):
        # A closed reporting pipe must not trigger Cost Model fallback.
        pass
