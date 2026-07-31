#!/usr/bin/env bash
# (c) Copyright, 2026, FlagOS contributors
#
# Inject the vendor suffix into cpp/pyproject.toml before building the native
# C++ extension package. PEP 621 requires [project].name to be a static string,
# so per-vendor package names (flag-gems-cpp-cuda, ...-musa, ...) are produced
# by rewriting the placeholder at build time.
#
# Usage: tools/set_cpp_vendor.sh <vendor>
#   <vendor>  one of: cuda musa npu gcu ix (lowercase FLAGGEMS_BACKEND)
#
# Idempotent: re-running with a different vendor rewrites the name again.

set -euo pipefail

vendor="${1:-}"
if [ -z "${vendor}" ]; then
  echo "Usage: $0 <vendor>  (cuda|musa|npu|gcu|ix)" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pyproject="${script_dir}/../cpp/pyproject.toml"

# Replace either the __VENDOR__ placeholder or a previously-set vendor suffix.
# Use a temp file + mv so this works with both GNU and BSD/macOS sed.
tmp="$(mktemp)"
sed -E "s/^name = \"flag-gems-cpp-[A-Za-z0-9_]+\"/name = \"flag-gems-cpp-${vendor}\"/" \
  "${pyproject}" > "${tmp}"
mv "${tmp}" "${pyproject}"

echo "cpp/pyproject.toml name set to flag-gems-cpp-${vendor}"
