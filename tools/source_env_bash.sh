# Source a vendor env script under a real bash and print the env vars it
#!/bin/bash
# added or changed as `export KEY=VALUE` lines.
#
# Some vendor scripts (e.g. the hygon DTK env.sh) locate themselves via the
# bash-only ${BASH_SOURCE[0]}, which is empty when sourced from a non-bash
# shell (zsh, dash, ...) and silently mis-derives paths. Running them here
# under bash and re-exporting only the diff makes activation work the same
# way regardless of which shell sources .venv/bin/activate.
#
# Usage: bash tools/source_env_bash.sh <script-to-source>
# Output (on stdout): export KEY='VALUE' lines, one per changed/added var.

set -euo pipefail

script="$1"

before_file="$(mktemp)"
after_file="$(mktemp)"
trap 'rm -f "${before_file}" "${after_file}"' EXIT

env -0 >"${before_file}"
# Vendor scripts may reference variables (e.g. CMAKE_PREFIX_PATH) that are
# normally pre-defined by profile scripts in a login shell. Relax -eu while
# sourcing so an unset-variable reference doesn't kill this whole helper.
set +eu
source "${script}" >/dev/null 2>&1
set -eu
env -0 >"${after_file}"

python3 - "${before_file}" "${after_file}" <<'PY'
import shlex
import sys

def parse(path):
    env = {}
    with open(path, "rb") as f:
        dump = f.read()
    for entry in dump.split(b"\0"):
        if not entry:
            continue
        key, _, value = entry.partition(b"=")
        env[key.decode()] = value.decode()
    return env

before = parse(sys.argv[1])
after = parse(sys.argv[2])

for key, value in after.items():
    if before.get(key) != value:
        print(f"export {key}={shlex.quote(value)}")
PY
