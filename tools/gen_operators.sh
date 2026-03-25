#!/bin/bash

set -e

if [ -z "$BASH_VERSION" ]; then
    echo "[ERROR]This script must be run using bash!" >&2
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

python3.11 "${SCRIPT_DIR}/generate_operators_doc.py" "$@"
