#!/bin/bash

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

set -e

if [ -z "$BASH_VERSION" ]; then
    echo "[ERROR] This script must be run using bash!" >&2
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

python3 "${SCRIPT_DIR}/generate_operators_doc.py" "$@"
