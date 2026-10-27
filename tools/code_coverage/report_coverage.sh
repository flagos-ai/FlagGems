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

export PR_ID=$1

ID_SHA="${PR_ID}-${GITHUB_SHA}"

if [ -e /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}/${PR_ID}-${GITHUB_SHA}-python-coverage-full ]; then
    echo "[+] Full Python Coverage Report: "
    echo "http://120.92.44.177/PR${PR_ID}/${ID_SHA}/${PR_ID}-${GITHUB_SHA}-python-coverage-full"
elif [ -e /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}/${PR_ID}-${GITHUB_SHA}-python-coverage-diff ]; then
    echo "[+] Python Coverage Report Only With PR Code Change: "
    echo "http://120.92.44.177/PR${PR_ID}/${ID_SHA}/${PR_ID}-${GITHUB_SHA}-python-coverage-diff"
elif [ -e /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}/${PR_ID}-${GITHUB_SHA}-python-coverage-diff-discard ]; then
    echo "[+] Python Coverage Report With PR Code Change But Without Triton JIT Functions: (> 90% required.)"
    echo "http://120.92.44.177/PR${PR_ID}/${ID_SHA}/${PR_ID}-${GITHUB_SHA}-python-coverage-diff-discard"
fi
