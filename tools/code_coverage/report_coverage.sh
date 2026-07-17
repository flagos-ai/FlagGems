#!/bin/bash

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
