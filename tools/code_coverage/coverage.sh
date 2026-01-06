#!/bin/bash

set -e
echo "  Usage:  $0  pull_request_id "

export PR_ID=$1

# For local test
if [ -z "$PR_ID" ]; then
  export PR_ID=168
fi
if [ -z "$GITHUB_SHA" ]; then
  export GITHUB_SHA=abcdefg
fi
ID_SHA="${PR_ID}-${GITHUB_SHA}"
echo ID_SHA $ID_SHA

rm -rf /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}/${PR_ID}-${GITHUB_SHA}-python-coverage-full
rm -rf /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}/${PR_ID}-${GITHUB_SHA}-python-coverage-diff
rm -rf /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}/${PR_ID}-${GITHUB_SHA}-python-coverage-diff-discard

PYTHON_BIN=/usr/bin/python3.11

export FlagGemsROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
echo FlagGemsROOT ${FlagGemsROOT}

FILES="/home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}/${ID_SHA}-op* /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}/${ID_SHA}-model*"
coverage combine -q --keep --data-file=${ID_SHA} $FILES
coverage report -m --data-file=${ID_SHA}
coverage xml -i --data-file=${ID_SHA} -o ${ID_SHA}-python-coverage.xml

${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/python_coverage.py > ${ID_SHA}-python-coverage.info

lcov --extract ${ID_SHA}-python-coverage.info \
    '*FlagGems*' \
    -o ${ID_SHA}-python-coverage-full.tmp \
    --rc lcov_branch_coverage=0
mv -f ${ID_SHA}-python-coverage-full.tmp ${ID_SHA}-python-coverage-full.info

genhtml -o ${ID_SHA}-python-coverage-full \
    -t 'Python Diff Coverage ${ID_SHA}' \
    --no-function-coverage \
    --no-branch-coverage \
    --ignore-errors source \
    ${ID_SHA}-python-coverage-full.info

# git
COVERAGE_DIFF_PATTERN="`${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/pull_request.py files ${PR_ID}`"
echo COVERAGE_DIFF_PATTERN ${COVERAGE_DIFF_PATTERN}
${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/pull_request.py diff ${PR_ID} > ${ID_SHA}-python-git-diff.out

# bash
lcov --extract ${ID_SHA}-python-coverage-full.info \
    ${COVERAGE_DIFF_PATTERN} \
    -o ${ID_SHA}-python-coverage-diff.info \
    --rc lcov_branch_coverage=0

if [ -s "${ID_SHA}-python-coverage-diff.info" ]; then
    echo "python-coverage-diff.info is NOT Empty"
else
    echo -e "==================== Python Coverage Result ====================\n"
    echo  "python-coverage-diff.info is Empty!"
    echo  "This means the files modified in your PR are not tested by python coverage!"
    echo  "Pass! Please check carefully if you need add test for your files!"
    echo -e "\n================================================================"
    mv -f ${ID_SHA}* /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
    exit
fi

${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/coverage_diff.py ${ID_SHA}-python-coverage-diff.info ${ID_SHA}-python-git-diff.out > ${ID_SHA}-python-coverage-diff.tmp
mv -f ${ID_SHA}-python-coverage-diff.tmp ${ID_SHA}-python-coverage-diff.info

genhtml -o ${ID_SHA}-python-coverage-diff \
    -t 'Python Diff Coverage' \
    --no-function-coverage \
    --no-branch-coverage \
    --ignore-errors source \
   ${ID_SHA}-python-coverage-diff.info

${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/jit_func_position.py ${COVERAGE_DIFF_PATTERN} > ${ID_SHA}-python-triton-jit-position.info
${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/coverage_diff_discard.py ${ID_SHA}-python-coverage-diff.info ${ID_SHA}-python-triton-jit-position.info > ${ID_SHA}-python-coverage-discard-diff.info

genhtml -o ${ID_SHA}-python-coverage-diff-discard \
    -t 'Python Diff Coverage Discard JIT' \
    --no-function-coverage \
    --no-branch-coverage \
    --ignore-errors source \
    ${ID_SHA}-python-coverage-discard-diff.info

mv -f ${ID_SHA}* /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}

lcov --list /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}/${ID_SHA}-python-coverage-discard-diff.info
echo -e "\n==================== Python Coverage Result ====================\n"
${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/coverage_lines.py  /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}/${ID_SHA}-python-coverage-discard-diff.info 0.9
