#!/bin/bash

set -e

echo "  Usage:  $0  pull_request_id "
export PR_ID=$1
echo "PR_ID: $PR_ID"

ID_SHA_ATTEMPT="${PR_ID}-${GITHUB_SHA}-${GITHUB_RUN_ATTEMPT}"
echo ID_SHA_ATTEMPT $ID_SHA_ATTEMPT

PYTHON_BIN=/usr/bin/python3.11

export FlagGemsROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
echo FlagGemsROOT ${FlagGemsROOT}

FILES="/PR_Coverage/PR_${PR_ID}_Coverage/${ID_SHA_ATTEMPT}/${ID_SHA_ATTEMPT}*"
coverage combine -q --keep --data-file=${ID_SHA_ATTEMPT} $FILES
coverage report -m --data-file=${ID_SHA_ATTEMPT}
coverage xml -i --data-file=${ID_SHA_ATTEMPT} -o ${ID_SHA_ATTEMPT}-python-coverage.xml

${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/python_coverage.py > ${ID_SHA_ATTEMPT}-python-coverage.info

lcov --extract ${ID_SHA_ATTEMPT}-python-coverage.info \
    '*FlagGems*' \
    -o ${ID_SHA_ATTEMPT}-python-coverage-full.tmp \
    --rc lcov_branch_coverage=0
mv -f ${ID_SHA_ATTEMPT}-python-coverage-full.tmp ${ID_SHA_ATTEMPT}-python-coverage-full.info

genhtml -o ${ID_SHA_ATTEMPT}-python-coverage-full \
    -t 'Python Diff Coverage ${ID_SHA_ATTEMPT}' \
    --no-function-coverage \
    --no-branch-coverage \
    --ignore-errors source \
    ${ID_SHA_ATTEMPT}-python-coverage-full.info

# git
COVERAGE_DIFF_PATTERN="`${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/pull_request.py files ${PR_ID}`"
echo COVERAGE_DIFF_PATTERN ${COVERAGE_DIFF_PATTERN}
${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/pull_request.py diff ${PR_ID} > ${ID_SHA_ATTEMPT}-python-git-diff.out

# bash
lcov --extract ${ID_SHA_ATTEMPT}-python-coverage-full.info \
    ${COVERAGE_DIFF_PATTERN} \
    -o ${ID_SHA_ATTEMPT}-python-coverage-diff.info \
    --rc lcov_branch_coverage=0

if [ -s "${ID_SHA_ATTEMPT}-python-coverage-diff.info" ]; then
    echo "python-coverage-diff.info is NOT Empty"
else
    echo -e "==================== Python Coverage Result ====================\n"
    echo  "python-coverage-diff.info is Empty!"
    echo  "This means the files modified in your PR are not tested by python coverage!"
    echo  "Pass! Please check carefully if you need add test for your files!"
    echo -e "\n================================================================"
    mv -f ${ID_SHA_ATTEMPT}* /PR_Coverage/PR_${PR_ID}_Coverage/${ID_SHA_ATTEMPT}
    exit
fi

${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/coverage_diff.py ${ID_SHA_ATTEMPT}-python-coverage-diff.info ${ID_SHA_ATTEMPT}-python-git-diff.out > ${ID_SHA_ATTEMPT}-python-coverage-diff.tmp
mv -f ${ID_SHA_ATTEMPT}-python-coverage-diff.tmp ${ID_SHA_ATTEMPT}-python-coverage-diff.info

genhtml -o ${ID_SHA_ATTEMPT}-python-coverage-diff \
    -t 'Python Diff Coverage' \
    --no-function-coverage \
    --no-branch-coverage \
    --ignore-errors source \
   ${ID_SHA_ATTEMPT}-python-coverage-diff.info

${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/jit_func_position.py ${COVERAGE_DIFF_PATTERN} > ${ID_SHA_ATTEMPT}-python-triton-jit-position.info
${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/coverage_diff_discard.py ${ID_SHA_ATTEMPT}-python-coverage-diff.info ${ID_SHA_ATTEMPT}-python-triton-jit-position.info > ${ID_SHA_ATTEMPT}-python-coverage-discard-diff.info

genhtml -o ${ID_SHA_ATTEMPT}-python-coverage-diff-discard \
    -t 'Python Diff Coverage Discard JIT' \
    --no-function-coverage \
    --no-branch-coverage \
    --ignore-errors source \
    ${ID_SHA_ATTEMPT}-python-coverage-discard-diff.info

mv -f ${ID_SHA_ATTEMPT}* /PR_Coverage/PR_${PR_ID}_Coverage/${ID_SHA_ATTEMPT}

lcov --list /PR_Coverage/PR_${PR_ID}_Coverage/${ID_SHA_ATTEMPT}/${ID_SHA_ATTEMPT}-python-coverage-discard-diff.info
echo -e "\n==================== Python Coverage Result ====================\n"
${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/coverage_lines.py  /PR_Coverage/PR_${PR_ID}_Coverage/${ID_SHA_ATTEMPT}/${ID_SHA_ATTEMPT}-python-coverage-discard-diff.info 0.9
