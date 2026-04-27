#!/bin/bash

set -e

if [ -z "$BASH_VERSION" ]; then
    echo "[ERROR]This script must be run using bash!" >&2
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
TEST_FILES="${SCRIPT_DIR}/../tests/test_*.py"
MD_FILE="${SCRIPT_DIR}/../OperatorList.md"

# all
grep 'pytest.mark.' ${TEST_FILES} |grep -v parametrize |grep -v 'skip(' |grep -v 'skipif(' |awk -F':@pytest.mark.' '{print $1"\t"$2}' |sort -k 2 |uniq >mark.txt

# unique
grep 'pytest.mark.' ${TEST_FILES} |grep -v parametrize |grep -v 'skip(' |grep -v 'skipif(' |awk -F':@pytest.mark.' '{print $2}' |sort |uniq |awk '!seen[$1]++' >mark.uniq.txt

# recorded in MD_FILE
grep '-' "${MD_FILE}" |grep -v '#' |grep -v 'N/A' |awk '{print $2}' |sort >mark.md.txt

MARK_NUM=`wc -l mark.txt |awk '{print $1}'`
MARK_UNIQ_NUM=`wc -l mark.uniq.txt |awk '{print $1}'`
MARK_MD_NUM=`wc -l mark.md.txt |awk '{print $1}'`
echo "Generated successfully: mark.txt (${MARK_NUM}), mark.uniq.txt (${MARK_UNIQ_NUM}), mark.md.txt (${MARK_MD_NUM})"

echo "-------- diff mark.uniq.txt mark.md.txt --------"
diff mark.uniq.txt mark.md.txt || true
echo "------------------------------------------------"

TEST_OP_FILES=`ls ${TEST_FILES} |grep "_ops.py" |grep -v "test_named_ops.py"`
EXCLUDED_MARKS=("pytest.mark.parametrize\(" \
                "pytest.mark.skip\(" \
                "pytest.mark.skipif\(" \
                "pytest.mark.xfail\(" \
                "pytest.mark.usefixtures\(" \
                "pytest.mark.filterwarnings\(" \
                "pytest.mark.timeout\(" \
                "pytest.mark.tryfirst\(" \
                "pytest.mark.trylast\(")

test_file_count=0
for file in ${TEST_OP_FILES}; do
    echo "Checking file: ${file}"
    set +e

    awk -v marks="${EXCLUDED_MARKS[*]}" '
    BEGIN {
        test_func = 0; decorated = 0; error = 0;
        split(marks, excluded_marks, " ")
    }

    /^@pytest\.mark\./ {
        test_func = 1
        excluded = 0
        for (i in excluded_marks) {
            if ($0 ~ excluded_marks[i]) {
                excluded = 1
                break
            }
        }
        if (excluded == 0) {
            decorated = 1
        }
        next
    }

    /^def / {
        if (test_func == 1) {
            if (decorated == 0) {
                print "[ERROR]"$0
                error = 1
            }
            test_func = 0
            decorated = 0
        }
    }

    END {
        if (error == 1) {
            exit 1
        }
    }
    ' "$file"

    if [ $? -ne 0 ]; then
        echo "[ERROR]There are some test_op_func without 'pytest.mark.{OP_NAME}' in ${file}"
        exit 1
    fi

    set -e
    test_file_count=$((test_file_count + 1))
done

echo "Finish checking ${test_file_count} files successfully."
