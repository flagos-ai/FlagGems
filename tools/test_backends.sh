#!/bin/bash

# test_backends.sh <VENDOR> <PR_ID>

echo $CHANGED_FILES
VENDOR=$1

# TODO(Qiming): Further simplify this
../setup.sh $VENDOR

# 3. Run tests
# TODO(Qiming): Handle CHANGED_FILES and other parameters
# TODO(Qiming): Run performance tests as well
# TODO(Qiming): Merge the following logic with run_tests.py
tools/test-op.sh $PR_ID
