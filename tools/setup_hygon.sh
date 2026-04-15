#!/bin/bash

uv pip install --index ${FLAGOS_PYPI} \
    flagtree==0.5.0+hcu3.0 \
    torch==2.9.0+das.opt1.dtk2604
uv pip install -e .[hygon,test]

source /opt/dtk-26.04/env.sh
