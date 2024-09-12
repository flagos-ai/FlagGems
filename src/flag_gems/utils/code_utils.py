# The code for IndentedBuffer is adapted from
# https://github.com/pytorch/pytorch/blob/ed48ea9997c2b04736096e4b6669543ab2e627d5/torch/_inductor/utils.py#L742
# The code for Namespace is adapted from
# https://github.com/pytorch/pytorch/blob/ed48ea9997c2b04736096e4b6669543ab2e627d5/torch/fx/graph.py#L115

# License from pytorch(https://github.com/pytorch/pytorch)

# From PyTorch:

# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# From Caffe2:

# Copyright (c) 2016-present, Facebook Inc. All rights reserved.

# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.

# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.

# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.

# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain

# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.

# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.

# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.

# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.

# All rights reserved.

import builtins
import contextlib
import keyword
import re
from collections import defaultdict
from io import StringIO
from typing import Dict, Set


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent=0):
        self._lines = []
        self._indent = initial_indent

    def getvalue(self) -> str:
        buf = StringIO()
        for line in self._lines:
            assert isinstance(line, str)
            buf.write(line)
            buf.write("\n")
        return buf.getvalue()

    def clear(self):
        self._lines.clear()

    def __bool__(self):
        return bool(self._lines)

    def prefix(self):
        return " " * (self._indent * self.tabwidth)

    def newline(self):
        self.writeline("\n")

    def writeline(self, line):
        if line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    def writemultiline(self, s):
        self.writelines(s.splitlines())

    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()


class NameSpace:
    def __init__(self):
        self._used_names: Set[str] = set()
        self._base_count: Dict[str, int] = defaultdict(int)

        self._illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")
        self._name_suffix_regex = re.compile(r"(.*)_(\d+)$")

    def create_name(self, candidate: str) -> str:
        """Create a unique name.

        Arguments:
            candidate: used as the basis for the unique name, relevant to the user.
        """
        # delete all characters that are illegal in a Python identifier
        candidate = self._illegal_char_regex.sub("_", candidate)

        if not candidate:
            candidate = "_unnamed"

        if candidate[0].isdigit():
            candidate = f"_{candidate}"

        match = self._name_suffix_regex.match(candidate)
        if match is None:
            base = candidate
            num = None
        else:
            base, num_str = match.group(1, 2)
            num = int(num_str)

        candidate = base if num is None else f"{base}_{num}"
        if not num:
            num = self._base_count[base]

        while candidate in self._used_names or self._is_illegal_name(candidate):
            num += 1
            candidate = f"{base}_{num}"

        self._used_names.add(candidate)
        self._base_count[base] = num
        return candidate

    def _is_illegal_name(self, name: str) -> bool:
        # 1. keywords are never allowed as names.
        if name in keyword.kwlist:
            return True

        # 2. Can't shadow a builtin name, unless you *are* that builtin.
        if name in builtins.__dict__:
            return True

        return False
