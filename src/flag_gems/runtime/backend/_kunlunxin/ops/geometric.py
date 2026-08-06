# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from flag_gems.ops.geometric import geometric as _geometric
from flag_gems.ops.geometric import geometric_ as _geometric_


def geometric(input, p=0.5, *, generator=None):
    return _geometric(input, p, generator=generator)


def geometric_(input, p=0.5, *, generator=None):
    return _geometric_(input, p, generator=generator)
