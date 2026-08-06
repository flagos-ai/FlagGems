# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from flag_gems.ops.split_with_sizes_copy import split_with_sizes_copy as _split_with_sizes_copy


def split_with_sizes_copy(input, split_sizes, dim=0):
    return _split_with_sizes_copy(input, split_sizes, dim)
