# Dao matrix rows extracted from:
# https://github.com/Dao-AILab/fast-hadamard-transform
# Commit: e7706faf8d1c3b9f241e36860640ad1dac644ede
# Source: csrc/fast_hadamard_transform_special.h
# Copyright (c) 2023, Tri Dao.
#
# Upstream LICENSE (verbatim):
# BSD 3-Clause License
#
# Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from functools import lru_cache

import torch
import torch.nn.functional as F

# Output-row coefficients in increasing input-index order, from upstream only.
SPECIAL_HADAMARD_ROWS = {
    12: (
        "+-++++++++++",
        "--+-+-+-+-+-",
        "+++-++----++",
        "+---+--+-++-",
        "+++++-++----",
        "+-+---+--+-+",
        "++--+++-++--",
        "+--++---+--+",
        "++----+++-++",
        "+--+-++---+-",
        "++++----+++-",
        "+-+--+-++---",
    ),
    20: (
        "+----+----++--++-++-",
        "-+----+---+++---+-++",
        "--+----+---+++-+-+-+",
        "---+----+---+++++-+-",
        "----+----++--++-++-+",
        "-+++++-----+--+++--+",
        "+-+++-+---+-+--+++--",
        "++-++--+---+-+--+++-",
        "+++-+---+---+-+--+++",
        "++++-----++--+-+--++",
        "--++-+-++-+-----++++",
        "---++-+-++-+---+-+++",
        "+---++-+-+--+--++-++",
        "++---++-+----+-+++-+",
        "-++---++-+----+++++-",
        "-+--+--++-+----+----",
        "+-+-----++-+----+---",
        "-+-+-+---+--+----+--",
        "--+-+++------+----+-",
        "+--+--++------+----+",
    ),
    28: (
        "+------++----++-+--+-+--++--",
        "-+-----+++-----+-+--+-+--++-",
        "--+-----+++---+-+-+----+--++",
        "---+-----+++---+-+-+-+--+--+",
        "----+-----+++---+-+-+++--+--",
        "-----+-----++++--+-+--++--+-",
        "------++----++-+--+-+--++--+",
        "--++++-+-------++--+++-+--+-",
        "---++++-+-----+-++--+-+-+--+",
        "+---+++--+----++-++--+-+-+--",
        "++---++---+----++-++--+-+-+-",
        "+++---+----+----++-++--+-+-+",
        "++++--------+-+--++-++--+-+-",
        "-++++--------+++--++--+--+-+",
        "-+-++-++--++--+--------++++-",
        "+-+-++--+--++--+--------++++",
        "-+-+-++--+--++--+----+---+++",
        "+-+-+-++--+--+---+---++---++",
        "++-+-+-++--+------+--+++---+",
        "-++-+-+-++--+------+-++++---",
        "+-++-+---++--+------+-++++--",
        "-++--++-+-++-+++----++------",
        "+-++--++-+-++-+++-----+-----",
        "++-++---+-+-++-+++-----+----",
        "-++-++-+-+-+-+--+++-----+---",
        "--++-++++-+-+----+++-----+--",
        "+--++-+-++-+-+----+++-----+-",
        "++--++-+-++-+-+----++------+",
    ),
    40: (
        "+-------------------+-------------------",
        "++-++----+-+-++++--+++-++----+-+-++++--+",
        "+++-++----+-+-++++--+++-++----+-+-++++--",
        "+-++-++----+-+-++++-+-++-++----+-+-++++-",
        "+--++-++----+-+-+++++--++-++----+-+-++++",
        "++--++-++----+-+-+++++--++-++----+-+-+++",
        "+++--++-++----+-+-+++++--++-++----+-+-++",
        "++++--++-++----+-+-+++++--++-++----+-+-+",
        "+++++--++-++----+-+-+++++--++-++----+-+-",
        "+-++++--++-++----+-++-++++--++-++----+-+",
        "++-++++--++-++----+-++-++++--++-++----+-",
        "+-+-++++--++-++----++-+-++++--++-++----+",
        "++-+-++++--++-++----++-+-++++--++-++----",
        "+-+-+-++++--++-++---+-+-+-++++--++-++---",
        "+--+-+-++++--++-++--+--+-+-++++--++-++--",
        "+---+-+-++++--++-++-+---+-+-++++--++-++-",
        "+----+-+-++++--++-+++----+-+-++++--++-++",
        "++----+-+-++++--++-+++----+-+-++++--++-+",
        "+++----+-+-++++--++-+++----+-+-++++--++-",
        "+-++----+-+-++++--+++-++----+-+-++++--++",
        "+--------------------+++++++++++++++++++",
        "++-++----+-+-++++--+--+--++++-+-+----++-",
        "+++-++----+-+-++++-----+--++++-+-+----++",
        "+-++-++----+-+-++++--+--+--++++-+-+----+",
        "+--++-++----+-+-++++-++--+--++++-+-+----",
        "++--++-++----+-+-+++--++--+--++++-+-+---",
        "+++--++-++----+-+-++---++--+--++++-+-+--",
        "++++--++-++----+-+-+----++--+--++++-+-+-",
        "+++++--++-++----+-+------++--+--++++-+-+",
        "+-++++--++-++----+-+-+----++--+--++++-+-",
        "++-++++--++-++----+---+----++--+--++++-+",
        "+-+-++++--++-++----+-+-+----++--+--++++-",
        "++-+-++++--++-++------+-+----++--+--++++",
        "+-+-+-++++--++-++----+-+-+----++--+--+++",
        "+--+-+-++++--++-++---++-+-+----++--+--++",
        "+---+-+-++++--++-++--+++-+-+----++--+--+",
        "+----+-+-++++--++-++-++++-+-+----++--+--",
        "++----+-+-++++--++-+--++++-+-+----++--+-",
        "+++----+-+-++++--++----++++-+-+----++--+",
        "+-++----+-+-++++--++-+--++++-+-+----++--",
    ),
}


@lru_cache(maxsize=None)
def special_hadamard_matrix(M, device="cpu"):
    """Cache the small upstream matrix in fp32 on the requested device."""
    return torch.tensor(
        [
            [1 if sign == "+" else -1 for sign in row]
            for row in SPECIAL_HADAMARD_ROWS[M]
        ],
        dtype=torch.float32,
        device=device,
    )


def special_hadamard_reference(x, M, scale=1.0, transpose=False):
    """Apply H_M ⊗ Sylvester H_N, with implicit zero padding and crop.

    Dao first pads dim to a multiple of 4*M, then rounds dim/M to a power
    of two. The adjoint pads the cropped
    gradient, uses H_M transpose, and crops back to the input dimension.
    All arithmetic is fp32; only the scaled result is cast to the input dtype.
    """
    dim = x.shape[-1]
    aligned = ((dim + 4 * M - 1) // (4 * M)) * 4 * M
    n = 1 << ((aligned // M) - 1).bit_length()
    padded = M * n
    blocks = F.pad(x.float(), (0, padded - dim)).reshape(*x.shape[:-1], M, n)
    width = 1
    while width < n:
        pairs = blocks.reshape(*blocks.shape[:-1], n // (2 * width), 2, width)
        a, b = pairs.unbind(-2)
        blocks = torch.stack((a + b, a - b), dim=-2).reshape(blocks.shape)
        width *= 2
    matrix = special_hadamard_matrix(M, str(x.device))
    if transpose:
        matrix = matrix.T
    out = torch.matmul(matrix, blocks).reshape(*x.shape[:-1], padded)
    return (out[..., :dim] * scale).to(x.dtype)
