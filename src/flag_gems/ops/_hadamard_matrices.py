"""Sign rows for the order-12, order-20, order-28, and order-40 transforms.

Source: https://github.com/Dao-AILab/fast-hadamard-transform
Revision: ee7706faf8d1c3b9f241e36860640ad1dac644ede
File: csrc/fast_hadamard_transform_special.h, hadamard_mult_thread_{12,20,28,40}.
Author (upstream AUTHORS): Tri Dao, tri@tridao.me.

Each string is one output row, in upstream input-column order; '+' means +1
and '-' means -1. All four matrices satisfy H @ H.T = order * I. Only the
order-12 matrix is symmetric; the others require a transpose for backward.
"""

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
