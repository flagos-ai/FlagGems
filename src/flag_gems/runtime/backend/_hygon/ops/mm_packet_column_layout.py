"""Derive a within-32-column bit rotation from the compiler's actual linear layout."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.hcu import packet_read

@g.constexpr_function
def column_layout(layout, pack_registers=False):
    assert layout.rank == 2 and layout.shape[1] >= 32 and layout.shape[1] % 32 == 0
    def rotate(bases):
        return [[r, (n & ~31) | ((n & 15) << 1) | ((n >> 4) & 1)] for r, n in bases]
    registers = rotate(layout.reg_bases)
    if pack_registers and [0, 1] in registers:
        # A pure unit column basis is a pair owned by this same thread.
        # Move it to the least-significant register bit so vector stores see
        # adjacent BF16 values; lane/warp ownership and every value stay fixed.
        index = registers.index([0, 1])
        registers = [registers[index]] + registers[:index] + registers[index + 1:]
    return gl.DistributedLinearLayout(registers, rotate(layout.lane_bases),
                                     rotate(layout.warp_bases), rotate(layout.block_bases), layout.shape)

@g.jit
def read_b_column_chart(view, DB: gl.constexpr, BASIS: gl.constexpr,
                        BK: gl.constexpr, BN: gl.constexpr):
    physical: gl.constexpr = column_layout(gl.to_linear_layout(DB, [BK, BN]))
    values = packet_read(view, physical, BASIS)
    values = gl.reshape(values, [BK, BN // 32, 16, 2])
    values = gl.permute(values, [0, 1, 3, 2])
    return gl.convert_layout(gl.reshape(values, [BK, BN]), DB)
