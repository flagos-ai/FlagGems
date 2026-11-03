"""Exact grouped CTA ordering with only positive compile-time divisors."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def grouped_program_coordinates(pid, count_m: gl.constexpr,
                                count_n: gl.constexpr, GROUP_M: gl.constexpr):
    # The caller launches count_m * count_n consecutive program IDs.
    gl.static_assert(count_m > 0 and count_n > 0 and GROUP_M > 0)
    # Program IDs in the supported launch grid are nonnegative.
    pid = pid.to(gl.uint32)
    group_width: gl.constexpr = GROUP_M * count_n
    if count_n == 1:
        # Keep rotated tail-row ordering; the single column is always zero.
        if count_m > GROUP_M and count_m % GROUP_M != 0:
            first_tail: gl.constexpr = (count_m // GROUP_M) * GROUP_M
            tail_rows: gl.constexpr = count_m % GROUP_M
            pm = gl.where(pid >= first_tail, first_tail + pid % tail_rows, pid)
            return pm.to(gl.int32), gl.cast(0, gl.int32)
        return pid.to(gl.int32), gl.cast(0, gl.int32)
    elif count_m <= GROUP_M:
        return (pid % count_m).to(gl.int32), (pid // count_m).to(gl.int32)
    elif count_m % GROUP_M == 0:
        return ((pid // group_width) * GROUP_M + pid % GROUP_M).to(gl.int32), ((pid % group_width) // GROUP_M).to(gl.int32)
    else:
        full_groups: gl.constexpr = count_m // GROUP_M
        full_span: gl.constexpr = full_groups * group_width
        tail_m: gl.constexpr = count_m % GROUP_M
        in_tail = pid >= full_span
        full_m = (pid // group_width) * GROUP_M + pid % GROUP_M
        full_n = (pid % group_width) // GROUP_M
        # Preserve the original absolute modulo, including rotated tail rows.
        tail_m_index = full_groups * GROUP_M + pid % tail_m
        tail_n = (pid - full_span) // tail_m
        return gl.where(in_tail, tail_m_index, full_m).to(gl.int32), gl.where(in_tail, tail_n, full_n).to(gl.int32)
