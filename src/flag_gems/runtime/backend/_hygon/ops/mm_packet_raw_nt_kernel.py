"""BF16 NT matmul with disjoint LDS windows and staggered compute waves."""
from flag_gems.utils import libentry
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna3 import mfma
from triton.experimental.gluon.language.hcu import structured_register_load, packet_buffer_store
from triton.experimental.gluon.language.hcu.raw_window import raw_window_owner, raw_window_copy, raw_window_read, raw_window_wait, raw_warp_group_id
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr
@builtin
def cta_barrier(_semantic=None):
    _semantic.builder.create_local_barrier()
@builtin
def mark_protocol(specialize=True, lgkm_cleanup=True, _semantic=None, _generator=None):
    specialize, lgkm_cleanup = _unwrap_if_constexpr(specialize), _unwrap_if_constexpr(lgkm_cleanup)
    assert type(specialize) is bool and type(lgkm_cleanup) is bool
    _generator.module.set_attr('ttg.hcu.raw_window_ws',_semantic.builder.get_int32_attr(1))
    _generator.module.set_attr('ttg.hcu.raw_window_ws.staggered',_semantic.builder.get_unit_attr())
    if not specialize:
        _generator.module.set_attr('ttg.hcu.raw_window_ws.disable_wave_specialization',_semantic.builder.get_unit_attr())
    if not lgkm_cleanup:
        _generator.module.set_attr('ttg.hcu.raw_window_ws.disable_lgkm_cleanup',_semantic.builder.get_unit_attr())


# One asm invocation covers every accumulator register in a compute thread.
# Tied values plus a memory clobber prevent high-K LDS reads from being hoisted
# while low-K MMAs still need their B bank. No data movement instruction.
ACC_CONSTRAINTS=gl.constexpr(','.join(['=v']*128+[str(i)for i in range(128)]+['~{memory}']))

@g.jit
def accumulator_fence(acc, PRIORITY:gl.constexpr):
    if PRIORITY == 1:
        acc=gl.inline_asm_elementwise('s_setprio 1',ACC_CONSTRAINTS,[acc],dtype=gl.float32,is_pure=False,pack=128)
    elif PRIORITY == 0:
        acc=gl.inline_asm_elementwise('s_setprio 0',ACC_CONSTRAINTS,[acc],dtype=gl.float32,is_pure=False,pack=128)
    else:
        acc=gl.inline_asm_elementwise('',ACC_CONSTRAINTS,[acc],dtype=gl.float32,is_pure=False,pack=128)
    return acc

@g.jit
def program_coordinates(M:gl.constexpr,N:gl.constexpr,GROUP_M:gl.constexpr):
    if GROUP_M == 0:
        return gl.program_id(0), gl.program_id(1)
    else:
        count_m: gl.constexpr = (M + 255) // 256
        count_n: gl.constexpr = (N + 255) // 256
        flat = gl.program_id(0) + gl.program_id(1) * count_m
        group_width: gl.constexpr = GROUP_M * count_n
        group = flat // group_width
        # Divide only by positive compile-time constants. Full and tail
        # groups form the same finite bijection, without a dynamic divisor.
        if count_m % GROUP_M == 0:
            return group * GROUP_M + flat % GROUP_M, (flat % group_width) // GROUP_M
        else:
            full_width: gl.constexpr = (count_m // GROUP_M) * group_width
            tail_m: gl.constexpr = count_m % GROUP_M
            full_m = group * GROUP_M + flat % GROUP_M
            full_n = (flat % group_width) // GROUP_M
            tail_flat = flat - full_width
            last_m = (count_m // GROUP_M) * GROUP_M + flat % tail_m
            last_n = tail_flat // tail_m
            return gl.where(flat < full_width, full_m, last_m), gl.where(flat < full_width, full_n, last_n)


@g.jit
def load_a64_parts(A,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,column,da:gl.constexpr,GROUP_M:gl.constexpr):
    pid_m,pid_n=program_coordinates(M,N,GROUP_M)
    full=structured_register_load(A,(M,K),(K,1),(pid_m*256,column,1),(256,64),da)
    # Splitting the middle binary K axis returns contiguous K0:32/K32:64.
    # Real layout lowering must prove this stays within each lane's registers.
    lo,hi=gl.split(full.reshape((256,2,32)).permute((0,2,1)))
    return gl.convert_layout(lo,da),gl.convert_layout(hi,da)

@g.jit
def producer_one(B,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,t,window,GROUP_M:gl.constexpr):
    pid_m,pid_n=program_coordinates(M,N,GROUP_M)
    basis:gl.constexpr=(1,2,4,8,16,32,256,512,1024,64,128,2048,4096,8192)
    token=raw_window_copy(B,window,N,pid_n*256,t*64,True,K,(256,64),basis)
    # Issue next-window DMA while compute consumes the other window. First
    # CTA closes previous compute phase; VM0 then second CTA publishes B_t.
    cta_barrier()
    raw_window_wait(token)
    cta_barrier()

@g.jit
def producer(B,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,owner,GROUP_M:gl.constexpr):
    for pair in range(K//128):
        producer_one(B,M,N,K,pair*2,owner.index(0),GROUP_M)
        producer_one(B,M,N,K,pair*2+1,owner.index(1),GROUP_M)
    if (K//64)%2:
        producer_one(B,M,N,K,(K//64)-1,owner.index(0),GROUP_M)
    # Both compute groups remain active through the terminal rendezvous.
    cta_barrier()
    cta_barrier()

@g.jit
def compute_one(A,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,t,window,acc,av0,av1,high,PRIO:gl.constexpr,FENCE:gl.constexpr,GLOBAL_WAIT:gl.constexpr,GROUP_M:gl.constexpr):
    mma:gl.constexpr=AMDMFMALayout(3,[16,16,16],True,[8,1],tiles_per_warp=[2,2],mmac_layout=1)
    da:gl.constexpr=gl.DotOperandLayout(0,mma,8)
    db:gl.constexpr=gl.DotOperandLayout(1,mma,8)
    basis:gl.constexpr=(1,2,4,8,16,32,256,512,1024,64,128,2048,4096,8192)
    bv0=raw_window_read(window,(32,256),db,(256,64),basis,(0,0),(1,0),full_wait=True,global_wait=GLOBAL_WAIT)
    if high:
        cta_barrier()
    if PRIO:
        acc=accumulator_fence(acc,1)
    acc=mfma(av0,bv0,acc)
    if FENCE or PRIO:
        acc=accumulator_fence(acc,0 if PRIO else -1)
    if not high:
        cta_barrier()
    next0,next1=load_a64_parts(A,M,N,K,(t+1)*64,da,GROUP_M)
    bv1=raw_window_read(window,(32,256),db,(256,64),basis,(0,32),(1,0),full_wait=True)
    if high:
        cta_barrier()
    if PRIO:
        acc=accumulator_fence(acc,1)
    acc=mfma(av1,bv1,acc)
    if FENCE or PRIO:
        acc=accumulator_fence(acc,0 if PRIO else -1)
    if not high:
        cta_barrier()
    return acc,next0,next1

@g.jit
def consumer(A,C,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,C_REBASE:gl.constexpr,FULL_C_TILE:gl.constexpr,STATIC_C_BOUNDS:gl.constexpr,owner,PRIO:gl.constexpr,FENCE:gl.constexpr,GLOBAL_WAIT:gl.constexpr,GROUP_M:gl.constexpr):
    pid_m,pid_n=program_coordinates(M,N,GROUP_M)
    mma:gl.constexpr=AMDMFMALayout(3,[16,16,16],True,[8,1],tiles_per_warp=[2,2],mmac_layout=1)
    da:gl.constexpr=gl.DotOperandLayout(0,mma,8)
    acc=gl.full((256,256),0,gl.float32,mma)
    av0,av1=load_a64_parts(A,M,N,K,0,da,GROUP_M)
    high=raw_warp_group_id(4)!=0
    cta_barrier()
    cta_barrier()
    for pair in range(K//128):
        acc,av0,av1=compute_one(A,M,N,K,pair*2,owner.index(0),acc,av0,av1,high,PRIO,FENCE,GLOBAL_WAIT,GROUP_M)
        acc,av0,av1=compute_one(A,M,N,K,pair*2+1,owner.index(1),acc,av0,av1,high,PRIO,FENCE,GLOBAL_WAIT,GROUP_M)
    if (K//64)%2:
        acc,av0,av1=compute_one(A,M,N,K,(K//64)-1,owner.index(0),acc,av0,av1,high,PRIO,FENCE,GLOBAL_WAIT,GROUP_M)
    m=pid_m*256+gl.arange(0,256,layout=gl.SliceLayout(1,mma))
    n=pid_n*256+gl.arange(0,256,layout=gl.SliceLayout(0,mma))
    if C_REBASE:
        origin=pid_m.to(gl.int64)*256*N+pid_n.to(gl.int64)*256
        cbase=C+origin
        local_m=gl.arange(0,256,layout=gl.SliceLayout(1,mma))
        local_n=gl.arange(0,256,layout=gl.SliceLayout(0,mma))
        offsets=local_m[:,None]*N+local_n[None,:]
    else:
        cbase=C
        offsets=m[:,None]*N+n[None,:]
    value=acc.to(gl.bfloat16)
    if FULL_C_TILE:
        # The launch grid is ceil(M/256) x ceil(N/256). Division avoids a
        # potentially overflowing tile-origin plus extent predicate.
        if (pid_m<M//256)&(pid_n<N//256):
            packet_buffer_store(value,cbase,offsets)
        else:
            packet_buffer_store(value,cbase,offsets,((STATIC_C_BOUNDS&(M%256==0))|(m[:,None]<M))&((STATIC_C_BOUNDS&(N%256==0))|(n[None,:]<N)))
    else:
        packet_buffer_store(value,cbase,offsets,((STATIC_C_BOUNDS&(M%256==0))|(m[:,None]<M))&((STATIC_C_BOUNDS&(N%256==0))|(n[None,:]<N)))


@libentry()
@g.jit
def _mm_packet_raw_nt_kernel(A,B,C,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,PRIO:gl.constexpr=True,FENCE:gl.constexpr=True,GLOBAL_WAIT:gl.constexpr=False,WAVE_SPECIALIZE:gl.constexpr=True,LGKM_CLEANUP:gl.constexpr=True,C_REBASE:gl.constexpr=True,FULL_C_TILE:gl.constexpr=False,STATIC_C_BOUNDS:gl.constexpr=False,GROUP_M:gl.constexpr=0):
    gl.static_assert(GROUP_M >= 0)
    gl.static_assert(K>=64 and K%64==0)
    gl.static_assert(M>0 and N>0 and M<2147483648 and N<2147483648)
    gl.static_assert(2*M*N<2147483648)
    owner=raw_window_owner(gl.bfloat16,2)
    mark_protocol(WAVE_SPECIALIZE,LGKM_CLEANUP)
    gl.warp_specialize([(consumer,(A,C,M,N,K,C_REBASE,FULL_C_TILE,STATIC_C_BOUNDS,owner,PRIO,FENCE,GLOBAL_WAIT,GROUP_M)),(producer,(B,M,N,K,owner,GROUP_M))],[4],[32])
    owner._keep_alive()
