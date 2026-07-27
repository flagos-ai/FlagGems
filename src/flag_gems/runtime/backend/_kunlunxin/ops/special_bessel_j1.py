# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def special_bessel_j1_func(x):
    xf = x.to(tl.float32)
    ax = tl.abs(xf)
    z = ax * ax

    rp = (
        ((-8.999712257055594e8 * z + 4.5222829799819404e11) * z - 7.274942452218183e13)
        * z
        + 3.682957328638529e15
    )
    rq = z + 620.8364781180543
    rq = rq * z + 256987.25675774883
    rq = rq * z + 83514679.14319493
    rq = rq * z + 2.215115954797925e10
    rq = rq * z + 4.749141220799914e12
    rq = rq * z + 7.843696078762359e14
    rq = rq * z + 8.952223361846274e16
    rq = rq * z + 5.322786203326801e18
    small = ax * (z - 14.681970642123893) * (z - 49.218456321694604) * rp / rq

    w = 5.0 / ax
    zz = w * w
    pp = 7.621256162081731e-4
    pp = pp * zz + 7.313970569409176e-2
    pp = pp * zz + 1.1271960812968492
    pp = pp * zz + 5.112079511468077
    pp = pp * zz + 8.424045901417725
    pp = pp * zz + 5.214515986823615
    pp = pp * zz + 1.0
    pq = 5.713231280725487e-4
    pq = pq * zz + 6.884559087544954e-2
    pq = pq * zz + 1.105142326340617
    pq = pq * zz + 5.073863861286015
    pq = pq * zz + 8.399855543276042
    pq = pq * zz + 5.209828486823618
    pq = pq * zz + 1.0

    qp = 5.108625947501766e-2
    qp = qp * zz + 4.982138729512334
    qp = qp * zz + 75.82382841325453
    qp = qp * zz + 366.7796093601508
    qp = qp * zz + 710.8563049989261
    qp = qp * zz + 597.4896124006136
    qp = qp * zz + 211.68875710057214
    qp = qp * zz + 25.20702058580237
    qq = zz + 74.23732770356752
    qq = qq * zz + 1056.4488603826282
    qq = qq * zz + 4986.410583376536
    qq = qq * zz + 9562.318924047562
    qq = qq * zz + 7997.041604473507
    qq = qq * zz + 2826.192785176391
    qq = qq * zz + 336.0936078106983

    phase = ax - 2.356194490192345
    phase = phase - 6.283185307179586 * tl.floor(
        (phase + 3.141592653589793) / 6.283185307179586
    )
    large = (
        (pp / pq) * tl.cos(phase) - w * (qp / qq) * tl.sin(phase)
    ) * tl.sqrt(0.6366197723675814 / ax)

    result = tl.where(ax <= 5.0, small, large)
    result = tl.where(xf < 0.0, -result, result)
    return result.to(x.dtype)


def special_bessel_j1(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_BESSEL_J1")
    return special_bessel_j1_func(A)
