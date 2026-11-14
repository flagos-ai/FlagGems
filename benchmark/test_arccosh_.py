import pytest
import torch

import flag_gems

from . import base, consts


class KlxArccoshInplaceBenchmark(base.UnaryPointwiseBenchmark):
    def build_inputs(self, case):
        if flag_gems.vendor_name == "kunlunxin" and case.dtype == torch.bfloat16:
            base.Config.skipped_case_ids.add(case.case_id)
            pytest.skip("klx native arccosh_ BF16 unsupported; master-batch03 native evidence")
        return super().build_inputs(case)


@pytest.mark.arccosh_
def test_arccosh_():
    bench = KlxArccoshInplaceBenchmark(
        op_name="arccosh_",
        torch_op=lambda a: a.arccosh_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
