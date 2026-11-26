import pytest
import torch

import flag_gems

from . import base, consts, utils

# CSR inputs for the ``crow_indices`` benchmark: (nnz, shape) pairs. The op only
# hands back the internal row-pointer view (O(1) in nnz), so the sweep varies
# the metadata the harness touches while the values dtype follows the
# benchmark dtype sweep (crow/ccol stay int64).
CROW_INDICES_CASES = [
    (256, (64, 128)),
    (4096, (512, 512)),
    (65536, (2048, 2048)),
    (1048576, (4096, 8192)),
]


class CrowIndicesBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CROW_INDICES_CASES

    def get_input_iter(self, cur_dtype):
        # aten::crow_indices only returns the sparse tensor's internal
        # row-pointer storage; the values dtype follows the benchmark sweep
        # while crow/ccol are always int64. Yield a 1-tuple: the harness
        # unpacks each yielded item as an argument list.
        for nnz, shape in self.shapes:
            crow = torch.zeros(shape[0] + 1, dtype=torch.int64, device=self.device)
            counts = torch.bincount(
                torch.randint(0, shape[0], (nnz,), dtype=torch.int64, device="cpu"),
                minlength=shape[0],
            ).to(self.device)
            crow[1:] = counts.cumsum(0)
            ccol = torch.randint(
                0, shape[1], (nnz,), dtype=torch.int64, device=self.device
            )
            values = utils.generate_tensor_input((nnz,), cur_dtype, self.device)
            sparse = torch.sparse_csr_tensor(crow, ccol, values, shape)
            yield (sparse,)


@pytest.mark.crow_indices
def test_crow_indices():
    bench = CrowIndicesBenchmark(
        op_name="crow_indices",
        torch_op=torch.ops.aten.crow_indices,
        gems_op=flag_gems.crow_indices,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
