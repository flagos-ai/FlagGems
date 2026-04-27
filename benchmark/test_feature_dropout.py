import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmarkExcluse1D


def feature_dropout_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, 0.5, True


class FeatureDropoutBenchmark(GenericBenchmarkExcluse1D):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (8, 64),
            (16, 128),
            (32, 256),
            (4, 64, 32, 32),
            (8, 128, 16, 16),
            (16, 256, 8, 8),
            (2, 512, 14, 14),
            (4, 1024, 7, 7),
        ]

    def set_more_shapes(self):
        return None


@pytest.mark.feature_dropout
def test_feature_dropout():
    bench = FeatureDropoutBenchmark(
        input_fn=feature_dropout_input_fn,
        op_name="feature_dropout",
        torch_op=torch.feature_dropout,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.feature_dropout_
def test_feature_dropout_():
    bench = FeatureDropoutBenchmark(
        input_fn=feature_dropout_input_fn,
        op_name="feature_dropout_",
        torch_op=torch.feature_dropout_,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
