import pytest
import torch


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda:0")
