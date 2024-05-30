def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="cuda",
        required=False,
        choices=["cuda", "cpu"],
        help="record latency in cuda or cpu",
    )


def pytest_configure(config):
    value = config.getoption("--mode")
    global CPU_MODE
    CPU_MODE = value == "cpu"
