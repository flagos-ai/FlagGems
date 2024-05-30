def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cuda",
        required=False,
        choices=["cuda", "cpu"],
        help="device to run reference tests on",
    )


def pytest_configure(config):
    value = config.getoption("--device")
    global TO_CPU
    TO_CPU = value == "cpu"
