def pytest_addoption(parser):
    parser.addoption(
        "--vcc", action="store", default=None, help="Path to VCC training .h5ad file"
    )
    parser.addoption(
        "--scperturb", action="store", default=None, help="Path to scPerturb root."
    )
