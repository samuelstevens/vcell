def pytest_addoption(parser):
    parser.addoption(
        "--replogle", action="store", default=None, help="Path to replogle .h5ad file"
    )
