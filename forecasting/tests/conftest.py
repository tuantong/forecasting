import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))  # isort:skip


# for vscode debugging: https://stackoverflow.com/a/62563106/14121677
if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


def pytest_addoption(parser):
    parser.addoption("--forecast_data_test_dir", action="store")


@pytest.fixture(scope="session")
def forecast_data_test_dir(request):
    name_value = request.config.option.forecast_data_test_dir
    if name_value is None:
        pytest.skip()
    return name_value
