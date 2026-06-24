import importlib.util
from pathlib import Path

from sqlalchemy.engine import make_url


def test_test_database_url_keeps_real_password():
    spec = importlib.util.spec_from_file_location(
        "project_conftest", Path(__file__).with_name("conftest.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    url = make_url(module.TEST_DB_URL)

    assert url.database.endswith("_test")
    assert url.password != "***"
    assert url.password
