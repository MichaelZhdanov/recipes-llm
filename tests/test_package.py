from importlib.metadata import version

import assistant


def test_version_comes_from_package_metadata() -> None:
    assert assistant.__version__
    assert assistant.__version__ == version("recipes-llm")
