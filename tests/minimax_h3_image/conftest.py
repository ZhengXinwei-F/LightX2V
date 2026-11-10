"""CPU test setup for MiniMax-H3-Image.

Run this directory in its own pytest process so the CPU platform is selected
before ``lightx2v`` binds ``AI_DEVICE``::

    pytest tests/minimax_h3_image
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


import cpu_platform  # noqa: E402,F401
import pytest  # noqa: E402


def pytest_collection_modifyitems(config, items):
    from lightx2v_platform.base.global_var import AI_DEVICE

    if AI_DEVICE == "cpu":
        return
    reason = f"lightx2v was imported with AI_DEVICE={AI_DEVICE!r} before the CPU platform; run `pytest tests/minimax_h3_image` on its own"
    for item in items:
        if "minimax_h3_image" in str(item.fspath):
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def tiny_release(tmp_path_factory):
    from fixtures import build_tiny_release

    return build_tiny_release(str(tmp_path_factory.mktemp("h3img")))
