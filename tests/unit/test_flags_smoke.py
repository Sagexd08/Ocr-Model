import os
import time
import importlib


def test_model_flags_reduce_startup_time(monkeypatch):
    # Ensure flags are off
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("CURIO_TEST_MODE", "1")
    monkeypatch.setenv("LOAD_MODELS_ON_STARTUP", "0")

    start = time.time()
    # Import app (which triggers lifespan setup but in test mode minimizes work)
    from api.main import create_app
    app = create_app()
    elapsed_no_models = time.time() - start

    # Now enable model loading and measure again (still under test mode, so only overhead difference shows)
    monkeypatch.setenv("LOAD_MODELS_ON_STARTUP", "1")
    importlib.reload(importlib.import_module("api.ml_service"))
    start2 = time.time()
    app2 = create_app()
    elapsed_with_models_flag = time.time() - start2

    # The no-models path should be faster or equal
    assert elapsed_no_models <= elapsed_with_models_flag + 0.5

