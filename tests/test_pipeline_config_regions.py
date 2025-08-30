import importlib


def test_region_classifier_in_profiles():
    cfg = importlib.import_module("configs.pipeline_config")
    pipelines = cfg.pipeline_config

    for profile in ("default", "performance", "quality"):
        assert profile in pipelines
        procs = pipelines[profile]["processors"]
        ids = [p["id"] for p in procs]
        assert any("region_classifier.RegionClassifierProcessor" == i for i in ids)

