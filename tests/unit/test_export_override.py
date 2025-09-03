from worker.pipeline.pipeline_builder import PipelineBuilder


def test_export_format_override_applies_to_exporter_in_pipeline():
    pb = PipelineBuilder()
    pipeline = pb.build_pipeline()

    exporter = None
    for proc_id, inst in pipeline:
        if inst.__class__.__name__ == "Exporter":
            exporter = inst
            break

    assert exporter is not None, "Exporter processor not found in pipeline"
    assert exporter.default_format == "json"

    export_override = "pdf"

    updated = []
    for proc_id, inst in pipeline:
        if inst.__class__.__name__ == "Exporter":
            inst.default_format = export_override
        updated.append((proc_id, inst))

    exporter2 = None
    for proc_id, inst in updated:
        if inst.__class__.__name__ == "Exporter":
            exporter2 = inst
            break

    assert exporter2 is not None
    assert exporter2.default_format == "pdf"

