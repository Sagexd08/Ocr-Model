from PIL import Image

from streamlit_demo.components.visualization import render_region_overlays, render_ocr_overlay

def test_render_region_overlays_empty():
    img = Image.new("RGB", (200, 200), color=(255,255,255))
    out = render_region_overlays(img, regions=[])
    assert out is not None
    assert out.size == img.size


def test_render_ocr_overlay_empty():
    img = Image.new("RGB", (200, 200), color=(255,255,255))
    res = {"tokens": []}
    out = render_ocr_overlay(img, res)
    assert out is not None
    assert out.size == img.size

