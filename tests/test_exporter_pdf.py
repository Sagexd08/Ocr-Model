from pathlib import Path
from PIL import Image

from worker.pipeline.processors.exporter import Exporter, ExportFormat
from worker.types import Document, Page, Token, Bbox, Region


def test_exporter_pdf_minimal(tmp_path):
    # Build a tiny document with one page, token, and region
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    img_path = tmp_path / "page.png"
    img.save(img_path)

    page = Page(
        page_num=1,
        width=200,
        height=100,
        image=str(img_path),
        tokens=[Token(text="Hello", bbox=Bbox(x1=10, y1=10, x2=60, y2=30), confidence=0.9)],
        regions=[Region(type="header", bbox=Bbox(x1=5, y1=5, x2=195, y2=35), confidence=0.8)],
    )
    doc = Document(pages=[page])

    exporter = Exporter({"output_dir": str(tmp_path), "formats": {"pdf": {"format_type": "pdf"}}})
    path = exporter._export_pdf(doc, "testdoc", ExportFormat(format_type="pdf"))

    assert path
    assert Path(path).exists()
    assert Path(path).suffix.lower() == ".pdf"

