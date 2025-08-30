from PIL import Image
from worker.pipeline.processors.exporter import Exporter, ExportFormat
from worker.types import Document, Page, Token, Bbox, Region


def test_smoke_annotation(tmp_path):
    img = Image.new("RGB", (120, 60), color=(255, 255, 255))
    page = Page(
        page_num=1,
        width=120,
        height=60,
        image=img,
        tokens=[Token(text="hi", bbox=Bbox(x1=10,y1=10,x2=30,y2=30), confidence=0.6)],
        regions=[Region(type="paragraph", bbox=Bbox(x1=5,y1=5,x2=100,y2=50), confidence=0.8)],
    )
    doc = Document(pages=[page])
    exporter = Exporter({"output_dir": str(tmp_path), "formats": {"pdf": {"format_type": "pdf"}}})
    path = exporter._export_pdf(doc, "smoke", ExportFormat(format_type="pdf"))
    assert path

