import argparse
from pathlib import Path
from PIL import Image

from worker.pipeline.processors.region_classifier import RegionClassifierProcessor


def main():
    parser = argparse.ArgumentParser(description="Preview region-classifier overlays on an image")
    parser.add_argument("image", type=str, help="Path to an image (png/jpg)")
    parser.add_argument("--out", type=str, default=None, help="Output path for preview PNG")
    args = parser.parse_args()

    img_path = Path(args.image)
    img = Image.open(img_path).convert("RGB")

    # Minimal Document+Page structure
    from worker.types import Document, Page, Bbox, Region
    doc = Document()
    page = Page(page_num=1, image=img)
    doc.pages.append(page)

    proc = RegionClassifierProcessor()
    doc = proc.process(doc)

    regions = [
        {"type": r.type, "bbox": [r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2], "confidence": r.confidence}
        for r in doc.pages[0].regions
    ]

    from streamlit_demo.components.visualization import render_region_overlays
    out_img = render_region_overlays(img, regions, show_labels=True)
    out_path = Path(args.out) if args.out else img_path.with_name(img_path.stem + "_regions.png")
    out_img.save(out_path)
    print(f"Saved preview: {out_path}")


if __name__ == "__main__":
    main()

