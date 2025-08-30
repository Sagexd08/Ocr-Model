from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    import layoutparser as lp  # type: ignore
except Exception:
    lp = None  # optional


class RegionClassifierLP:
    """
    Region classifier using LayoutParser / Detectron2 if available.
    Fallback: simple heuristic regions (header/body/footer) based on page geometry.

    Output format: List of {"type": str, "bbox": [x1,y1,x2,y2], "confidence": float}
    """

    def __init__(self, model_config: Optional[str] = None, label_map: Optional[Dict[int, str]] = None):
        self.label_map = label_map or {
            0: "paragraph",
            1: "title",
            2: "list",
            3: "table",
            4: "figure",
            5: "header",
            6: "footer",
        }
        self.model = None
        if lp is not None:
            try:
                # Attempt to load a default PubLayNet detector; CPU/GPU auto-handled by LP
                # Users can override with model_config if provided
                if model_config:
                    self.model = lp.Detectron2LayoutModel(model_config)
                else:
                    self.model = lp.Detectron2LayoutModel(
                        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                        label_map=self.label_map,
                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                    )
            except Exception:
                self.model = None

    def detect(self, image: Any) -> List[Dict[str, Any]]:
        # normalize to PIL
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(image).convert("RGB")

        if self.model is not None:
            try:
                layout = self.model.detect(img)
                results: List[Dict[str, Any]] = []
                for l in layout:
                    x_1, y_1, x_2, y_2 = map(float, l.block.coordinates)
                    label = getattr(l, "type", None) or self.label_map.get(getattr(l, "label", -1), "region")
                    score = float(getattr(l, "score", 0.9))
                    results.append({
                        "type": str(label),
                        "bbox": [x_1, y_1, x_2, y_2],
                        "confidence": score,
                    })
                return results
            except Exception:
                pass

        # Fallback heuristic: header (top 10%), footer (bottom 10%), body
        W, H = img.size
        h_band = int(0.1 * H)
        results = [
            {"type": "header", "bbox": [0, 0, W, h_band], "confidence": 0.5},
            {"type": "paragraph", "bbox": [0, h_band, W, H - h_band], "confidence": 0.5},
            {"type": "footer", "bbox": [0, H - h_band, W, H], "confidence": 0.5},
        ]
        return results

