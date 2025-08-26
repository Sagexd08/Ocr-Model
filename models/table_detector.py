from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor
import torch
from PIL import Image

class TableDetector:
    def __init__(self, model_name='microsoft/table-transformer-detection'):
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(model_name)

    def detect(self, page_image):
        if isinstance(page_image, str):
            page_image = Image.open(page_image).convert("RGB")
        elif not isinstance(page_image, Image.Image):
            page_image = Image.fromarray(page_image).convert("RGB")

        encoding = self.feature_extractor(page_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**encoding)

        width, height = page_image.size
        results = self.feature_extractor.post_process_object_detection(outputs, threshold=0.9, target_sizes=[(height, width)])[0]

        return results['boxes'].tolist()
