import torch
import timm
from PIL import Image
from transformers import pipeline

class RendererClassifier:
    def __init__(self, model_name='mobilenetv3_small_100'):
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        # Get the data config for the model to resize and normalize properly
        self.data_config = timm.data.resolve_data_config({}, model=self.model)
        self.transforms = timm.data.create_transform(**self.data_config)

    def classify(self, document):
        # For now, we will just use the first page's image for classification.
        if not document.pages:
            return "unknown"

        image = document.pages[0].image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        # Preprocess the image
        input_tensor = self.transforms(image).unsqueeze(0) # create a mini-batch as expected by the model

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Get the predicted class
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        
        # For a real application, we would map the category id to a meaningful label.
        # For now, we will just return the category id as a string.
        return str(top1_catid.item())
