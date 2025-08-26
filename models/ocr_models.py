from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class OCRModel:
    def __init__(self, model_name='microsoft/trocr-base-handwritten'):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def run_ocr(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # For a real application, we would also need to get the bounding boxes of the words.
        # This can be done using various techniques, such as looking at the attention maps
        # of the model, or by using a separate text detection model.
        # For now, we will just return the generated text.
        return generated_text, []
