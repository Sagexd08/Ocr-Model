from ..types import Document, Page, Bbox, Token
import pdfplumber
from PIL import Image
import io
import docx
from skimage import io as ski_io
from skimage.transform import rotate
from skimage.color import rgb2gray
from skimage.filters import threshold_local
import numpy as np

def preprocess(document: Document, file_data) -> Document:
    if document.mime_type == "application/pdf":
        return preprocess_pdf(document, file_data)
    elif document.mime_type.startswith("image/"):
        return preprocess_image(document, file_data)
    elif document.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return preprocess_docx(document, file_data)
    else:
        return document

def preprocess_pdf(document: Document, file_data) -> Document:
    with pdfplumber.open(io.BytesIO(file_data)) as pdf:
        for i, pdf_page in enumerate(pdf.pages):
            page = Page(
                page_number=i + 1,
                tokens=[],
                image=pdf_page.to_image().original
            )
            for obj in pdf_page.extract_words():
                token = Token(
                    text=obj["text"],
                    bbox=Bbox(
                        x1=obj["x0"],
                        y1=obj["top"],
                        x2=obj["x1"],
                        y2=obj["bottom"],
                    ),
                    confidence=1.0
                )
                page.tokens.append(token)
            document.pages.append(page)
    return document

def preprocess_image(document: Document, file_data) -> Document:
    image = ski_io.imread(io.BytesIO(file_data), as_gray=True)
    
    
    angle = 90 - np.rad2deg(np.arctan2(*np.hsplit(np.int32(np.sum(np.where(image > 0.5, 1, 0), axis=0)), 2)))
    rotated = rotate(image, angle[0], resize=True)

    
    local_thresh = threshold_local(rotated, 15, offset=10)
    binary_local = rotated > local_thresh

    page = Page(
        page_number=1,
        tokens=[],
        image=Image.fromarray((binary_local * 255).astype(np.uint8))
    )
    document.pages.append(page)
    return document

def preprocess_docx(document: Document, file_data) -> Document:
    doc = docx.Document(io.BytesIO(file_data))
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    
    
    page = Page(
        page_number=1,
        tokens=[
            Token(
                text='\n'.join(full_text),
                bbox=Bbox(x1=0, y1=0, x2=1, y2=1),
                confidence=1.0
            )
        ],
    image=None
    )
    document.pages.append(page)
    return document