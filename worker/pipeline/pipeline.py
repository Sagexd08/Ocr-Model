from .phases import preprocessing, ocr, postprocessing

def run_pipeline(data):
    data = preprocessing.preprocess(data)
    data = ocr.ocr(data)
    data = postprocessing.postprocess(data)
    return data
