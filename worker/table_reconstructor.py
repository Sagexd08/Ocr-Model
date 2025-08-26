from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
import numpy as np

class TableReconstructor:
    def __init__(self, table_image):
        self.table_image = table_image

    def detect_lines(self):
        grayscale_image = self.table_image.convert("L")
        image_array = np.array(grayscale_image)
        edges = canny(image_array, 2, 1, 25)
        h, theta, d = hough_line(edges)
        _, angles, dists = hough_line_peaks(h, theta, d)
        return angles, dists

    def reconstruct(self):
        angles, dists = self.detect_lines()
        return [["dummy", "table"]]