from PIL import Image, ImageFilter
import cv2
import numpy as np

class BasicOperations:
    def __init__(self, config=None, default_size=(800, 600), default_blur_radius=2, edge_detection_defaults=None):
        """
        Initialize basic image operations with default parameters
        
        :param default_size: Tuple (width, height) for resize operations
        :param default_blur_radius: Default Gaussian blur radius
        :param edge_detection_defaults: Dictionary of default edge detection parameters
        """
        self.default_size = default_size
        self.default_blur_radius = default_blur_radius
        self.edge_detection_defaults = edge_detection_defaults or {
            'method': 'canny',
            'low_threshold': 50,
            'high_threshold': 200
        }

    def resize_image(self, image, size=None):
        """
        Resize image to specified dimensions or default size
        
        :param image: PIL Image object
        :param size: Optional tuple (width, height)
        :return: Resized PIL Image
        """
        size = size or self.default_size
        return image.resize(size)

    def convert_to_grayscale(self, image):
        """Convert image to grayscale"""
        return image.convert("L")

    def detect_edges(self, image, method=None, low_threshold=None, high_threshold=None):
        """
        Detect edges using specified method
        
        :param image: PIL Image object
        :param method: 'canny' or 'sobel'
        :param low_threshold: Canny lower threshold
        :param high_threshold: Canny upper threshold
        :return: Edge detection result as PIL Image
        """
        # Use instance defaults if parameters not provided
        method = method or self.edge_detection_defaults['method']
        low_threshold = low_threshold or self.edge_detection_defaults['low_threshold']
        high_threshold = high_threshold or self.edge_detection_defaults['high_threshold']

        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        if method == 'canny':
            edges = cv2.Canny(img_array, low_threshold, high_threshold)
        elif method == 'sobel':
            edges = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=5)
        else:
            raise ValueError(f"Unsupported edge detection method: {method}")

        return Image.fromarray(edges)

    def blur(self, image, radius=None):
        """
        Apply Gaussian blur with specified radius
        
        :param image: PIL Image object
        :param radius: Blur radius (uses default if not specified)
        :return: Blurred PIL Image
        """
        radius = radius or self.default_blur_radius
        return image.filter(ImageFilter.GaussianBlur(radius))