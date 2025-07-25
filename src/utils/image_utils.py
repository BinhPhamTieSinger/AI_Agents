from PIL import Image

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def show_image(image):
    image.show()