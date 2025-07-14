from PIL import Image
import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tools.image_editor import ImageEditor

import yaml

with open("config/base_config.yaml") as f:
    test_config = yaml.safe_load(f)

with open("config/segmentation_model/base_segmentation_model.yaml") as f:
    segmentation_config = yaml.safe_load(f)

with open("config/inpainting_model/migan.yaml") as f:
    inpainting_config = yaml.safe_load(f)

print("Test configuration loaded:")
print(test_config)

@pytest.fixture
def test_image():
    # Create a test image with a mock object
    img_path = "D:/AI_Learning/AI_Agents/images/Test_3.jpg"
    # Check if the file exists
    if os.path.exists(img_path):
        print(f"Test image already exists at {img_path}. Using existing file.")


    image = Image.new('RGB', (800, 600), color='red')
    
    # Add a simple "object" (green rectangle)
    pixels = image.load()
    for i in range(100, 200):
        for j in range(100, 200):
            pixels[i, j] = (0, 255, 0)
    
    # image.save(img_path)
    return str(img_path)

def test_mixed_operations(test_image, capsys):
    # Initialize editor with test config
    editor = ImageEditor(test_config, segmentation_config, inpainting_config)
    
    # Process request with explicit verbose instruction
    output_path = "D:/AI_Learning/AI_Agents/images/output.jpg"
    result = editor.edit_image(
        test_image, 
        "Greyscale then detect edges by canny method with low_threshold=0.2, high_threshold=0.8, "
        "resize to 2000*2000, and show detailed processing steps"
    )
    
    # Save result and capture logs
    result.save(output_path)
    
    # Get captured output
    captured = capsys.readouterr()
    print("\n=== CAPTURED LOGS ===")
    print(captured.out)
    
    # Print paths for manual verification
    print(f"\nInput image: {test_image}")
    print(f"Output image: {output_path}")