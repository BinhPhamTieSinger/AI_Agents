import gdown
import torch
from pathlib import Path

# Download models from Google Drive
def download_models():
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Example Google Drive file IDs
    MODEL_URLS = {
        "migan_512_places2.pt": "https://drive.google.com/uc?id=1-pOLBMTX8LI2go8eYC_X-9OC_YDKKjoB",
        "sam_vit_h_4b8939.pth": "https://drive.google.com/uc?id=1t1U-COZnWt4iBQIw9xh7h2XH6KMZsfAh"
    }
    
    for name, url in MODEL_URLS.items():
        output = model_dir/name
        if not output.exists():
            gdown.download(url, str(output), quiet=False)

# Call this at startup
download_models()

# app.py
import streamlit as st
from PIL import Image
from src.tools.image_editor import ImageEditor
import yaml
import time
import traceback

# Load configurations
with open("config/base_config.yaml") as f:
    test_config = yaml.safe_load(f)

with open("config/detection_model/grounding_dino.yaml") as f:
    detection_config = yaml.safe_load(f)

with open("config/segmentation_model/sam.yaml") as f:
    segmentation_config = yaml.safe_load(f)

with open("config/inpainting_model/bria.yaml") as f:
    inpainting_config = yaml.safe_load(f)

# Custom CSS styling
st.set_page_config(
    page_title="AI Image Editor",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
[data-testid="stSidebar"] .sidebar-content {
    padding: 2rem;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #667eea;
    border-radius: 10px;
    padding: 20px;
}
[data-testid="stImage"] {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.progress-bar {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    height: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

def process_image(image, prompt):
    editor = ImageEditor(test_config, detection_config, segmentation_config, inpainting_config)
    result = editor.edit_image(image, prompt)
    return result

# Main app layout
st.title("🖼 AI Image Editor")
st.markdown("Transform your images with AI-powered editing")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    process_btn = st.button("✨ Process Image", use_container_width=True)
    st.markdown("---")
    st.markdown("### Example Prompts")
    st.code("""
    - Remove the car and resize to 1920x1080
    - Convert to grayscale then detect edges
    - Delete the person and add a sunset
    - Enhance colors and apply Gaussian blur
    """)

# File upload and processing
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
prompt = st.text_input("Editing instructions", placeholder="Describe what you want to do...")

if uploaded_file and process_btn and prompt:
    try:
        # Show processing status
        status = st.empty()
        progress_bar = st.progress(0)
        
        # Load image
        status.markdown("### 🔄 Loading image...")
        # original_image = Image.open(uploaded_file)
        progress_bar.progress(10)
        
        # Process image
        status.markdown("### 🧠 Processing your request...")
        with st.spinner("Applying AI magic..."):
            start_time = time.time()
            result = process_image(uploaded_file, prompt)
            progress_bar.progress(90)
            
        # Show results
        status.markdown("### ✅ Processing complete!")
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        
        # Display comparison
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(uploaded_file), caption="Original Image", use_container_width=True)
        with col2:
            st.image(result, caption="Edited Result", use_container_width=True)
            
        # Add download button
        st.download_button(
            label="📥 Download Result",
            data=result.tobytes(),
            file_name="images/edited_image.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
        
        # Show processing time
        st.success(f"Processing completed in {time.time() - start_time:.1f} seconds")

    except Exception as e:
        st.error("❌ Error processing image")
        st.error(str(e))
        st.code(traceback.format_exc())
        progress_bar.empty()

elif uploaded_file and not prompt:
    st.warning("⚠️ Please enter editing instructions")
elif uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
else:
    st.info("ℹ️ Please upload an image to get started")