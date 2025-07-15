# app.py
import streamlit as st
from PIL import Image
from src.tools.image_editor import ImageEditor
import yaml
import time

# Load configurations
with open("config/base_config.yaml") as f:
    base_config = yaml.safe_load(f)
with open("config/segmentation_model/base_segmentation_model.yaml") as f:
    seg_config = yaml.safe_load(f)
with open("config/inpainting_model/migan.yaml") as f:
    inpaint_config = yaml.safe_load(f)

# App styling
st.set_page_config(
    page_title="AI Image Editor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# App header
st.title("✨ AI-Powered Image Editor")
st.markdown("""
<style>
[data-testid="stSidebar"][aria-expanded="true"]{
    background-image: linear-gradient(#2b5876, #4e4376);
}
</style>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    process_btn = st.button("🚀 Process Image")
    st.markdown("---")
    st.markdown("### 📝 Example Prompts")
    st.code("""
    - Remove the car and resize to 1920x1080
    - Convert to grayscale then detect edges
    - Delete the person and add a sunset
    - Enhance colors and apply Gaussian blur
    """)

# Main content
col1, col2 = st.columns([1, 2])
uploaded_file = col1.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    prompt = col2.text_input("💡 Processing Instructions", 
                           placeholder="Describe what you want to do with the image...")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", use_column_width=True)

    if process_btn and prompt:
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_container = st.container()
        log_container = st.expander("Processing Logs")
        
        try:
            with ImageEditor(base_config, seg_config, inpaint_config) as editor:
                # Initialize processing
                status_text.info("🚦 Initializing AI models...")
                progress_bar.progress(10)
                time.sleep(1)
                
                # Process image
                status_text.info("🔍 Analyzing your request...")
                progress_bar.progress(30)
                result = editor.edit_image(original_image, prompt)
                
                # Show processing steps
                status_text.info("🎨 Applying transformations...")
                progress_bar.progress(70)
                if editor.classifier.memory:
                    with result_container:
                        st.subheader("Processing Steps")
                        cols = st.columns(len(editor.classifier.memory.history))
                        for idx, step in enumerate(editor.classifier.memory.history):
                            cols[idx].image(
                                step['output'], 
                                caption=f"Step {idx+1}: {step['step']}",
                                use_column_width=True
                            )
                
                # Display final result
                status_text.success("✅ Processing complete!")
                progress_bar.progress(100)
                with col2:
                    st.image(result, caption="Final Result", use_column_width=True)
                    st.download_button(
                        label="📥 Download Result",
                        data=result.tobytes(),
                        file_name="edited_image.jpg",
                        mime="image/jpeg"
                    )
                
                # Show logs
                with log_container:
                    st.code("\n".join(editor.planner.log_buffer))

        except Exception as e:
            status_text.error(f"❌ Error: {str(e)}")
            progress_bar.progress(0)
            st.exception(e)

        finally:
            time.sleep(2)
            progress_bar.empty()
else:
    st.info("ℹ️ Please upload an image to get started")