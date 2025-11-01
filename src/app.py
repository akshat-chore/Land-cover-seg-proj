import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from patchify import patchify, unpatchify
from utils.constants import Constants
from utils.plot import visualize
import math
import os

# ==================== CONFIG ====================
MODEL_DIR = "D:\\land-cover-seg\\Land-Cover-Semantic-Segmentation-PyTorch\\models"  # Absolute path to the models directory
MODEL_NAME = "trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth"
model_path = os.path.join(MODEL_DIR, MODEL_NAME)

ENCODER = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"
TEST_CLASSES = ['background', 'building', 'water']
PATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="Land Cover Segmentation", layout="wide")
st.title("🌍 Land Cover Semantic Segmentation Dashboard")
st.markdown(
    "Upload a satellite image to run segmentation using the pretrained **U-Net (EfficientNet-B0)** model."
)

# Sidebar info
with st.sidebar:
    st.header("⚙️ Model Configuration")
    st.write(f"**Model:** {MODEL_NAME}")
    st.write(f"**Encoder:** {ENCODER}")
    st.write(f"**Device:** {DEVICE.upper()}")
    st.write(f"**Classes:** {', '.join(TEST_CLASSES)}")

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    st.info("Loading model into memory...")
    import segmentation_models_pytorch as smp

    # Allow Unet deserialization
    torch.serialization.add_safe_globals([smp.Unet])

    # Load full model safely
    model = torch.load(model_path, map_location=torch.device(DEVICE), weights_only=False)
    model.eval()

    if DEVICE == "cuda":
        model = model.half()  # FP16 for faster GPU inference

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    return model, preprocessing_fn

model, preprocessing_fn = load_model()
st.success("✅ Model loaded successfully.")

# ==================== IMAGE UPLOAD ====================
uploaded_file = st.file_uploader("📤 Upload an image (.tif, .jpg, .png)", type=["tif", "jpg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Run Segmentation"):
        st.info("Processing image...")
        progress = st.progress(0)

        # Pad image to match patch size
        pad_h = (math.ceil(image.shape[0] / PATCH_SIZE) * PATCH_SIZE) - image.shape[0]
        pad_w = (math.ceil(image.shape[1] / PATCH_SIZE) * PATCH_SIZE) - image.shape[1]
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        # Patchify image
        patches = patchify(image_padded, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE // 2)[:, :, 0, :, :, :]
        mask_patches = np.empty(patches.shape[:-1], dtype=patches.dtype)

        # Inference loop
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                img_patch = preprocessing_fn(patches[i, j])
                img_patch = img_patch.transpose(2, 0, 1).astype("float16" if DEVICE=="cuda" else "float32")
                x_tensor = torch.from_numpy(img_patch).to(DEVICE).unsqueeze(0)
                if DEVICE == "cuda":
                    x_tensor = x_tensor.half()
                with torch.no_grad():
                    pred_mask = model.predict(x_tensor)
                pred_mask = pred_mask.squeeze().cpu().numpy().round()
                pred_mask = pred_mask.transpose(1, 2, 0)
                pred_mask = pred_mask.argmax(2)
                mask_patches[i, j, :, :] = pred_mask
            progress.progress(int((i + 1) / patches.shape[0] * 100))

        # Unpatchify and crop to original size
        pred_mask = unpatchify(mask_patches, image_padded.shape[:-1])
        pred_mask = pred_mask[:image.shape[0], :image.shape[1]]

        # Display classes found
        classes_found = [Constants.CLASSES.value[c] for c in np.unique(pred_mask)]
        st.success(f"✅ Segmentation complete. Classes detected: {', '.join(classes_found)}")

        # Visualization
        plot_fig = visualize(image=image, predicted_mask=pred_mask)
        st.subheader("🖼️ Segmentation Results")
        st.pyplot(plot_fig)

        # Download mask
        st.download_button(
            label="💾 Download Predicted Mask",
            data=cv2.imencode(".png", pred_mask.astype(np.uint8))[1].tobytes(),
            file_name="predicted_mask.png",
            mime="image/png"
        )

else:
    st.info("👆 Upload a satellite image to begin segmentation.")
