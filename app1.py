import os
import cv2
import sys
import time
import numpy as np
import streamlit as st
from PIL import Image

# ------------------------------
# Dummy Detection Functions
# ------------------------------
# (Replace these with your actual YOLO detection and Tesseract OCR implementations)
class BBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class Detection:
    def __init__(self, bbox):
        self.bbox = bbox

def prepare_img(image, img_height):
    """Resize the image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    scale = img_height / h
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, img_height))
    return resized

def yolo_detect(image):
    """
    Dummy YOLO detection function.
    Returns a list of bounding boxes in the form (x1, y1, x2, y2, label).
    For demonstration, it returns four boxes covering four regions.
    """
    h, w = image.shape[:2]
    boxes = [
        (int(0.1 * w), int(0.1 * h), int(0.4 * w), int(0.2 * h), "Test Name"),
        (int(0.5 * w), int(0.1 * h), int(0.7 * w), int(0.2 * h), "Value"),
        (int(0.1 * w), int(0.3 * h), int(0.4 * w), int(0.4 * h), "Unit"),
        (int(0.5 * w), int(0.3 * h), int(0.7 * w), int(0.4 * h), "Ref Value")
    ]
    return boxes

def run_tesseract_ocr(crop, label):
    """
    Dummy Tesseract OCR function.
    Returns a fake string based on the label.
    """
    if label == "Test Name":
        return "TOTAL TRIIODOTHYRONINE (T3)"
    elif label == "Value":
        return "79"
    elif label == "Unit":
        return "ng/dl"
    elif label == "Ref Value":
        return "60-200"
    return "N/A"

# ------------------------------
# Streamlit App Setup
# ------------------------------
st.set_page_config(page_title="YOLO + Tesseract OCR Demo", layout="wide")

# Display banner image (img2.jpg) resized to a short heading size (e.g., 400px width)
banner_path = "C:/data science material/Project_10/yolov3/image/img2.jpg"
if os.path.exists(banner_path):
    banner_img = Image.open(banner_path)
    banner_img = banner_img.resize((400, int(banner_img.height * 400 / banner_img.width)))
    st.image(banner_img)
else:
    st.title("YOLO + Tesseract OCR Demo")

st.markdown("""
<div style='text-align:center;'>
    <h3>YOLO + Tesseract Workflow</h3>
    <p>by Minal Devikar</p>
    <p>Upload an image or use the default image to run detection and OCR.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Image Input Section
# ------------------------------
uploaded_file = st.file_uploader("Upload a lab report image", type=["jpg", "jpeg", "png", "bmp"])
default_image_path = "C:/data science material/Project_10/yolov3/image/img1.jpg"

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(uploaded_file, caption="Uploaded Image")
else:
    if os.path.exists(default_image_path):
        image_cv = cv2.imread(default_image_path)
        st.image(Image.open(default_image_path), caption="Default Image")
    else:
        st.error("No image uploaded and default image not found.")
        st.stop()

# ------------------------------
# Run Detection and OCR Section
# ------------------------------
if st.button("Run YOLO + Tesseract OCR"):
    st.write("**Running YOLO detection...**")
    boxes = yolo_detect(image_cv)

    st.write("**Running Tesseract OCR on each bounding box...**")
    results = []
    for (x1, y1, x2, y2, label) in boxes:
        crop = image_cv[y1:y2, x1:x2]
        text = run_tesseract_ocr(crop, label)
        results.append((label, text))

    # Draw bounding boxes on a copy of the image
    display_image = image_cv.copy()
    for (x1, y1, x2, y2, label) in boxes:
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    st.image(display_image_rgb, caption="Detection Result")

    st.write("**Extracted Text**")
    table_data = []
    for (label, text) in results:
        table_data.append([label, text])
    st.table(table_data)

    st.success("Done! Replace dummy logic with your actual implementations.")
