import streamlit as st
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pickle
import os
import tempfile
from PIL import Image
import numpy as np

st.set_page_config(page_title="Image Captioning + Segmentation", layout="centered")
st.title("📸 Image Segmentation + Caption Generator")

# Load the caption model from .pkl
@st.cache_resource
def load_caption_model():
    with open("caption_generator.pkl", "rb") as f:
        return pickle.load(f)

generate_caption_from_objects = load_caption_model()

# Function to extract labels from VOC .xml annotation
def extract_labels(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    labels = [obj.find('name').text for obj in root.findall('object')]
    return labels

# Function to draw bounding boxes from annotation
def draw_boxes(image_path, annotation_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tree = ET.parse(annotation_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)

        # Draw bounding box and label
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return image

# Upload section
st.subheader("🖼 Upload Image and Annotation (.xml)")
uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
uploaded_xml = st.file_uploader("Upload XML Annotation", type=["xml"])

if uploaded_img and uploaded_xml:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded files to temp dir
        img_path = os.path.join(tmpdir, uploaded_img.name)
        xml_path = os.path.join(tmpdir, uploaded_xml.name)

        with open(img_path, "wb") as f:
            f.write(uploaded_img.read())

        with open(xml_path, "wb") as f:
            f.write(uploaded_xml.read())

        # Draw boxes
        try:
            boxed_img = draw_boxes(img_path, xml_path)
            labels = extract_labels(xml_path)
            caption = generate_caption_from_objects(labels)

            st.image(boxed_img, caption=caption, use_column_width=True)
            st.success(f"🧠 Caption: {caption}")
            st.write("📦 Detected Objects:", ", ".join(labels))

        except Exception as e:
            st.error(f"❌ Error processing files: {e}")
else:
    st.info("Please upload both an image and its annotation.")
