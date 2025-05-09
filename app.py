import streamlit as st
import numpy as np
from PIL import Image
import keras
from keras.models import load_model
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Age and Gender Predictor",
    page_icon="👤",
    layout="centered"
)

# App title and description
st.title("Age and Gender Prediction")
st.write("Upload an image or use your camera to predict age and gender.")

# Dictionary for gender mapping
gender_dict = {0: "Female", 1: "Male"}

import cv2

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def detect_and_crop_face(img: Image.Image):
#     # Convert PIL image to OpenCV format
#     img_cv = np.array(img.convert('RGB'))
#     gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
#     if len(faces) == 0:
#         return None  # No face detected
    
#     # Use the first detected face
#     (x, y, w, h) = faces[0]
#     face = img.crop((x, y, x + w, y + h))
#     return face
def detect_and_crop_face(img: Image.Image, padding: int = 50):
    # Convert PIL image to OpenCV format
    img_cv = np.array(img.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return None  # No face detected

    # Use the first detected face
    (x, y, w, h) = faces[0]
    
    # Add padding and keep within image bounds
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, img.width)
    y2 = min(y + h, img.height)
    
    # Crop the face with padding
    face = img.crop((x1, y1, x2, y2))
    return face



# Function to preprocess the image
def preprocess_image(img):
    # Convert to grayscale
    img = img.convert('L')
    # Resize to 128x128 pixels
    img = img.resize((128, 128), Image.LANCZOS)
    # Convert to numpy array
    img_array = np.array(img)
    # Normalize to 0-1
    img_array = img_array / 255.0
    # Reshape for model input
    img_array = img_array.reshape(1, 128, 128, 1)
    return img_array

# Function to make predictions
def predict_age_gender(img_array, model):
    pred = model.predict(img_array)
    pred_gender = gender_dict[int(round(pred[0][0][0]))]
    pred_age = int(pred[1][0])
    return pred_gender, pred_age

# Load the model directly from file
@st.cache_resource
def load_model_once():
    try:
        # Define custom objects if needed for your model
        custom_objects = {}
        # Specify the path to your model file
        model_path = "model.keras"
        model = load_model(model_path, custom_objects=custom_objects)
        
        # Compile the model with appropriate loss functions and metrics
        model.compile(
            loss={
                'gender_output': 'binary_crossentropy',
                'age_output': 'mae'
            },
            optimizer='adam',
            metrics={
                'gender_output': ['accuracy'],
                'age_output': ['mae']
            }
        )
        
        return model, None
    except Exception as e:
        return None, str(e)

# Load model
model, error = load_model_once()

if error:
    st.error(f"Error loading model: {error}")
    st.error("Please make sure the model.keras file is in the same directory as this app.")
else:
    st.success("Model loaded successfully!")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Camera"])

# Upload Image Tab
with tab1:
    st.write("Drop your image here or click to browse")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if model is not None:
            # Process and predict
            with st.spinner("Processing image..."):
                # Preprocess the image
                # processed_img = preprocess_image(image)
                cropped_face = detect_and_crop_face(image, padding=75)
                if cropped_face is None:
                     st.warning("No face detected in the image. Please try another image.")
                else:
                      processed_img = preprocess_image(cropped_face)
                      st.image(cropped_face, caption="Croped Image", use_container_width=True)
                    #   st.image(processed_img, caption="processed Image", use_container_width=True)
                      pred_gender, pred_age = predict_age_gender(processed_img, model)
                # Make prediction
                # pred_gender, pred_age = predict_age_gender(processed_img, model)
                
                # Display results
                      st.subheader("Prediction Results:")
                # st.write(f"**Gender:** {pred_gender}")
                      st.write(f"**Age:** {pred_age} years")
                
                # Display the processed grayscale image
                # st.image(processed_img.reshape(128, 128), caption="Processed Image (Grayscale)", use_container_width=True)

# Camera Tab
with tab2:
    st.write("Take a photo with your camera")
    
    # Use webcam
    img_file_buffer = st.camera_input("Take a picture", label_visibility="collapsed")
    
    if img_file_buffer is not None:
        # Display the image
        image = Image.open(img_file_buffer)
        
        if model is not None:
            # Process and predict
            with st.spinner("Processing image..."):
                # Preprocess the image
                # processed_img = preprocess_image(image)
                cropped_face = detect_and_crop_face(image, padding=25)
                if cropped_face is None:
                    st.warning("No face detected in the image. Please try another image.")
                else:
                    processed_img = preprocess_image(cropped_face)
                    st.image(cropped_face, caption="Croped Image", use_container_width=True)
                    pred_gender, pred_age = predict_age_gender(processed_img, model)

                
                # Make prediction
                # pred_gender, pred_age = predict_age_gender(processed_img, model)
                
                # Display results
                    st.subheader("Prediction Results:")
                # st.write(f"**Gender:** {pred_gender}")
                    st.write(f"**Age:** {pred_age} years")
                
                # Display the processed grayscale image
                # st.image(processed_img.reshape(128, 128), caption="Processed Image (Grayscale)", use_column_width=True)

# Add instructions in the sidebar
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Upload an image or use camera
    2. Wait for processing
    3. View predictions for age and gender
    """)
    
    st.header("Notes")
    st.write("""
    - Works best with clear face images
    - Images are processed in grayscale
    - Make sure lighting is adequate
    """)

# Simple footer
st.markdown("---")
st.write("Age and Gender Prediction App")