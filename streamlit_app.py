import os, random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from download_model import download_model

st.set_page_config(page_title="Pneumonia Detection", page_icon="🤕", layout="centered")

download_message = download_model()

model_path = 'model/pneu-model.keras'

demo_image_files = os.listdir('demo_images/')
# random.shuffle(demo_image_files)

@st.cache_resource
def load_cached_model():
    return load_model(model_path)

def predict(image, model):
    img = np.array(image)
    if img.shape[-1] == 4: 
        img = img[:, :, :3]  

    resize = tf.image.resize(img, (150, 150))
    resize = resize / 255.0  
    prediction = model.predict(np.expand_dims(resize, 0))
    predicted_class = np.round(prediction).astype(int)
    return predicted_class, prediction

def main():
    st.write("## Binary Classification of X-Ray Images using Inception V3 (Pneumonia or Normal)")
    
    if download_message:
        st.write(f"{download_message}")
    loaded_model = load_cached_model()

    st.write(loaded_model)
    st.write(f"Model Loaded: {os.path.basename(model_path)}({os.path.getsize(model_path) / (1024 * 1024):.2f} MB)")
    
    st.write("## Image Input")
    
    demo_image_file = None
    uploaded_image_file = None
    
    tab1, tab2 = st.tabs(["Upload Image", "Choose Demo Image"])
    
    with tab1:
        uploaded_image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif"])

        if uploaded_image_file:
            st.write("You uploaded an image:", uploaded_image_file.name)
            image_name = uploaded_image_file.name 
            upload_image = Image.open(uploaded_image_file).convert('RGB')  
            st.image(upload_image, caption=image_name)
            
            predict_button = st.button("Predict", use_container_width=True, key='upload')

            if predict_button:
                
                predicted_class, prediction = predict(upload_image, loaded_model)

                if predicted_class == 1:
                    result_str = "The X-Ray Image is of Pneumonia"
                else:
                    result_str = "The X-Ray Image is Normal"

                st.write(f'## {result_str}')
                st.write(f'## Prediction: {prediction}')
            
    with tab2:
        demo_image_file = st.selectbox("Choose a demo image", demo_image_files, index=None)

        if demo_image_file:
            st.write("You selected demo image:", demo_image_file)
                    
            demo_image_file_path = os.path.join('demo_images', demo_image_file)
            demo_image = Image.open(demo_image_file_path).convert('RGB')  
            st.image(demo_image, caption=demo_image_file)
            
            predict_button = st.button("Predict", use_container_width=True, key='demo')
            
            if predict_button:
                
                predicted_class, prediction = predict(demo_image, loaded_model)

                if predicted_class == 1:
                    result_str = "The X-Ray Image is of Pneumonia"
                else:
                    result_str = "The X-Ray Image is Normal"

                st.write(f'## {result_str}')
                st.write(f'## Prediction: {prediction}')

if __name__ == "__main__":
    main()
