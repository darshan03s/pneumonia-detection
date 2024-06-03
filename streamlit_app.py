import os, random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from download_model import download_model

download_message = download_model()

model_path = 'model/pneu-model.keras'

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
    st.title("Binary Classification of X-Ray Images using Inception V3")
    st.write("## Pneumonia or Normal")
    
    st.write(f"{download_message}")
    loaded_model = load_cached_model()

    st.write(loaded_model)
    st.write(f"Model Loaded: {os.path.basename(model_path)}")
    st.write(f'Model Size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB')

    st.title("Image Input")

    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

    demo_images = os.listdir('demo_images/')
    demo_images_paths = [os.path.join('demo_images/', fname) for fname in demo_images]
    random.shuffle(demo_images_paths)
    options = demo_images_paths
    selected_option = st.selectbox('Choose a demo image:', options, key="demo_image_selectbox", index=None)

    if selected_option:
        st.write(f'You selected: {selected_option}')
        image_file = selected_option
    
    if image_file is not None:
        if isinstance(image_file, str): 
            image_name = os.path.basename(image_file)
        else:
            image_name = image_file.name 
            
        predict_button = st.button("Predict", use_container_width=True)
        if predict_button:
            image = Image.open(image_file).convert('RGB')  
            st.image(image, caption=image_name, use_column_width=True)
            
            predicted_class, prediction = predict(image, loaded_model)

            if predicted_class == 1:
                result_str = "The X-Ray Image is of Pneumonia"
            else:
                result_str = "The X-Ray Image is Normal"

            st.write(f'## {result_str}')
            st.write(f'## Prediction: {prediction}')

if __name__ == "__main__":
    main()
