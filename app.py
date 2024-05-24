import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define function to load and preprocess image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize to [0, 1] range
    return img

# Load the model
def load_model(weights_path):
    base_model = VGG19(input_shape=(128, 128, 3), include_top=False, weights=None)
    base_model.load_weights(weights_path, by_name=True)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    flat = Flatten()(x)
    class_1 = Dense(4608, activation='relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation='relu')(drop_out)
    output = Dense(2, activation='softmax')(class_2)

    model = Model(base_model.inputs, output)
    return model

# Load the pre-trained model
model_path = 'model_weights/vgg_unfrozen.h5'
vgg19_weights_path = 'C:/Users/Hp/anaconda3/envs/malaria/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = load_model(vgg19_weights_path)

# Define class names
class_names = ['Parasitized', 'Uninfected']

# Streamlit app
st.title('Malaria Detection')

st.write("Upload an image to classify as Parasitized or Uninfected.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_array = load_and_preprocess_image(uploaded_file)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    st.write(f'Prediction: {class_names[predicted_class]}')
    st.write(f'Confidence: {confidence:.2f}')

    fig, ax = plt.subplots()
    ax.barh(class_names, prediction[0], color='blue')
    ax.set_xlim(0, 1)
    st.pyplot(fig)
