import streamlit as st
from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json
from PIL import Image


# Load the model architecture from JSON
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Create an empty model
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights("model_weights.weights.h5")

# Define a function to make predictions
def predict(image_file):
    test_image = Image.open(image_file)
    test_image = test_image.resize((64, 64))
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loaded_model.predict(test_image)
    return result

# Set up the Streamlit app
st.title("Cat :cat: or Dog :dog: Classifier")

# Allow users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", 'webp'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    
    # Make predictions when the user clicks the 'Predict' button


if st.button('Predict'):
    #loading_image = Image.open('loading-loading-forever.gif')
    #st.image(loading_image, caption='Processing...', use_column_width=False)
    result = predict(uploaded_file)
    if result[0][0] == 1:
        st.write('This is a Dog :dog:')
    else:
        st.write('This is Cat :cat:')

