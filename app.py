import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import json
import pandas as pd

# Load the previously saved model
model = load_model('/Applications/Gursimran/Projects/Image_Classification/my_model.h5')

# Load class indices from JSON file
with open('/Applications/Gursimran/Projects/Image_Classification/class_indices.json') as json_file:
    class_indices = json.load(json_file)

# Sort the classes based on their indices to get the class names list
class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

# Load calorie counts from CSV file
calorie_data = pd.read_excel('/Applications/Gursimran/Projects/Image_Classification/food_calories.xlsx')
calorie_dict = pd.Series(calorie_data['Calories'].values, index=calorie_data['Food Item']).to_dict()

# Define a function to preprocess the image
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(expanded_img_array)

# Streamlit application UI
st.title('Image Classification and Calorie Estimation Application')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    preprocessed_image = preprocess_image(uploaded_file)
    predictions = model.predict(preprocessed_image)
    class_index = np.argmax(predictions, axis=1)[0]
    class_confidence = np.max(predictions, axis=1)[0]
    predicted_class_name = class_names[class_index]

    # Display the classification result and calorie count
    st.write(f"Prediction: {predicted_class_name} with a confidence of {class_confidence:.2f}")
    if predicted_class_name in calorie_dict:
        st.write(f"Estimated Calories: {calorie_dict[predicted_class_name]}")
    else:
        st.write("Calorie count not available for this item.")
