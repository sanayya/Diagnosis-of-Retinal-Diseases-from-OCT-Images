import prediction
import streamlit as st
import cv2
import numpy as np

st.set_page_config(
     page_title="Diagnosis of Retinal Diseases from OCT Images by Sanayya",
     page_icon=None,
     layout="centered",
     initial_sidebar_state="auto",
     menu_items=None
 )

st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Diagnosis of Retinal Diseases from OCT Images")

# Uploading the OCT Image
input_file = st.file_uploader("Upload an OCT Image",type=["png", "jpg", "jpeg"])

# Creating a Submit button
if st.button("Submit"):

  if (input_file is not None) and input_file.name.endswith(".png") or input_file.name.endswith(".jpg") or input_file.name.endswith(".jpeg"):
      file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
      image = cv2.imdecode(file_bytes, 1)

      # Getting prediction from the saved model
      pred = prediction.OCTPrediction(image)
      
      # Displaying the final predicted dataframe
      st.image(image, caption=pred)