import streamlit as st
from keras.models import load_model
from PIL import Image


from util import classify, set_background


set_background('bg/1.png')

# set title
st.title('Cherry Leaf Disease classification')

# set header
st.header('Please upload a cherry leaf image')

# upload file 
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('model/CherryLeafMobileNet.h5')

# load class names
classes = ['Cherry Healthy', 'Cherry Powdery mildew']
# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    predicted_class, confidence = classify(image, model, classes)

    # write classification
    st.write("## {}".format(predicted_class))
    st.write("### score: {}%".format(int(confidence * 10) / 10))