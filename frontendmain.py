import streamlit as st
import pandas as pd
import numpy as np

import requests
import streamlit as st
from PIL import Image

STYLES = {
    "candy": "candy",
    "composition 6": "composition_vii",
    "feathers": "feathers",
    "la_muse": "la_muse",
    "mosaic": "mosaic",
    "starry night": "starry_night",
    "the scream": "the_scream",
    "the wave": "the_wave",
    "udnie": "udnie",
}

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Garbage Classifier App")

# displays a file uploader widget
image = st.file_uploader("Choose an image")

# displays a button
if st.button("Tell me what it is!"):
    if image is not None:
        files = {"file": image.getvalue()}
        # print(image.getvalue())
        res = requests.post(f"http://0.0.0.0:8000/predict", files=files)
        res = res.json()
        print(res)
        # image = Image.open(img_path.get("name"))
        # st.image(image, width=500)
