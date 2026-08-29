import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)

from tensorflow.keras.preprocessing import image


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="AI Image Classification",
    page_icon="👁️",
    layout="wide"
)


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():

    model = MobileNetV2(
        weights="imagenet"
    )

    return model


model = load_model()


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("👁️ AI Image Classification System")

st.markdown(
    "Upload an image and use Deep Learning and "
    "Computer Vision to identify objects."
)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header("📋 How It Works")

st.sidebar.info(
    """
    1. Upload an image.

    2. The image is processed.

    3. MobileNetV2 analyzes the image.

    4. The system displays the top predictions.
    """
)


# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------

st.subheader("📤 Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# -------------------------------------------------
# IMAGE CLASSIFICATION
# -------------------------------------------------

if uploaded_file is not None:

    image_file = Image.open(
        uploaded_file
    )

    st.subheader(
        "🖼️ Uploaded Image"
    )

    st.image(
        image_file,
        use_container_width=True
    )


    # ---------------------------------------------
    # PREPROCESS IMAGE
    # ---------------------------------------------

    img = image_file.resize(
        (224, 224)
    )

    img_array = np.array(
        img
    )

    # Convert grayscale images to RGB

    if len(img_array.shape) == 2:

        img_array = np.stack(
            (
                img_array,
                img_array,
                img_array
            ),
            axis=-1
        )


    # Remove alpha channel if present

    if img_array.shape[-1] == 4:

        img_array = img_array[
            :, :, :3
        ]


    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )


    # ---------------------------------------------
    # PREDICTION
    # ---------------------------------------------

    with st.spinner(
        "AI is analyzing the image..."
    ):

        predictions = model.predict(
            img_array
        )

        decoded_predictions = decode_predictions(
            predictions,
            top=5
        )[0]


    # ---------------------------------------------
    # TOP PREDICTION
    # ---------------------------------------------

    top_prediction = (
        decoded_predictions[0]
    )

    class_name = (
        top_prediction[1]
        .replace(
            "_",
            " "
        )
        .title()
    )

    confidence = (
        top_prediction[2] * 100
    )


    st.divider()

    st.subheader(
        "🤖 AI Prediction"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Object",
        class_name
    )

    col2.metric(
        "Confidence Score",
        f"{confidence:.2f}%"
    )


    # ---------------------------------------------
    # TOP 5 PREDICTIONS
    # ---------------------------------------------

    st.subheader(
        "📊 Top 5 Predictions"
    )

    prediction_data = []

    for prediction in decoded_predictions:

        label = (
            prediction[1]
            .replace(
                "_",
                " "
            )
            .title()
        )

        probability = (
            prediction[2] * 100
        )

        prediction_data.append({

            "Object": label,
            "Confidence": probability
        })


    prediction_df = pd.DataFrame(
        prediction_data
    )


    st.dataframe(
        prediction_df,
        use_container_width=True
    )


    # ---------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------

    fig = px.bar(
        prediction_df,
        x="Confidence",
        y="Object",
        orientation="h",
        title="AI Prediction Confidence Scores"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ---------------------------------------------
    # AI INSIGHT
    # ---------------------------------------------

    st.subheader(
        "💡 AI Analysis"
    )

    if confidence >= 80:

        st.success(
            f"The AI model is highly confident "
            f"that the image contains a "
            f"{class_name}."
        )

    elif confidence >= 50:

        st.info(
            f"The AI model predicts that the "
            f"image most likely contains a "
            f"{class_name}, although other "
            f"objects may also be possible."
        )

    else:

        st.warning(
            "The model has low confidence. "
            "Try uploading a clearer image."
        )


# -------------------------------------------------
# PROJECT INFORMATION
# -------------------------------------------------

else:

    st.info(
        "Upload an image to begin AI classification."
    )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "AI Image Classification System | "
    "Python • TensorFlow • MobileNetV2 • "
    "Deep Learning • Computer Vision • Streamlit"
)
