import numpy as np

from PIL import Image

from tensorflow.keras.applications import (
    MobileNetV2
)

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)


def load_classification_model():

    model = MobileNetV2(
        weights="imagenet"
    )

    return model


def preprocess_image(
    image_path
):

    img = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    img = img.resize(
        (224, 224)
    )

    img_array = np.array(
        img
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    return img_array


def classify_image(
    image_path
):

    model = load_classification_model()

    processed_image = preprocess_image(
        image_path
    )

    predictions = model.predict(
        processed_image
    )

    decoded_predictions = decode_predictions(
        predictions,
        top=5
    )[0]

    results = []

    for prediction in decoded_predictions:

        results.append({

            "Class":
                prediction[1],

            "Confidence":
                float(
                    prediction[2]
                )
        })

    return results
