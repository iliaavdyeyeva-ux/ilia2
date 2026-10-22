mport os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# =========================
# CONFIG
# =========================

DATASET_PATH = "dataset/"
MODEL_PATH = "image_classifier.h5"

IMAGE_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.50
TOP_PREDICTIONS = 3


# =========================
# GET CLASS NAMES
# =========================

class_names = sorted([
    name for name in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, name))
])

num_classes = len(class_names)

if num_classes < 2:
    raise ValueError(
        f"Dataset must contain at least 2 class folders. "
        f"Found: {class_names}"
    )

print(f"Classes found: {class_names}")


# =========================
# LOAD MODEL
# =========================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find model: {MODEL_PATH}"
    )

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print(f"Input shape: {model.input_shape}")


# =========================
# PREDICT IMAGE
# =========================

def predict_image(image_path):

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    # -------------------------
    # Load image
    # -------------------------

    try:
        img = keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE,
            color_mode="rgb"
        )

    except Exception as e:
        print(f"❌ Could not load image: {e}")
        return

    # -------------------------
    # Convert image to array
    # -------------------------

    img_array = keras.utils.img_to_array(img)

    # Normalize pixel values
    img_array = img_array.astype("float32") / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------
    # Prediction
    # -------------------------

    prediction = model.predict(
        img_array,
        verbose=0
    )

    # =========================
    # BINARY CLASSIFICATION
    # =========================

    if num_classes == 2:

        # Handle models with sigmoid output
        if prediction.shape[-1] == 1:

            probability = float(prediction[0][0])

            if probability >= 0.5:
                predicted_index = 1
                confidence = probability
            else:
                predicted_index = 0
                confidence = 1 - probability

        # Handle models with 2 softmax outputs
        else:

            probabilities = prediction[0]

            predicted_index = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[predicted_index]
            )

        predicted_class = class_names[predicted_index]

        print("\n========================")
        print("PREDICTION")
        print("========================")
        print(f"Class: {predicted_class}")
        print(f"Confidence: {confidence * 100:.2f}%")

        if confidence < CONFIDENCE_THRESHOLD:
            print("⚠️ Low confidence prediction")

    # =========================
    # MULTI-CLASS CLASSIFICATION
    # =========================

    else:

        probabilities = prediction[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])

        print("\n========================")
        print("PREDICTION")
        print("========================")
        print(f"Class: {predicted_class}")
        print(f"Confidence: {confidence * 100:.2f}%")

        if confidence < CONFIDENCE_THRESHOLD:
            print("⚠️ Low confidence prediction")

        # -------------------------
        # Top predictions
        # -------------------------

        top_indices = np.argsort(
            probabilities
        )[::-1][:TOP_PREDICTIONS]

        print("\nTop predictions:")

        for index in top_indices:
            print(
                f"  {class_names[index]}: "
                f"{probabilities[index] * 100:.2f}%"
            )


# =========================
# RUN
# =========================

predict_image("images.jpeg")
