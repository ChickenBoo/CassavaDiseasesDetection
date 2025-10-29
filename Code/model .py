import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Resizing, Dropout

# Define the model architecture
def create_model():
    inputs = Input(shape=(244, 244, 3))

    # Add a resizing layer to match the input shape expected by the pretrained model
    x = Resizing(224, 224)(inputs)

    # Load your pretrained model (e.g., EfficientNetB0) without the top classification layer
    pretrained_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Connect the resized input to the pretrained model
    x = pretrained_model(x)
    x = GlobalAveragePooling2D()(x)  # Pooling to reduce dimensions

    # Add your custom layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.45)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.45)(x)
    outputs = Dense(10, activation='softmax')(x)  # Adjust to 10 classes

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create the model and load the weights
model = create_model()
model.load_weights('/home/pi/project/env/GUI/cassava_classification_model_checkpoint.weights.h5')

def predict(image):
    return model.predict(image)

def decode_predictions(predictions):
    # Define the class names and corresponding disease management suggestions
    class_names = [
        "Cassava Anthracnose Disease",
        "Cassava Bacterial Blight",
        "Cassava Brown Streak Disease (CBSD)",
        "Cassava Green Mite",
        "Cassava Root Rot Disease",
        "Mosaic Disease",
        "Normal Leaves",
        "Normal Stem",
        "Normal Tuber",
        "Witches' Broom"
    ]

    suggestions = [
        "Apply fungicides, use disease-free seeds, and control insect vectors",
        "Use disease-free cuttings, remove infected plants, rotate crops, and apply copper-based bactericides",
        "Use resistant varieties, control whitefly vectors, and remove infected plants",
        "Use acaricides, introduce natural predators, and plant resistant varieties",
        "Use fungicides, ensure good drainage, practice crop rotation, and plant resistant varieties",
        "Plant resistant varieties, control whitefly vectors, and remove infected plants",
        "Leaves are Normal",
        "Stem is Normal",
        "Tuber is Normal",
        "Remove infected plants, control insect vectors, and use resistant varieties"
    ]

    # Get the confidence level and the index of the highest probability
    confidence_level = np.max(predictions)
    class_idx = np.argmax(predictions)

    if confidence_level < 0.5:
        # If confidence is below 0.5, classify as "Not a cassava plant"
        disease_name = "Not a cassava plant"
        suggestion = "Not a cassava plant"
    else:
        # Get the corresponding disease name and suggestion
        disease_name = class_names[class_idx]
        suggestion = suggestions[class_idx]

    # Return the result as a formatted string
    result = f"{disease_name}: {suggestion} (Confidence: {confidence_level:.2f})"
    return result
