import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Resizing, Dropout

# Load your h5 model
inputs = Input(shape=(244, 244, 3))

# Add a resizing layer to match the input shape expected by the pretrained model
x= Resizing(224, 224)(inputs)

# Apply augmentations (assuming augment is a previously defined function or layer)
#Load your pretrained model (e.g., EfficientNetB0) without the top classification layer
pretrained_model = tf.keras.applications.EfficientNetB0(
	input_shape=(224, 224, 3),
	include_top=False,
	weights='imagenet'
	)
    # Connect the resized and augmented input to the pretrained model
x = pretrained_model(x)
x = GlobalAveragePooling2D()(x)  # Pooling to reduce dimensions

#Add your custom layers
x = Dense(128, activation='relu')(x)
x = Dropout(0.45)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.45)(x)
outputs = Dense(10, activation='softmax')(x)  # Adjust to 10 classes

    # Create the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Create the model and load the weights
model.load_weights('cassava_classification_model_checkpoint.weights.h5')

# Print the model summary
model.summary()
