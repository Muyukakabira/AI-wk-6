# =======================
# EDGE AI MODEL TRAINING
# =======================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset directory (arranged in subfolders per class)
DATA_DIR = "/content/dataset"
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 8

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset='validation'
)

# MobileNetV2 Transfer Learning Model
base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(train_gen.class_indices), activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save Model
model.save("/content/saved_model")

# Convert to TensorFlow Lite (Float16 Quantization)
converter = tf.lite.TFLiteConverter.from_saved_model("/content/saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
open("/content/recyclables_model.tflite", "wb").write(tflite_model)

print("TFLite model generated successfully!")
