# =======================
# TFLITE INFERENCE TEST
# =======================

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load TFLite model
interpreter = tf.lite.Interpreter("recyclables_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (224, 224)

# Load image for testing
img = image.load_img("/content/test_image.jpg", target_size=IMG_SIZE)
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], x)
interpreter.invoke()

prediction = interpreter.get_tensor(output_details[0]['index'])
print("Predicted probabilities:", prediction)
