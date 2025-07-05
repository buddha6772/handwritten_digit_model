import tensorflow as tf
from tensorflow.keras.datasets import mnist

model = tf.keras.models.load_model("digit_classifier_v2.h5")
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
