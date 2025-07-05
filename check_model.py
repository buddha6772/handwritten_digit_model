import tensorflow as tf

model = tf.keras.models.load_model("digit_classifier.h5")
model.summary()
