import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import cv2

# Load model
model = tf.keras.models.load_model("digit_classifier_v2.h5")

# App window
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Digit Recognizer")

        self.canvas_width = 280
        self.canvas_height = 280

        # Set up canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white",
                                highlightthickness=2, highlightbackground="black", cursor="cross")
        self.canvas.pack(pady=10)
        print("Canvas loaded successfully")

        # Create PIL image for drawing
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse drawing to canvas
        self.canvas.bind("<B1-Motion>", self.draw_lines)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack()

        self.predict_button = tk.Button(btn_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(btn_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Label to show result
        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 18))
        self.result_label.pack(pady=10)

    def draw_lines(self, event):
        x, y = event.x, event.y
        r = 6  # Brush radius
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=255)
        self.result_label.config(text="")

    def predict_digit(self):
        # Resize, invert, normalize
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img).astype("float32") / 255.0

        # Apply slight blur for smoother predictions
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Prepare for model input
        img = img.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        self.result_label.config(text=f"Predicted: {predicted_class} ({confidence:.2f}%)")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
