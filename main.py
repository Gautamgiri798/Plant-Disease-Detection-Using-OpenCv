import cv2
import tensorflow as tf
import numpy as np
from collections import deque

# -----------------------------
# Load your trained model
# -----------------------------
model = tf.keras.models.load_model("trained_model.keras")

# Class labels
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# -----------------------------
# Preprocess frame
# -----------------------------
def preprocess_frame(frame):
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Webcam capture
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Try changing index in cv2.VideoCapture().")
    exit()

print("Starting real-time plant disease detection... Press 'q' to quit.")

# Use a deque to stabilize predictions over the last N frames
history = deque(maxlen=5)  # You can increase maxlen for smoother results

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)  # Mirror view

    # Preprocess and predict
    img = preprocess_frame(frame)
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction)
    history.append(predicted_class)

    # Stabilize prediction by taking the most common prediction in the history
    label_index = max(set(history), key=history.count)
    label = class_names[label_index]

    # Draw label on a fresh copy of the frame
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Prediction: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Plant Disease Detection - Press 'q' to Quit", display_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
