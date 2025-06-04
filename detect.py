import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("C:\Traffic_sign_detection\model_traffic_data1.h5")

# Class labels
classes = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)", 3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)", 6: "End of Speed limit (80km/h)", 7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)", 9: "No passing", 10: "No passing veh over 3.5 tons", 11: "Right-of-way at intersection",
    12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles", 16: "Veh > 3.5 tons prohibited", 17: "No entry",
    18: "General caution", 19: "Dangerous curve left", 20: "Dangerous curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on the right", 25: "Road work", 26: "Traffic signals",
    27: "Pedestrians", 28: "Children crossing", 29: "Bicycle crossing", 30: "Beware of ice/snow",
    31: "Wild animals crossing", 32: "End speed + passing limits", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left", 38: "Keep right", 39: "Keep Left",
    40: "Roundabout mandatory", 41: "End of no passing", 42: "End of no passing veh > 3.5 tons"
}

# Input shape expected by the model
target_size = (33, 33)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and convert to grayscale
    resized = cv2.resize(frame, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Reshape to model input
    input_img = np.expand_dims(gray, axis=-1)        # (33, 33, 1)
    input_img = np.expand_dims(input_img, axis=0)    # (1, 33, 33, 1)
    input_img = input_img / 255.0                    # Normalize

    # Prediction
    predictions = model(input_img,training=False)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    label = classes.get(class_index, "Unknown")

    # Display prediction
    display_text = f"{label} ({confidence * 100:.2f}%)"
    cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    cv2.imshow("Traffic Sign Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
