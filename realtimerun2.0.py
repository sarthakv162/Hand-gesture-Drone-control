import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib

# Load trained ANN model and label encoder
model = tf.keras.models.load_model('hand_gesture_ann2.0.h5')
label_encoder = joblib.load('gesture_label_encoder2.0.pkl')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# OpenCV video capture
cap = cv2.VideoCapture(0)  # Use webcam (0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB (MediaPipe expects RGB format)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            h, w, _ = frame.shape
            wrist_x = hand_landmarks.landmark[0].x * w
            wrist_y = hand_landmarks.landmark[0].y * h
            landmarks_data = []

            for lm in hand_landmarks.landmark:
                # Get relative coordinates with respect to the wrist
                rel_x = (lm.x * w - wrist_x) / w
                rel_y = (lm.y * h - wrist_y) / h
                landmarks_data.append(rel_x)
                landmarks_data.append(rel_y)

            # Ensure landmarks_data has 42 values (21 landmarks × 2)
            if len(landmarks_data) == 42:
                # Convert to NumPy array and reshape for model input
                input_data = np.array(landmarks_data, dtype=np.float32).reshape(1, -1)

                # Predict gesture
                prediction = model.predict(input_data)
                predicted_class_index = np.argmax(prediction)
                predicted_gesture = label_encoder.inverse_transform([predicted_class_index])[0]

                # Display predicted gesture
                cv2.putText(frame, f'Gesture: {predicted_gesture}', (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show output frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
