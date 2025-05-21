import cv2
import mediapipe as mp
import os
import csv

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Define dataset path
DATASET_PATH = "dataset"  # Change this to your dataset path
OUTPUT_CSV = "hand_landmarks2.0.csv"

# Prepare CSV file
with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ["class_label"] + [f"L_{i}_{axis}" for i in range(21) for axis in ["x", "y"]]
    writer.writerow(header)

    # Loop through each class folder
    for class_name in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Process each image in the class folder
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get wrist coordinates
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wrist_x, wrist_y = wrist.x, wrist.y
                    
                    # Extract landmarks relative to wrist
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append(landmark.x - wrist_x)
                        landmarks.append(landmark.y - wrist_y)
                    
                    # Write data to CSV
                    writer.writerow([class_name] + landmarks)

hands.close()
print(f"Landmark extraction completed. Data saved to {OUTPUT_CSV}")
