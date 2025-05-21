# Hand Gesture Recognition Using ANN and MediaPipe

This project implements a hand gesture recognition system using an Artificial Neural Network (ANN) trained on MediaPipe hand landmark data. It supports real-time gesture prediction via webcam and includes tools for dataset creation and augmentation.

# Project Structure


assignlandmarks.py         # Extracts landmarks from images using MediaPipe
dataaumentate.py           # Performs data augmentation (scaling, rotation, flip, etc.)
annmodel.py                # Trains ANN model on the landmark data
realtimerun2.0.py          # Real-time hand gesture recognition via webcam
dataset/                   # Folder containing gesture class images
hand_landmarks2.0.csv      # CSV file with landmark coordinates
hand_landmarks_augmented3.0.csv # Augmented data CSV
hand_gesture_ann2.0.h5     # Trained ANN model (generated after training)
gesture_label_encoder2.0.pkl # Saved label encoder for class names


# How It Works

1. `assignlandmarks.py`: Extracts 21 hand landmark coordinates from labeled images and saves them relative to the wrist.
2. `dataaumentate.py`: Applies various augmentations (scaling, rotation, translation, flipping, noise) to the dataset to improve model generalization.
3. `annmodel.py`: Loads the augmented CSV, trains a feedforward ANN with TensorFlow, and saves the model and label encoder.
4. `realtimerun2.0.py`: Loads the trained model and predicts hand gestures in real time using webcam input.

#Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- TensorFlow
- NumPy
- Pandas
- scikit-learn
- joblib

Install dependencies with:

bash
pip install opencv-python mediapipe tensorflow pandas numpy scikit-learn joblib


# Usage

Step 1: Extract Hand Landmarks
Place images into `dataset/<class_name>/` folders, then run:
```bash
python assignlandmarks.py
```

Step 2: Augment the Dataset
```bash
python dataaumentate.py
```

Step 3: Train the ANN Model
```bash
python annmodel.py
```

Step 4: Run Real-Time Gesture Recognition
```bash
python realtimerun2.0.py
```

Press `q` to quit the webcam window.

#Sample Gestures

This model supports multiple gesture classes as defined by the folder names inside the dataset. You can add more gestures by including them in the dataset and retraining the model.

#Author

Sarthak Verma – Passionate about Machine Learning and Real-Time Computer Vision
