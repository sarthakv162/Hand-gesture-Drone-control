import pandas as pd
import numpy as np

# Load CSV file
INPUT_CSV = "hand_landmarks2.0.csv"
OUTPUT_CSV = "hand_landmarks_augmented3.0.csv"
df = pd.read_csv(INPUT_CSV)

# Extract class labels and landmark coordinates
class_labels = df.iloc[:, 0]
landmarks = df.iloc[:, 1:].values  # Extract only landmark coordinates

# Augmentation Functions
def scale(landmarks, factor):
    """Scales landmarks by a given factor."""
    return landmarks * factor

def rotate(landmarks, angle):
    """Rotates landmarks around the wrist (first landmark)."""
    angle = np.radians(angle)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    landmarks = landmarks.reshape(-1, 2)
    
    wrist = landmarks[0]  # Wrist as center
    rotated = np.dot(landmarks - wrist, np.array([[cos_a, -sin_a], [sin_a, cos_a]])) + wrist
    return rotated.flatten()

def translate(landmarks, shift_x, shift_y):
    """Translates landmarks by a small shift."""
    landmarks = landmarks.reshape(-1, 2)
    landmarks[:, 0] += shift_x  # Shift x
    landmarks[:, 1] += shift_y  # Shift y
    return landmarks.flatten()

def add_noise(landmarks, noise_level):
    """Adds small random noise to landmarks."""
    return landmarks + np.random.normal(0, noise_level, landmarks.shape)

def mirror_flip(landmarks):
    """Flips the hand horizontally by negating x-coordinates."""
    landmarks = landmarks.reshape(-1, 2)
    landmarks[:, 0] = -landmarks[:, 0]  # Flip x-axis
    return landmarks.flatten()

# Apply multiple augmentations
augmented_data = []
for i in range(len(landmarks)):
    original = landmarks[i]
    
    # Multiple scaling factors
    scaled_1 = scale(original, 1.1)
    scaled_2 = scale(original, 0.9)
    
    # Multiple rotations
    rotated_1 = rotate(original, 10)
    rotated_2 = rotate(original, -10)
    rotated_3 = rotate(original, 15)
    rotated_4 = rotate(original, -15)
    
    # Multiple translations
    translated_1 = translate(original, 0.02, 0.02)
    translated_2 = translate(original, -0.02, -0.02)
    
    # Multiple noise levels
    noisy_1 = add_noise(original, 0.01)
    noisy_2 = add_noise(original, 0.02)
    
    # Horizontal Flip
    flipped = mirror_flip(original)

    augmented_data.extend([
        [class_labels[i]] + list(original),
        [class_labels[i]] + list(scaled_1),
        [class_labels[i]] + list(scaled_2),
        [class_labels[i]] + list(rotated_1),
        [class_labels[i]] + list(rotated_2),
        [class_labels[i]] + list(rotated_3),
        [class_labels[i]] + list(rotated_4),
        [class_labels[i]] + list(translated_1),
        [class_labels[i]] + list(translated_2),
        [class_labels[i]] + list(noisy_1),
        [class_labels[i]] + list(noisy_2),
        [class_labels[i]] + list(flipped)
    ])

# Save augmented data
augmented_df = pd.DataFrame(augmented_data, columns=df.columns)
augmented_df.to_csv(OUTPUT_CSV, index=False)

print(f"Augmented dataset saved to {OUTPUT_CSV}")
