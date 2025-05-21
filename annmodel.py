import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
csv_file = 'hand_landmarks_augmented3.0.csv'
df = pd.read_csv(csv_file)

# Separate features (landmarks) and labels (gesture classes)
X = df.iloc[:, 1:].values  # All columns except the first (gesture labels)
y = df.iloc[:, 0].values  # First column (gesture labels)

# Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to numbers

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)

# Normalize the feature values
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Define the ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),  # 42 features (21 landmarks × 2)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')  # Output layer (number of gestures)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save('hand_gesture_ann2.0.h5')
print("Model saved as 'hand_gesture_ann2.0.h5'")

# Save the label encoder for later use
import joblib
joblib.dump(label_encoder, 'gesture_label_encoder2.0.pkl')
print("Label encoder saved as 'gesture_label_encoder2.0.pkl'")
