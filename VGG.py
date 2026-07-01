import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import ssl
import matplotlib.pyplot as plt


ssl._create_default_https_context = ssl._create_unverified_context


output_dir = "/Users/yuezhiluo/Downloads/processed_data"
mel_spectrograms = []
labels = []

# Load mel spectrograms and labels
for category in ["happy", "fear", "disgust", "anger", "neutral", "sad"]:
    data = np.load(f"{output_dir}/{category}.npy", allow_pickle=True)
    mel_spectrograms.append(data)
    labels.extend([category] * len(data))

# Concatenate all category arrays into a single array
mel_spectrograms = np.concatenate(mel_spectrograms, axis=0)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mel_spectrograms, categorical_labels, test_size=0.2, random_state=42)

# Reshape data for VGG input
# VGG expects 3 channels, but mel spectrograms are single channel, so we add a new axis
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
X_train = np.repeat(X_train, 3, axis=-1)  # Repeat channel to convert (128, 216, 1) -> (128, 216, 3)
X_test = np.repeat(X_test, 3, axis=-1)

# Define the VGG model using transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, X_train.shape[2], 3))

# Freeze some layers of VGG16 to allow partial retraining
for layer in base_model.layers[:10]:
    layer.trainable = False
for layer in base_model.layers[10:]:
    layer.trainable = True

# Build the final model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.05)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.05)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.05)))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping])

# Evaluate the model
evaluation = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

