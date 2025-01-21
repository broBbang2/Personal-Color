# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import resample
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

# GPU configuration for Mac (if available)
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.set_logical_device_configuration(device, [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
        print("GPU is being used for training.")
    else:
        print("No GPU found. Training will use CPU.")
except Exception as e:
    print(f"Error while setting GPU configuration: {e}")

# Function to load images and labels from a directory
def load_images_and_labels(image_dir):
    valid_images = []
    labels = []
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    for root, _, files in os.walk(image_dir):
        for file in files:
            if file == ".DS_Store":
                continue

            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file_path)[-1].lower()

            if file_ext not in supported_formats:
                continue

            try:
                with Image.open(file_path) as image:
                    image.verify()
                with Image.open(file_path) as image:
                    image = image.convert('RGB')
                    image = image.resize((64, 64))
                    valid_images.append(np.array(image))
                    label = os.path.basename(root)
                    labels.append(label)
            except (UnidentifiedImageError, IsADirectoryError):
                continue
            except Exception as e:
                print(f"Unexpected error with image {file_path}: {e}")

    return np.array(valid_images), np.array(labels)

# Load data
image_dir = os.path.expanduser('/Users/breeadee/Desktop/new/personal color')
valid_images, valid_labels = load_images_and_labels(image_dir)

# Encode labels
label_encoder = LabelEncoder()
valid_labels = label_encoder.fit_transform(valid_labels)

# Split data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(valid_images, valid_labels, test_size=0.2, random_state=42)

# Function to resample dataset
def limit_dataset(images, labels, limit=10000):
    unique_labels = np.unique(labels)
    resampled_images = []
    resampled_labels = []

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        label_images = images[label_indices]
        label_labels = labels[label_indices]

        if len(label_images) < limit:
            label_images, label_labels = resample(label_images, label_labels, replace=True, n_samples=limit, random_state=42)
        elif len(label_images) > limit:
            indices = np.random.choice(len(label_images), limit, replace=False)
            label_images = label_images[indices]
            label_labels = label_labels[indices]

        resampled_images.append(label_images)
        resampled_labels.append(label_labels)

    resampled_images = np.concatenate(resampled_images)
    resampled_labels = np.concatenate(resampled_labels)

    return resampled_images, resampled_labels

train_images, train_labels = limit_dataset(train_images, train_labels, limit=10000)

# Display class distribution after resampling
unique, counts = np.unique(train_labels, return_counts=True)
print("Class distribution after resampling:")
for label, count in zip(unique, counts):
    print(f"Class '{label}': {count} samples")

# One-hot encode labels
one_hot_encoder = OneHotEncoder(sparse_output=False)
train_labels = one_hot_encoder.fit_transform(train_labels.reshape(-1, 1))
val_labels = one_hot_encoder.transform(val_labels.reshape(-1, 1))

# Data augmentation
data_gen_args = {
    'rescale': 1./255,
    'rotation_range': 30,
    'width_shift_range': 0.3,
    'height_shift_range': 0.3,
    'shear_range': 0.3,
    'zoom_range': 0.3,
    'brightness_range': [0.7, 1.3],
    'horizontal_flip': True,
    'fill_mode': 'nearest',
    'channel_shift_range': 0.1
}

train_data_generator = ImageDataGenerator(**data_gen_args)
validation_data_generator = ImageDataGenerator(rescale=1./255)

# Create data generators
train_data = train_data_generator.flow(train_images, train_labels, batch_size=64)
validation_data = validation_data_generator.flow(val_images, val_labels, batch_size=32)

# Function to create the model
def create_model(num_classes):
    model = Sequential([
        Conv2D(16, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.004), input_shape=(64, 64, 3)),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.2),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.004)),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.2),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.004)),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.4),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.004)),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Callbacks
def scheduler(epoch, lr):
    return lr if epoch < 10 else lr * tf.math.exp(-0.2).numpy()

lr_scheduler = LearningRateScheduler(scheduler)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=8, min_lr=1e-6)

# Compile and train the model
model = create_model(num_classes=len(np.unique(valid_labels)))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=130,
    callbacks=[lr_scheduler, reduce_lr_on_plateau]
)

# Save the model
model.save_weights('/Users/breeadee/Desktop/new/personal_color_model.weights.h5')
model.save('/Users/breeadee/Desktop/new/personal_color_model.h5')
print("Model saved successfully!")

# Visualize training history
train_acc = history.history.get('accuracy')
val_acc = history.history.get('val_accuracy')
train_loss = history.history.get('loss')
val_loss = history.history.get('val_loss')

plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

# Best epoch and accuracy
best_epoch = np.argmax(history.history['val_accuracy']) + 1
best_accuracy = max(history.history['val_accuracy'])
print(f"Best Validation Accuracy: {best_accuracy:.4f} at Epoch {best_epoch}")
