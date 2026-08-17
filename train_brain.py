import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# ==========================================
# 0. AUTOMATIC DATASET CLEANUP
# ==========================================
print("[INFO] Sweeping dataset for broken or hidden files...")

dataset_dir = 'dataset'
removed_count = 0

for folder_name in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder_name)

    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            try:
                with Image.open(file_path) as img:
                    img.verify()

            except Exception:
                print(f"[WARNING] Removing corrupted file: {file_path}")

                try:
                    os.remove(file_path)
                    removed_count += 1
                except:
                    pass

print(f"[INFO] Sweep complete. Removed {removed_count} bad files.")

# ==========================================
# 1. PREPARE THE DATASET
# ==========================================
print("[INFO] Loading and augmenting dataset...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ==========================================
# 2. BUILD THE CUSTOM CNN ARCHITECTURE
# ==========================================
print("[INFO] Building the Custom CNN model...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(512, activation='relu'),
    Dropout(0.5),

    Dense(5, activation='softmax')
])

# ==========================================
# 3. COMPILE AND TRAIN
# ==========================================
print("[INFO] Compiling model...")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("[INFO] Starting the training phase (This will take some time)...")

history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator
)

# ==========================================
# 4. SAVE THE MODEL
# ==========================================
print("[INFO] Saving the trained AI brain...")

model.save("custom_flower_model.h5")

print("[INFO] Training complete! 'custom_flower_model.h5' has been generated.")
