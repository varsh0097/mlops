import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from helper_functions import log_info, get_env_variable, save_artifact

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = get_env_variable('DATA_DIR', os.path.join(BASE_DIR, 'mlops', 'data'))
ARTIFACTS_DIR = get_env_variable('ARTIFACTS_DIR', os.path.join(BASE_DIR, 'artifacts'))
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

def create_data_generators():
    """Create data generators with preprocessing and augmentation"""
    log_info("Creating data generators")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_dir = train_dir = r"C:\Users\visha\OneDrive\Desktop\mlops\data\train"
    valid_dir = r"C:\Users\visha\OneDrive\Desktop\mlops\data\valid"
    test_dir = r"C:\Users\visha\OneDrive\Desktop\mlops\data\test"
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    valid_generator = test_val_datagen.flow_from_directory(
        valid_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Save class indices for inference
    class_indices_path = os.path.join(ARTIFACTS_DIR, "class_indices.pkl")
    save_artifact(train_generator.class_indices, class_indices_path)
    
    return train_generator, valid_generator, test_generator

def visualize_data_samples(generator, save_path=None):
    """Display and save sample images from dataset"""
    images, labels = next(generator)
    class_names = list(generator.class_indices.keys())
    
    plt.figure(figsize=(12, 12))
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis("off")
    
    if save_path:
        plt.savefig(os.path.join(save_path, "data_samples.png"))
        log_info(f"Sample images saved at {save_path}/data_samples.png")
    
    plt.tight_layout()
    plt.show()