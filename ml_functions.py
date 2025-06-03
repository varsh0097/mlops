import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from helper_functions import log_info, get_env_variable

# Configuration
ARTIFACTS_DIR = get_env_variable('ARTIFACTS_DIR', 'artifacts')

def build_model(input_shape, num_classes):
    """Create CNN model architecture for lung cancer classification"""
    log_info("Building CNN model")
    
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_model(model, train_gen, valid_gen, epochs=2):
    """Train model with callbacks and history tracking"""
    log_info("Starting model training")
    
    model_path = os.path.join(ARTIFACTS_DIR, "model.lungcancercode.h5")
    
    callbacks = [
        ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            patience=7,
            restore_best_weights=True,
            monitor='val_loss',
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=valid_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    log_info(f"Model training completed. Best model saved at {model_path}")
    return history

def evaluate_model(model, test_gen):
    """Evaluate model performance on test set"""
    log_info("Evaluating model on test set")
    loss, accuracy = model.evaluate(test_gen)
    log_info(f"Test accuracy: {accuracy*100:.2f}%")
    log_info(f"Test loss: {loss:.4f}")
    return accuracy, loss

def save_training_plots(history, save_dir):
    """Save accuracy/loss plots for training history"""
    log_info("Saving training plots")
    
    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], 'r-', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'b-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
    
    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'], 'r-', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    
    log_info(f"Training plots saved in {save_dir}")