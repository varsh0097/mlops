import os
import numpy as np
from data_preprocessing import create_data_generators, visualize_data_samples
from ml_functions import build_model, train_model, evaluate_model, save_training_plots
from helper_functions import setup_logging, get_env_variable

def main():
    # Setup environment
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ARTIFACTS_DIR = get_env_variable('ARTIFACTS_DIR', os.path.join(BASE_DIR,'mlops' 'artifacts'))
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    setup_logging()
    
    # Data preparation
    train_gen, valid_gen, test_gen = create_data_generators()
    visualize_data_samples(train_gen, save_path=ARTIFACTS_DIR)
    
    # Model building
    num_classes = len(train_gen.class_indices)
    model = build_model(
        input_shape=(256, 256, 3),
        num_classes=num_classes
    )
    
    # Training
    history = train_model(model, train_gen, valid_gen, epochs=3)
    
    # Evaluation
    evaluate_model(model, test_gen)
    
    # Save artifacts
    save_training_plots(history, ARTIFACTS_DIR)
    
    print("\n" + "="*50)
    print("Training pipeline completed successfully!")
    print(f"Artifacts saved in: {ARTIFACTS_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()