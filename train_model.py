import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class SignLanguageTrainer:
    def __init__(self, data_dir="sign_language_data", model_dir="models"):
        """Initialize the trainer"""
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        self.model = None
        self.label_encoder = None
        self.history = None
        
    def load_data(self, data_file, labels_file):
        """Load the collected data"""
        print("\n=== Loading Data ===")
        print(f"Loading data from: {data_file}")
        print(f"Loading labels from: {labels_file}")
        
        # Load numpy arrays
        X = np.load(data_file)
        y = np.load(labels_file)
        
        print(f"✓ Loaded {len(X)} samples")
        print(f"✓ Feature shape: {X.shape}")
        print(f"✓ Number of unique gestures: {len(np.unique(y))}")
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\nClass distribution:")
        for gesture, count in zip(unique, counts):
            print(f"  {gesture}: {count} samples")
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.15, val_size=0.15):
        """Preprocess and split the data"""
        print("\n=== Preprocessing Data ===")
        
        # Encode labels to integers
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"✓ Encoded labels: {self.label_encoder.classes_}")
        
        # Convert to categorical (one-hot encoding)
        num_classes = len(self.label_encoder.classes_)
        y_categorical = keras.utils.to_categorical(y_encoded, num_classes)
        
        # Split data: first split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Then split train+val into train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        
        print(f"✓ Training samples: {len(X_train)}")
        print(f"✓ Validation samples: {len(X_val)}")
        print(f"✓ Testing samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, num_classes
    
    def build_model(self, input_shape, num_classes):
        """Build the neural network model"""
        print("\n=== Building Model ===")
        
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            
            # First hidden layer
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            # Second hidden layer
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            # Third hidden layer
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            
            # Output layer
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ Model architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        print("\n=== Training Model ===")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("\n✓ Training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model on test set"""
        print("\n=== Evaluating Model ===")
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"✓ Test Loss: {test_loss:.4f}")
        print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Print per-class accuracy
        print("\nPer-class accuracy:")
        for i, gesture in enumerate(self.label_encoder.classes_):
            mask = y_test_classes == i
            if np.sum(mask) > 0:
                class_accuracy = np.mean(y_pred_classes[mask] == i)
                print(f"  {gesture}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
        
        return test_accuracy
    
    def plot_training_history(self):
        """Plot training history"""
        print("\n=== Plotting Training History ===")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.model_dir, f"training_history_{timestamp}.png")
        plt.savefig(plot_file)
        print(f"✓ Training history plot saved: {plot_file}")
        
        plt.show()
    
    def save_model(self, model_name=None):
        """Save the trained model"""
        print("\n=== Saving Model ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name is None:
            model_name = f"sign_language_model_{timestamp}"
        
        # Save model
        model_file = os.path.join(self.model_dir, f"{model_name}.h5")
        self.model.save(model_file)
        print(f"✓ Model saved: {model_file}")
        
        # Save label encoder classes
        labels_file = os.path.join(self.model_dir, f"{model_name}_labels.npy")
        np.save(labels_file, self.label_encoder.classes_)
        print(f"✓ Labels saved: {labels_file}")
        
        # Save model info
        info = {
            "model_name": model_name,
            "timestamp": timestamp,
            "num_classes": len(self.label_encoder.classes_),
            "gestures": self.label_encoder.classes_.tolist(),
            "input_shape": self.model.input_shape[1],
        }
        
        info_file = os.path.join(self.model_dir, f"{model_name}_info.json")
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"✓ Model info saved: {info_file}")
        
        return model_file, labels_file


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("   SIGN LANGUAGE RECOGNITION - MODEL TRAINING")
    print("=" * 60)
    
    trainer = SignLanguageTrainer()
    
    # List available data files
    print("\nAvailable data files:")
    data_files = [f for f in os.listdir(trainer.data_dir) if f.startswith("hand_landmarks_") and f.endswith(".npy")]
    
    if not data_files:
        print("No data files found! Please run collect_data.py first.")
        return
    
    for i, file in enumerate(data_files):
        print(f"  {i+1}. {file}")
    
    # Get user selection
    if len(data_files) == 1:
        selection = 0
        print(f"\nUsing: {data_files[0]}")
    else:
        choice = input(f"\nSelect data file (1-{len(data_files)}): ").strip()
        selection = int(choice) - 1 if choice.isdigit() else 0
    
    data_file = os.path.join(trainer.data_dir, data_files[selection])
    labels_file = data_file.replace("hand_landmarks_", "labels_")
    
    # Load data
    X, y = trainer.load_data(data_file, labels_file)
    
    # Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = trainer.preprocess_data(X, y)
    
    # Build model
    input_shape = X_train.shape[1]
    trainer.build_model(input_shape, num_classes)
    
    # Get training parameters
    print("\n" + "="*60)
    epochs_input = input("Enter number of epochs (default 100): ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else 100
    
    batch_size_input = input("Enter batch size (default 32): ").strip()
    batch_size = int(batch_size_input) if batch_size_input.isdigit() else 32
    
    # Train model
    trainer.train_model(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    
    # Evaluate model
    test_accuracy = trainer.evaluate_model(X_test, y_test)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model
    model_name_input = input("\nEnter model name (press Enter for auto-generated): ").strip()
    model_name = model_name_input if model_name_input else None
    model_file, labels_file = trainer.save_model(model_name)
    
    print("\n" + "="*60)
    print("   TRAINING COMPLETE!")
    print("="*60)
    print(f"\n✓ Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"✓ Model saved to: {model_file}")
    print("\nYou can now use this model for real-time prediction!")
    print("Next step: Run the prediction script to test your model.")
    

if __name__ == "__main__":
    main()