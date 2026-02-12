import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json
from collections import deque
import time

class SignLanguagePredictor:
    def __init__(self, model_path, labels_path):
        """Initialize the predictor"""
        print("=== Initializing Sign Language Predictor ===")
        
        # Load the trained model
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Load labels
        print(f"Loading labels from: {labels_path}")
        self.labels = np.load(labels_path)
        print(f"✓ Labels loaded: {self.labels}")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=10)  # Store last 10 predictions
        self.confidence_threshold = 0.7
        
        # Sentence building
        self.sentence = []
        self.last_gesture = None
        self.gesture_stable_count = 0
        self.stability_threshold = 15  # Frames needed to confirm a gesture
        
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks as a flattened array"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks).reshape(1, -1)
    
    def predict_gesture(self, landmarks):
        """Predict gesture from landmarks"""
        # Get model prediction
        prediction = self.model.predict(landmarks, verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        # Add to buffer for smoothing
        if confidence > self.confidence_threshold:
            self.prediction_buffer.append((predicted_class, confidence))
        
        # Get most common prediction from buffer
        if len(self.prediction_buffer) > 0:
            classes = [pred[0] for pred in self.prediction_buffer]
            most_common = max(set(classes), key=classes.count)
            avg_confidence = np.mean([pred[1] for pred in self.prediction_buffer if pred[0] == most_common])
            
            return self.labels[most_common], avg_confidence, prediction
        
        return None, 0.0, prediction
    
    def add_to_sentence(self, gesture):
        """Add gesture to sentence with stability check"""
        if gesture == self.last_gesture:
            self.gesture_stable_count += 1
        else:
            self.gesture_stable_count = 0
            self.last_gesture = gesture
        
        # Add to sentence if gesture is stable
        if self.gesture_stable_count == self.stability_threshold:
            if gesture not in ['s']:  # 's' might be a mistake, filter if needed
                self.sentence.append(gesture)
                self.gesture_stable_count = 0
                return True
        
        return False
    
    def draw_ui(self, frame, gesture, confidence, all_predictions):
        """Draw UI elements on frame"""
        h, w, _ = frame.shape
        
        # Main prediction panel
        panel_height = 180
        cv2.rectangle(frame, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, panel_height), (255, 255, 255), 2)
        
        if gesture:
            # Display predicted gesture
            cv2.putText(frame, f"Gesture: {gesture}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Display confidence
            conf_text = f"Confidence: {confidence:.2%}"
            cv2.putText(frame, conf_text, (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = int((w - 40) * confidence)
            cv2.rectangle(frame, (20, 110), (20 + bar_width, 130), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, 110), (w - 20, 130), (255, 255, 255), 2)
            
            # Show all predictions
            cv2.putText(frame, "All Predictions:", (20, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "No hand detected", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        
        # Show top 3 predictions
        if all_predictions is not None:
            top_3_idx = np.argsort(all_predictions)[-3:][::-1]
            x_offset = 200
            for idx in top_3_idx:
                pred_text = f"{self.labels[idx]}: {all_predictions[idx]:.1%}"
                cv2.putText(frame, pred_text, (x_offset, 155), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                x_offset += 200
        
        # Sentence panel
        if len(self.sentence) > 0:
            sentence_panel_top = h - 100
            cv2.rectangle(frame, (10, sentence_panel_top), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, sentence_panel_top), (w - 10, h - 10), (255, 255, 255), 2)
            
            cv2.putText(frame, "Sentence:", (20, sentence_panel_top + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            sentence_text = " ".join(self.sentence[-5:])  # Show last 5 words
            cv2.putText(frame, sentence_text, (20, sentence_panel_top + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Instructions
        instructions = [
            "SPACE - Add to sentence  |  C - Clear sentence  |  Q - Quit"
        ]
        cv2.putText(frame, instructions[0], (20, h - 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Run real-time prediction"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n=== Real-Time Sign Language Translation ===")
        print("Controls:")
        print("  SPACE - Manually add current gesture to sentence")
        print("  C - Clear sentence")
        print("  Q - Quit")
        print("\nShow your gestures to the camera!")
        print("=" * 50)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.hands.process(rgb_frame)
            
            gesture = None
            confidence = 0.0
            all_predictions = None
            
            # If hand detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    # Extract landmarks and predict
                    landmarks = self.extract_landmarks(hand_landmarks)
                    gesture, confidence, all_predictions = self.predict_gesture(landmarks)
                    
                    # Auto-add to sentence if stable
                    if gesture:
                        added = self.add_to_sentence(gesture)
                        if added:
                            print(f"✓ Added '{gesture}' to sentence")
            else:
                # Reset stability if no hand
                self.gesture_stable_count = 0
                self.last_gesture = None
                self.prediction_buffer.clear()
            
            # Draw UI
            self.draw_ui(frame, gesture, confidence, all_predictions)
            
            # Show frame
            cv2.imshow('Sign Language Translator', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                if gesture:
                    self.sentence.append(gesture)
                    print(f"✓ Manually added '{gesture}' to sentence")
            elif key == ord('c'):
                self.sentence.clear()
                print("✓ Sentence cleared")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Final sentence
        if len(self.sentence) > 0:
            print("\n" + "=" * 50)
            print("Final Sentence:")
            print(" ".join(self.sentence))
            print("=" * 50)


def list_available_models(model_dir="models"):
    """List all available trained models"""
    if not os.path.exists(model_dir):
        return []
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    return model_files


def main():
    print("=" * 60)
    print("   SIGN LANGUAGE TRANSLATOR - REAL-TIME PREDICTION")
    print("=" * 60)
    
    model_dir = "models"
    
    # List available models
    model_files = list_available_models(model_dir)
    
    if not model_files:
        print("\nNo trained models found!")
        print("Please run train_model.py first to train a model.")
        return
    
    print("\nAvailable models:")
    for i, model_file in enumerate(model_files):
        print(f"  {i+1}. {model_file}")
    
    # Get user selection
    if len(model_files) == 1:
        selection = 0
        print(f"\nUsing: {model_files[0]}")
    else:
        choice = input(f"\nSelect model (1-{len(model_files)}): ").strip()
        selection = int(choice) - 1 if choice.isdigit() else 0
    
    model_file = os.path.join(model_dir, model_files[selection])
    labels_file = model_file.replace('.h5', '_labels.npy')
    
    # Check if labels file exists
    if not os.path.exists(labels_file):
        print(f"\nError: Labels file not found: {labels_file}")
        return
    
    # Initialize predictor
    predictor = SignLanguagePredictor(model_file, labels_file)
    
    input("\nPress Enter to start the translator...")
    
    # Run prediction
    predictor.run()
    
    print("\nTranslation session ended.")


if __name__ == "__main__":
    main()