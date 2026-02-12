import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime

class SignLanguageDataCollector:
    def __init__(self, data_dir="sign_language_data"):
        """Initialize the data collector"""
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Define your gestures here
        self.gestures = []
        self.current_gesture_index = 0
        self.samples_per_gesture = 500  # Number of samples to collect per gesture
        self.current_samples = 0
        
        # Storage for collected data
        self.all_data = []
        self.all_labels = []
        
    def setup_gestures(self):
        """Let user input the gestures they want to collect"""
        print("\n=== Sign Language Data Collection Setup ===")
        print("Enter the names of gestures you want to collect (one per line)")
        print("Press Enter on an empty line when done\n")
        
        while True:
            gesture = input(f"Gesture #{len(self.gestures) + 1}: ").strip()
            if gesture == "":
                break
            if gesture:
                self.gestures.append(gesture)
        
        if not self.gestures:
            print("No gestures entered! Adding default gestures...")
            self.gestures = ["Hello", "ThankYou", "Yes", "No", "Help"]
        
        print(f"\nGestures to collect: {', '.join(self.gestures)}")
        
        num_samples = input(f"\nSamples per gesture (default 200): ").strip()
        if num_samples.isdigit():
            self.samples_per_gesture = int(num_samples)
        
        print(f"\nWill collect {self.samples_per_gesture} samples for each gesture")
        
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks as a flattened array"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def collect_data(self):
        """Main data collection loop"""
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        collecting = False
        cooldown = 0
        
        print("\n=== Data Collection Started ===")
        print("Controls:")
        print("  SPACE - Start/Stop collecting samples")
        print("  N - Skip to next gesture")
        print("  S - Save all data and exit")
        print("  Q - Quit without saving")
        print("\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Collect data if in collection mode
                    if collecting and cooldown == 0:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        self.all_data.append(landmarks)
                        self.all_labels.append(self.gestures[self.current_gesture_index])
                        self.current_samples += 1
                        cooldown = 2  # Small cooldown to avoid duplicate captures
            
            # Handle cooldown
            if cooldown > 0:
                cooldown -= 1
            
            # Check if we've collected enough samples for current gesture
            if self.current_samples >= self.samples_per_gesture:
                print(f"✓ Completed collecting {self.samples_per_gesture} samples for '{self.gestures[self.current_gesture_index]}'")
                self.current_gesture_index += 1
                self.current_samples = 0
                collecting = False
                
                if self.current_gesture_index >= len(self.gestures):
                    print("\n🎉 All gestures collected!")
                    print("Press 'S' to save or 'Q' to quit without saving")
            
            # Display UI
            self.draw_ui(frame, collecting)
            
            # Show the frame
            cv2.imshow('Sign Language Data Collection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting without saving...")
                break
            elif key == ord(' '):
                if self.current_gesture_index < len(self.gestures):
                    collecting = not collecting
                    status = "STARTED" if collecting else "PAUSED"
                    print(f"{status} collecting for '{self.gestures[self.current_gesture_index]}'")
            elif key == ord('n'):
                if self.current_gesture_index < len(self.gestures) - 1:
                    print(f"Skipping '{self.gestures[self.current_gesture_index]}' (collected {self.current_samples} samples)")
                    self.current_gesture_index += 1
                    self.current_samples = 0
                    collecting = False
            elif key == ord('s'):
                self.save_data()
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
    
    def draw_ui(self, frame, collecting):
        """Draw UI elements on the frame"""
        h, w, _ = frame.shape
        
        # Background panel
        cv2.rectangle(frame, (10, 10), (w - 10, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 150), (255, 255, 255), 2)
        
        # Current gesture info
        if self.current_gesture_index < len(self.gestures):
            gesture = self.gestures[self.current_gesture_index]
            progress = f"{self.current_samples}/{self.samples_per_gesture}"
            
            cv2.putText(frame, f"Gesture: {gesture}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Progress: {progress}", (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Gesture {self.current_gesture_index + 1}/{len(self.gestures)}", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Collection status
            status_text = "COLLECTING" if collecting else "PAUSED"
            status_color = (0, 255, 0) if collecting else (0, 165, 255)
            cv2.putText(frame, status_text, (20, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Progress bar
            bar_width = w - 40
            progress_width = int((self.current_samples / self.samples_per_gesture) * bar_width)
            cv2.rectangle(frame, (20, h - 40), (20 + bar_width, h - 20), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, h - 40), (20 + progress_width, h - 20), (0, 255, 0), -1)
        else:
            cv2.putText(frame, "All gestures collected! Press 'S' to save", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def save_data(self):
        """Save collected data to files"""
        if len(self.all_data) == 0:
            print("No data to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to numpy arrays
        data_array = np.array(self.all_data)
        labels_array = np.array(self.all_labels)
        
        # Save as numpy files
        data_file = os.path.join(self.data_dir, f"hand_landmarks_{timestamp}.npy")
        labels_file = os.path.join(self.data_dir, f"labels_{timestamp}.npy")
        
        np.save(data_file, data_array)
        np.save(labels_file, labels_array)
        
        # Also save as CSV for easy inspection
        csv_file = os.path.join(self.data_dir, f"dataset_{timestamp}.csv")
        with open(csv_file, 'w') as f:
            # Write header
            header = [f"landmark_{i}" for i in range(data_array.shape[1])]
            f.write(','.join(header) + ',label\n')
            
            # Write data
            for i in range(len(data_array)):
                row = ','.join(map(str, data_array[i])) + f',{labels_array[i]}\n'
                f.write(row)
        
        print(f"\n✓ Data saved successfully!")
        print(f"  - NumPy data: {data_file}")
        print(f"  - NumPy labels: {labels_file}")
        print(f"  - CSV file: {csv_file}")
        print(f"  - Total samples: {len(self.all_data)}")
        print(f"  - Unique gestures: {len(set(self.all_labels))}")
        
        # Print statistics
        print("\nDataset statistics:")
        unique, counts = np.unique(labels_array, return_counts=True)
        for gesture, count in zip(unique, counts):
            print(f"  {gesture}: {count} samples")


if __name__ == "__main__":
    collector = SignLanguageDataCollector()
    collector.setup_gestures()
    
    input("\nPress Enter to start data collection...")
    collector.collect_data()
    
    print("\nData collection complete!")