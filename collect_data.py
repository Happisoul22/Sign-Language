import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

class DataCollector:
    def __init__(self, data_dir="sign_language_data"):
        """Initialize data collector"""
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support 2 hands!
            min_detection_confidence=0.5,  # Lower threshold for better detection
            min_tracking_confidence=0.5
        )
        
        # Create data directory
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Storage
        self.all_data = []
        self.all_labels = []
    
    def list_existing_datasets(self):
        """List existing datasets"""
        data_files = [f for f in os.listdir(self.data_dir) 
                     if f.startswith('hand_landmarks_') and f.endswith('.npy')]
        
        if not data_files:
            return []
        
        data_files.sort(reverse=True)
        return data_files
    
    def load_existing_dataset(self, data_file):
        """Load existing dataset"""
        data_path = os.path.join(self.data_dir, data_file)
        labels_file = data_file.replace('hand_landmarks_', 'labels_')
        labels_path = os.path.join(self.data_dir, labels_file)
        
        if not os.path.exists(labels_path):
            print(f"❌ Labels file not found: {labels_file}")
            return None, None
        
        data = np.load(data_path)
        labels = np.load(labels_path)
        
        print(f"\n✓ Loaded existing dataset: {data_file}")
        print(f"   Total samples: {len(data)}")
        
        unique, counts = np.unique(labels, return_counts=True)
        print(f"   Existing gestures: {len(unique)}")
        for gesture, count in zip(unique, counts):
            print(f"      - {gesture}: {count} samples")
        
        return data.tolist(), labels.tolist()
    
    def extract_landmarks(self, hand_landmarks_list):
        """Extract hand landmarks - uses first detected hand"""
        if not hand_landmarks_list:
            return None
        
        # Use first hand for consistency
        hand_landmarks = hand_landmarks_list[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def collect_data(self, gestures, samples_per_gesture, extend_mode=False, existing_data=None, existing_labels=None):
        """Collect data from webcam"""
        if extend_mode and existing_data is not None:
            self.all_data = existing_data
            self.all_labels = existing_labels
            print(f"\n📦 Extending existing dataset!")
            print(f"   Starting with {len(existing_data)} existing samples")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        current_gesture_idx = 0
        collecting = False
        samples_collected = 0
        
        print(f"\n{'='*70}")
        print("   DATA COLLECTION STARTED")
        print(f"{'='*70}\n")
        print("Controls:")
        print("  SPACE - Start/Stop collecting")
        print("  N - Next gesture")
        print("  S - Save and exit")
        print("  Q - Quit without saving")
        print(f"{'='*70}\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Check if all gestures are completed
            if current_gesture_idx >= len(gestures):
                # All gestures done - show completion message
                cv2.putText(frame, "ALL GESTURES COLLECTED!", (10, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, "Press 'S' to SAVE or 'Q' to QUIT", (10, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    print("\nSaving data...")
                    break
                elif key == ord('q'):
                    print("\nQuitting without saving...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
                continue
            
            # Current gesture info
            gesture_name = gestures[current_gesture_idx]
            
            # Draw info with backgrounds for better visibility
            status = "COLLECTING" if collecting else "PAUSED"
            color = (0, 255, 0) if collecting else (0, 0, 255)
            
            # Count detected hands
            hands_detected = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            hands_color = (0, 255, 0) if hands_detected > 0 else (0, 0, 255)
            
            # Helper function to draw text with background
            def draw_text_with_bg(img, text, pos, font_scale=1, thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                x, y = pos
                # Draw background rectangle
                cv2.rectangle(img, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), bg_color, -1)
                # Draw text
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
            
            # Draw all info with backgrounds
            draw_text_with_bg(frame, f"Gesture: {gesture_name}", (10, 30), 0.8, 2)
            draw_text_with_bg(frame, f"Progress: {samples_collected}/{samples_per_gesture}", (10, 70), 0.8, 2)
            draw_text_with_bg(frame, f"Gesture {current_gesture_idx + 1}/{len(gestures)}", (10, 110), 0.8, 2)
            draw_text_with_bg(frame, status, (10, 150), 0.8, 2, color)
            draw_text_with_bg(frame, f"Hands: {hands_detected}", (10, 190), 0.8, 2, hands_color)
            
            # Draw hand landmarks (ALWAYS, even when paused!)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
            
            # Collect data only when actively collecting
            if results.multi_hand_landmarks and collecting:
                # Extract from first hand only (for consistency)
                landmarks = self.extract_landmarks(results.multi_hand_landmarks)
                if landmarks:
                    self.all_data.append(landmarks)
                    self.all_labels.append(gesture_name)
                    samples_collected += 1
                    
                    if samples_collected >= samples_per_gesture:
                        print(f"✓ Completed collecting {samples_per_gesture} samples for '{gesture_name}'")
                        samples_collected = 0
                        collecting = False
                        current_gesture_idx += 1
                        print("\n✓ All gestures collected!" if current_gesture_idx >= len(gestures) else "")
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - Start/Stop
                if current_gesture_idx < len(gestures):
                    collecting = not collecting
                    status_msg = "started" if collecting else "paused"
                    print(f"Collection {status_msg} for '{gesture_name}'")
            
            elif key == ord('n'):  # Next gesture
                if current_gesture_idx < len(gestures) - 1:
                    current_gesture_idx += 1
                    samples_collected = 0
                    collecting = False
                    print(f"Skipped to gesture: {gestures[current_gesture_idx]}")
            
            elif key == ord('s'):  # Save
                print("\nSaving data...")
                break
            
            elif key == ord('q'):  # Quit
                print("\nQuitting without saving...")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def save_data(self, filename_suffix=""):
        """Save collected data"""
        if len(self.all_data) == 0:
            print("❌ No data to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to numpy arrays
        data_array = np.array(self.all_data)
        labels_array = np.array(self.all_labels)
        
        # Save files
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        data_file = os.path.join(self.data_dir, f"hand_landmarks_{timestamp}{suffix}.npy")
        labels_file = os.path.join(self.data_dir, f"labels_{timestamp}{suffix}.npy")
        csv_file = os.path.join(self.data_dir, f"dataset_{timestamp}{suffix}.csv")
        
        np.save(data_file, data_array)
        np.save(labels_file, labels_array)
        
        # Save CSV
        with open(csv_file, 'w') as f:
            header = [f"landmark_{i}" for i in range(data_array.shape[1])]
            f.write(','.join(header) + ',label\n')
            
            for i in range(len(data_array)):
                row = ','.join(map(str, data_array[i])) + f',{labels_array[i]}\n'
                f.write(row)
        
        print(f"\n{'='*70}")
        print("   DATA SAVED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"✓ Data: {data_file}")
        print(f"✓ Labels: {labels_file}")
        print(f"✓ CSV: {csv_file}")
        print(f"\n📊 Dataset statistics:")
        print(f"   Total samples: {len(self.all_data)}")
        
        unique, counts = np.unique(labels_array, return_counts=True)
        for gesture, count in zip(unique, counts):
            print(f"   {gesture}: {count} samples")
        
        print(f"{'='*70}\n")


def main():
    """Main function"""
    collector = DataCollector()
    
    print("\n" + "="*70)
    print("   SIGN LANGUAGE DATA COLLECTION")
    print("="*70)
    
    # Check for existing datasets
    existing_datasets = collector.list_existing_datasets()
    
    extend_mode = False
    existing_data = None
    existing_labels = None
    
    if existing_datasets:
        print("\n📦 Found existing datasets:")
        for i, dataset in enumerate(existing_datasets):
            print(f"  [{i+1}] {dataset}")
        
        print("\n" + "="*70)
        print("Do you want to:")
        print("  [1] Create NEW dataset (start fresh)")
        print("  [2] EXTEND existing dataset (add more gestures)")
        print("="*70)
        
        choice = input("Your choice (1/2): ").strip()
        
        if choice == '2':
            # Select dataset to extend
            dataset_idx = input(f"Select dataset to extend (1-{len(existing_datasets)}): ").strip()
            try:
                idx = int(dataset_idx) - 1
                if 0 <= idx < len(existing_datasets):
                    existing_data, existing_labels = collector.load_existing_dataset(existing_datasets[idx])
                    if existing_data is not None:
                        extend_mode = True
                        print("\n✓ Will extend this dataset with NEW gestures!")
                    else:
                        print("❌ Failed to load dataset. Creating new instead.")
                else:
                    print("❌ Invalid selection. Creating new dataset.")
            except:
                print("❌ Invalid input. Creating new dataset.")
    
    # Get gestures to collect
    print("\n" + "="*70)
    print("Enter gesture names (one per line)")
    print("Press Enter on empty line when done")
    print("="*70)
    
    if extend_mode:
        print("⚠️  Enter ONLY the NEW gestures you want to ADD")
        print("    (Old gestures are already in the dataset)")
    
    gestures = []
    gesture_num = 1
    
    while True:
        gesture = input(f"Gesture #{gesture_num}: ").strip()
        if not gesture:
            break
        gestures.append(gesture)
        gesture_num += 1
    
    if not gestures:
        print("❌ No gestures entered!")
        return
    
    print(f"\n{'='*70}")
    if extend_mode:
        print(f"Will ADD these NEW gestures to existing dataset:")
    else:
        print(f"Gestures to collect:")
    for g in gestures:
        print(f"  - {g}")
    print(f"{'='*70}")
    
    # Samples per gesture
    samples_input = input("\nSamples per gesture (default 200): ").strip()
    samples_per_gesture = int(samples_input) if samples_input else 200
    
    print(f"\nWill collect {samples_per_gesture} samples for each gesture")
    
    input("\nPress Enter to start data collection...")
    
    # Collect data
    if collector.collect_data(gestures, samples_per_gesture, extend_mode, existing_data, existing_labels):
        # Save data
        suffix = "extended" if extend_mode else ""
        collector.save_data(suffix)
        print("✅ Data collection complete!")
        print("\n🎯 Next step: Train your model")
        print("   Run: python train_model.py")
    else:
        print("❌ Collection cancelled")


if __name__ == "__main__":
    main()