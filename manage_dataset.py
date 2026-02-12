import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

class DatasetManager:
    def __init__(self, data_dir="sign_language_data"):
        """Initialize the dataset manager"""
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Current dataset
        self.X = None
        self.y = None
        self.data_file = None
        self.labels_file = None
        
        # For data collection
        self.temp_samples = []
        
    def load_dataset(self):
        """Load existing dataset"""
        print("\n" + "=" * 70)
        print("   SIGN LANGUAGE DATASET MANAGER")
        print("=" * 70)
        
        # List available data files
        data_files = [f for f in os.listdir(self.data_dir) 
                     if f.startswith("hand_landmarks_") and f.endswith(".npy")]
        
        if not data_files:
            print("\n❌ No existing datasets found!")
            print("Please run collect_data.py first to create a dataset.")
            return False
        
        print("\n📁 Available datasets:")
        for i, file in enumerate(data_files):
            print(f"  {i+1}. {file}")
        
        # Get user selection
        if len(data_files) == 1:
            selection = 0
            print(f"\n✓ Using: {data_files[0]}")
        else:
            while True:
                choice = input(f"\nSelect dataset (1-{len(data_files)}): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(data_files):
                    selection = int(choice) - 1
                    break
                print("Invalid selection. Try again.")
        
        self.data_file = os.path.join(self.data_dir, data_files[selection])
        self.labels_file = self.data_file.replace("hand_landmarks_", "labels_")
        
        # Load data
        print(f"\n📂 Loading: {self.data_file}")
        self.X = np.load(self.data_file)
        self.y = np.load(self.labels_file)
        
        print(f"✓ Loaded {len(self.X)} samples")
        
        return True
    
    def display_gestures(self):
        """Display all gestures in the dataset"""
        print("\n" + "=" * 70)
        print("   CURRENT GESTURES IN DATASET")
        print("=" * 70)
        
        unique_gestures = np.unique(self.y)
        
        print(f"\n📊 Total gestures: {len(unique_gestures)}")
        print(f"📊 Total samples: {len(self.X)}\n")
        
        for i, gesture in enumerate(unique_gestures):
            count = np.sum(self.y == gesture)
            print(f"  [{i+1}] {gesture:<20} - {count} samples")
        
        print("\n" + "=" * 70)
        
        return unique_gestures
    
    def delete_gesture(self, gestures):
        """Delete a gesture from the dataset"""
        print("\n" + "=" * 70)
        print("   DELETE GESTURE")
        print("=" * 70)
        
        print("\n📋 Available gestures:")
        for i, gesture in enumerate(gestures):
            count = np.sum(self.y == gesture)
            print(f"  [{i+1}] {gesture} ({count} samples)")
        
        # Get gesture to delete
        while True:
            choice = input(f"\nEnter gesture number to delete (1-{len(gestures)}) or 0 to cancel: ").strip()
            if choice.isdigit():
                choice_num = int(choice)
                if choice_num == 0:
                    print("❌ Delete cancelled")
                    return
                if 1 <= choice_num <= len(gestures):
                    gesture_to_delete = gestures[choice_num - 1]
                    break
            print("Invalid selection. Try again.")
        
        # Confirm deletion
        count = np.sum(self.y == gesture_to_delete)
        confirm = input(f"\n⚠️  Delete '{gesture_to_delete}' ({count} samples)? (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            print("❌ Delete cancelled")
            return
        
        # Delete gesture
        mask = self.y != gesture_to_delete
        self.X = self.X[mask]
        self.y = self.y[mask]
        
        print(f"\n✓ Deleted '{gesture_to_delete}' ({count} samples)")
        print(f"✓ Remaining samples: {len(self.X)}")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks as a flattened array"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def collect_samples(self, gesture_name, num_samples):
        """Collect new samples for a gesture"""
        print(f"\n📹 Opening camera to collect {num_samples} samples for '{gesture_name}'...")
        print("\nControls:")
        print("  SPACE - Start/Stop collecting")
        print("  Q - Quit collection")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        collecting = False
        samples_collected = 0
        self.temp_samples = []
        cooldown = 0
        
        input("\nPress Enter to open camera...")
        
        while cap.isOpened() and samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
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
                        self.temp_samples.append(landmarks)
                        samples_collected += 1
                        cooldown = 2
            
            if cooldown > 0:
                cooldown -= 1
            
            # Draw UI
            h, w, _ = frame.shape
            cv2.rectangle(frame, (10, 10), (w - 10, 150), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (w - 10, 150), (255, 255, 255), 2)
            
            cv2.putText(frame, f"Gesture: {gesture_name}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Progress: {samples_collected}/{num_samples}", (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            status_text = "COLLECTING" if collecting else "PAUSED"
            status_color = (0, 255, 0) if collecting else (0, 165, 255)
            cv2.putText(frame, status_text, (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Progress bar
            bar_width = w - 40
            progress_width = int((samples_collected / num_samples) * bar_width)
            cv2.rectangle(frame, (20, h - 40), (20 + bar_width, h - 20), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, h - 40), (20 + progress_width, h - 20), (0, 255, 0), -1)
            
            cv2.imshow('Dataset Manager - Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                collecting = not collecting
                print(f"{'▶️  STARTED' if collecting else '⏸️  PAUSED'} collecting")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Collected {len(self.temp_samples)} samples")
        return len(self.temp_samples)
    
    def modify_gesture(self, gestures):
        """Modify/add samples to a gesture"""
        print("\n" + "=" * 70)
        print("   MODIFY GESTURE")
        print("=" * 70)
        
        print("\n📋 Available gestures:")
        for i, gesture in enumerate(gestures):
            count = np.sum(self.y == gesture)
            print(f"  [{i+1}] {gesture} ({count} samples)")
        print(f"  [{len(gestures)+1}] Add NEW gesture")
        
        # Get gesture to modify
        while True:
            choice = input(f"\nSelect gesture (1-{len(gestures)+1}) or 0 to cancel: ").strip()
            if choice.isdigit():
                choice_num = int(choice)
                if choice_num == 0:
                    print("❌ Modify cancelled")
                    return
                if 1 <= choice_num <= len(gestures) + 1:
                    break
            print("Invalid selection. Try again.")
        
        # Get gesture name
        if choice_num == len(gestures) + 1:
            gesture_name = input("\n✏️  Enter NEW gesture name: ").strip()
            if not gesture_name:
                print("❌ Invalid gesture name")
                return
            current_count = 0
        else:
            gesture_name = gestures[choice_num - 1]
            current_count = np.sum(self.y == gesture_name)
        
        print(f"\n📊 Current samples for '{gesture_name}': {current_count}")
        
        # Get number of samples to add
        num_samples_input = input("How many samples to add (default 200): ").strip()
        num_samples = int(num_samples_input) if num_samples_input.isdigit() else 200
        
        # Collect samples
        collected = self.collect_samples(gesture_name, num_samples)
        
        if collected > 0:
            # Add to dataset
            new_X = np.array(self.temp_samples)
            new_y = np.array([gesture_name] * len(self.temp_samples))
            
            self.X = np.vstack([self.X, new_X])
            self.y = np.concatenate([self.y, new_y])
            
            print(f"\n✓ Added {collected} samples to '{gesture_name}'")
            print(f"✓ Total samples for '{gesture_name}': {np.sum(self.y == gesture_name)}")
            print(f"✓ Total dataset size: {len(self.X)}")
    
    def save_dataset(self):
        """Save the modified dataset"""
        print("\n" + "=" * 70)
        print("   SAVE DATASET")
        print("=" * 70)
        
        save = input("\n💾 Save changes to dataset? (yes/no): ").strip().lower()
        
        if save != 'yes':
            print("❌ Changes not saved")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save files
        new_data_file = os.path.join(self.data_dir, f"hand_landmarks_{timestamp}_modified.npy")
        new_labels_file = os.path.join(self.data_dir, f"labels_{timestamp}_modified.npy")
        new_csv_file = os.path.join(self.data_dir, f"dataset_{timestamp}_modified.csv")
        
        np.save(new_data_file, self.X)
        np.save(new_labels_file, self.y)
        
        # Save CSV
        with open(new_csv_file, 'w') as f:
            header = [f"landmark_{i}" for i in range(self.X.shape[1])]
            f.write(','.join(header) + ',label\n')
            
            for i in range(len(self.X)):
                row = ','.join(map(str, self.X[i])) + f',{self.y[i]}\n'
                f.write(row)
        
        print(f"\n✓ Dataset saved:")
        print(f"  📄 {new_data_file}")
        print(f"  📄 {new_labels_file}")
        print(f"  📄 {new_csv_file}")
        
        # Show final statistics
        print(f"\n📊 Final dataset statistics:")
        unique_gestures = np.unique(self.y)
        for gesture in unique_gestures:
            count = np.sum(self.y == gesture)
            print(f"  {gesture}: {count} samples")
        print(f"\n  Total: {len(self.X)} samples")
    
    def run(self):
        """Main interactive loop"""
        if not self.load_dataset():
            return
        
        while True:
            gestures = self.display_gestures()
            
            print("\n📋 OPTIONS:")
            print("  [1] Modify / Add gesture")
            print("  [2] Delete gesture")
            print("  [3] Quit and Save")
            print("  [4] Quit without Saving")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                self.modify_gesture(gestures)
            elif choice == '2':
                self.delete_gesture(gestures)
            elif choice == '3':
                self.save_dataset()
                break
            elif choice == '4':
                confirm = input("\n⚠️  Quit without saving? All changes will be lost! (yes/no): ").strip().lower()
                if confirm == 'yes':
                    print("\n❌ Exiting without saving")
                    break
            else:
                print("\n❌ Invalid option. Try again.")
        
        self.hands.close()
        print("\n" + "=" * 70)
        print("   DATASET MANAGER CLOSED")
        print("=" * 70)


if __name__ == "__main__":
    manager = DatasetManager()
    manager.run()