import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from pathlib import Path

class VideoDataCollector:
    def __init__(self, data_dir="sign_language_data"):
        """Initialize video-based data collector"""
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support both hands!
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create data directory
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Storage
        self.all_data = []
        self.all_labels = []
    
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks as array"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def process_video(self, video_path, gesture_name, frame_skip=2, preview=True):
        """
        Process video and extract hand landmarks
        
        Args:
            video_path: Path to video file
            gesture_name: Name of the gesture in the video
            frame_skip: Process every Nth frame (2 = every other frame)
            preview: Show preview window while processing
        """
        print(f"\n{'='*70}")
        print(f"Processing Video: {os.path.basename(video_path)}")
        print(f"Gesture: {gesture_name}")
        print(f"{'='*70}\n")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            return 0
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📹 Video Info:")
        print(f"   Total Frames: {total_frames}")
        print(f"   FPS: {fps}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Frame Skip: {frame_skip} (processing every {frame_skip} frames)")
        print()
        
        frame_count = 0
        processed_count = 0
        samples_collected = 0
        
        print("⏳ Processing video...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames based on frame_skip parameter
            if frame_count % frame_skip != 0:
                continue
            
            processed_count += 1
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.hands.process(rgb_frame)
            
            # Extract landmarks if hands detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on frame for preview
                    if preview:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                        )
                    
                    # Extract landmarks
                    landmarks = self.extract_landmarks(hand_landmarks)
                    self.all_data.append(landmarks)
                    self.all_labels.append(gesture_name)
                    samples_collected += 1
            
            # Show preview
            if preview:
                # Add info overlay
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Samples: {samples_collected}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Video Processing - Press Q to Skip', frame)
                
                # Press Q to skip current video
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⏭️  Video processing skipped by user")
                    break
        
        cap.release()
        if preview:
            cv2.destroyAllWindows()
        
        print(f"\n✓ Video processed!")
        print(f"   Frames processed: {processed_count}/{total_frames}")
        print(f"   Samples collected: {samples_collected}")
        
        return samples_collected
    
    def process_multiple_videos(self, video_list, preview=True):
        """
        Process multiple videos
        
        Args:
            video_list: List of tuples [(video_path, gesture_name), ...]
            preview: Show preview window
        """
        print("\n" + "="*70)
        print("   VIDEO-BASED DATA COLLECTION")
        print("="*70)
        
        total_samples = 0
        
        for i, (video_path, gesture_name) in enumerate(video_list):
            print(f"\n[{i+1}/{len(video_list)}] Processing: {gesture_name}")
            samples = self.process_video(video_path, gesture_name, frame_skip=2, preview=preview)
            total_samples += samples
        
        print(f"\n{'='*70}")
        print(f"✓ All videos processed!")
        print(f"   Total samples collected: {total_samples}")
        print(f"   Unique gestures: {len(set(self.all_labels))}")
        print(f"{'='*70}")
    
    def save_dataset(self):
        """Save collected data to files"""
        if len(self.all_data) == 0:
            print("❌ No data to save!")
            return
        
        print(f"\n{'='*70}")
        print("   SAVING DATASET")
        print(f"{'='*70}\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to numpy arrays
        data_array = np.array(self.all_data)
        labels_array = np.array(self.all_labels)
        
        # Save as numpy files
        data_file = os.path.join(self.data_dir, f"hand_landmarks_{timestamp}_video.npy")
        labels_file = os.path.join(self.data_dir, f"labels_{timestamp}_video.npy")
        
        np.save(data_file, data_array)
        np.save(labels_file, labels_array)
        
        # Save as CSV
        csv_file = os.path.join(self.data_dir, f"dataset_{timestamp}_video.csv")
        with open(csv_file, 'w') as f:
            header = [f"landmark_{i}" for i in range(data_array.shape[1])]
            f.write(','.join(header) + ',label\n')
            
            for i in range(len(data_array)):
                row = ','.join(map(str, data_array[i])) + f',{labels_array[i]}\n'
                f.write(row)
        
        print(f"✓ Dataset saved successfully!")
        print(f"\n📁 Files saved:")
        print(f"   {data_file}")
        print(f"   {labels_file}")
        print(f"   {csv_file}")
        
        # Show statistics
        print(f"\n📊 Dataset statistics:")
        print(f"   Total samples: {len(self.all_data)}")
        unique, counts = np.unique(labels_array, return_counts=True)
        for gesture, count in zip(unique, counts):
            print(f"   {gesture}: {count} samples")
        
        print(f"\n{'='*70}")


def interactive_mode():
    """Interactive mode for video data collection"""
    collector = VideoDataCollector()
    
    print("\n" + "="*70)
    print("   VIDEO-BASED SIGN LANGUAGE DATA COLLECTION")
    print("="*70)
    print("\n📹 This tool extracts training data from video files!")
    print("\nHow it works:")
    print("  1. Record videos of yourself performing each gesture")
    print("  2. Provide the video file path and gesture name")
    print("  3. The tool automatically extracts hand landmarks")
    print("  4. Repeat for all gestures")
    print("  5. Save the complete dataset")
    
    video_list = []
    
    print("\n" + "="*70)
    print("Add your video files:")
    print("="*70)
    print("(Enter empty path to finish)")
    
    while True:
        print(f"\nVideo #{len(video_list) + 1}:")
        video_path = input("  Video file path: ").strip().strip('"')
        
        if not video_path:
            break
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"  ❌ File not found: {video_path}")
            retry = input("  Try again? (yes/no): ").strip().lower()
            if retry != 'yes':
                continue
            else:
                continue
        
        gesture_name = input("  Gesture name: ").strip()
        
        if not gesture_name:
            print("  ❌ Gesture name cannot be empty!")
            continue
        
        video_list.append((video_path, gesture_name))
        print(f"  ✓ Added: {gesture_name} - {os.path.basename(video_path)}")
    
    if not video_list:
        print("\n❌ No videos added. Exiting.")
        return
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    for i, (path, gesture) in enumerate(video_list):
        print(f"  {i+1}. {gesture}: {os.path.basename(path)}")
    
    confirm = input(f"\nProcess {len(video_list)} video(s)? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("❌ Cancelled.")
        return
    
    # Processing options
    print(f"\n{'='*70}")
    print("Processing Options:")
    print(f"{'='*70}")
    
    preview = input("Show preview while processing? (yes/no, default: yes): ").strip().lower()
    preview = preview != 'no'
    
    # Process videos
    collector.process_multiple_videos(video_list, preview=preview)
    
    # Save dataset
    save = input("\n💾 Save dataset? (yes/no): ").strip().lower()
    if save == 'yes':
        collector.save_dataset()
        print("\n✅ Done! You can now use this dataset to train your model.")
        print("   Run: python train_model.py")
    else:
        print("❌ Dataset not saved.")


def batch_mode():
    """Batch mode - provide video folder"""
    collector = VideoDataCollector()
    
    print("\n" + "="*70)
    print("   BATCH VIDEO PROCESSING")
    print("="*70)
    
    folder_path = input("\nEnter folder containing videos: ").strip().strip('"')
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    if not video_files:
        print("❌ No video files found in folder!")
        return
    
    print(f"\n✓ Found {len(video_files)} video file(s):")
    for i, file in enumerate(video_files):
        print(f"  {i+1}. {file}")
    
    print("\nFor each video, you'll be asked to provide the gesture name.")
    print("Video filename format: gesture_name.mp4 (optional)")
    
    video_list = []
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        
        # Try to extract gesture name from filename
        suggested_name = os.path.splitext(video_file)[0].replace('_', ' ').replace('-', ' ')
        
        print(f"\n{'='*70}")
        print(f"Video: {video_file}")
        gesture_name = input(f"Gesture name (suggested: '{suggested_name}'): ").strip()
        
        if not gesture_name:
            gesture_name = suggested_name
        
        video_list.append((video_path, gesture_name))
        print(f"✓ Added: {gesture_name}")
    
    # Process all videos
    collector.process_multiple_videos(video_list, preview=True)
    
    # Save
    collector.save_dataset()
    print("\n✅ Batch processing complete!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("   SIGN LANGUAGE VIDEO DATA COLLECTOR")
    print("="*70)
    print("\n📹 Collect training data from video files!")
    print("\nModes:")
    print("  [1] Interactive mode - Add videos one by one")
    print("  [2] Batch mode - Process all videos in a folder")
    
    choice = input("\nSelect mode (1/2): ").strip()
    
    if choice == '1':
        interactive_mode()
    elif choice == '2':
        batch_mode()
    else:
        print("❌ Invalid choice!")