import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from collections import deque
import pyttsx3
import threading
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .gesture-display {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        padding: 20px;
        background-color: #f0f0f0;
        border-radius: 10px;
        margin: 20px 0;
    }
    .sentence-display {
        font-size: 1.8rem;
        color: #333;
        padding: 20px;
        background-color: #e3f2fd;
        border-radius: 10px;
        min-height: 80px;
        margin: 20px 0;
    }
    .confidence-bar {
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.2rem;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'sentence' not in st.session_state:
    st.session_state.sentence = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'current_gesture' not in st.session_state:
    st.session_state.current_gesture = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0

class SignLanguageTranslator:
    def __init__(self, model_path, labels_path):
        """Initialize the translator"""
        # Load model
        self.model = keras.models.load_model(model_path)
        self.labels = np.load(labels_path)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=10)
        self.confidence_threshold = 0.7
        
        # Gesture stability
        self.last_gesture = None
        self.gesture_stable_count = 0
        self.stability_threshold = 15
        
        # Text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
    
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks).reshape(1, -1)
    
    def predict_gesture(self, landmarks):
        """Predict gesture from landmarks"""
        prediction = self.model.predict(landmarks, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        if confidence > self.confidence_threshold:
            self.prediction_buffer.append((predicted_class, confidence))
        
        if len(self.prediction_buffer) > 0:
            classes = [pred[0] for pred in self.prediction_buffer]
            most_common = max(set(classes), key=classes.count)
            avg_confidence = np.mean([pred[1] for pred in self.prediction_buffer 
                                     if pred[0] == most_common])
            return self.labels[most_common], avg_confidence, prediction
        
        return None, 0.0, prediction
    
    def check_gesture_stability(self, gesture):
        """Check if gesture is stable enough to add to sentence"""
        if gesture == self.last_gesture:
            self.gesture_stable_count += 1
        else:
            self.gesture_stable_count = 0
            self.last_gesture = gesture
        
        if self.gesture_stable_count == self.stability_threshold:
            self.gesture_stable_count = 0
            return True
        return False
    
    def speak_text(self, text):
        """Speak the given text using text-to-speech"""
        def speak():
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        
        # Run in separate thread to avoid blocking
        thread = threading.Thread(target=speak)
        thread.start()
    
    def process_frame(self, frame):
        """Process a single frame and return results"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = None
        confidence = 0.0
        all_predictions = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # Predict
                landmarks = self.extract_landmarks(hand_landmarks)
                gesture, confidence, all_predictions = self.predict_gesture(landmarks)
        else:
            self.gesture_stable_count = 0
            self.last_gesture = None
            self.prediction_buffer.clear()
        
        return frame, gesture, confidence, all_predictions

def load_model():
    """Load the trained model"""
    model_dir = "models"
    
    if not os.path.exists(model_dir):
        return None, None, "Models directory not found!"
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    
    if not model_files:
        return None, None, "No trained models found!"
    
    # Use the most recent model
    model_files.sort(reverse=True)
    model_file = os.path.join(model_dir, model_files[0])
    labels_file = model_file.replace('.h5', '_labels.npy')
    
    if not os.path.exists(labels_file):
        return None, None, f"Labels file not found: {labels_file}"
    
    try:
        translator = SignLanguageTranslator(model_file, labels_file)
        return translator, model_files[0], None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 Sign Language Translator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Load model
        if not st.session_state.model_loaded:
            with st.spinner("Loading model..."):
                translator, model_name, error = load_model()
                
                if error:
                    st.error(error)
                    st.stop()
                else:
                    st.session_state.translator = translator
                    st.session_state.model_loaded = True
                    st.success(f"✅ Model loaded: {model_name}")
        
        st.divider()
        
        # Camera settings
        st.subheader("📹 Camera Settings")
        camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0)
        
        st.divider()
        
        # Model info
        st.subheader("📊 Model Info")
        if st.session_state.model_loaded:
            st.write(f"**Gestures:** {len(st.session_state.translator.labels)}")
            with st.expander("View all gestures"):
                for i, label in enumerate(st.session_state.translator.labels):
                    st.write(f"{i+1}. {label}")
        
        st.divider()
        
        # Instructions
        st.subheader("📖 Instructions")
        st.markdown("""
        1. Click **Start Camera** to begin
        2. Show your gesture to the camera
        3. Wait for stable recognition
        4. Gesture auto-adds to sentence
        5. Click **Speak** to hear the sentence
        6. Use **Clear** to reset
        """)
    
    # Main content area - 2 columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        # Video placeholder
        video_placeholder = st.empty()
        
        # Control buttons
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            if st.button("🎥 Start Camera" if not st.session_state.running else "⏹️ Stop Camera", 
                        type="primary", use_container_width=True):
                st.session_state.running = not st.session_state.running
        
        with button_col2:
            if st.button("➕ Add to Sentence", use_container_width=True):
                if st.session_state.current_gesture:
                    st.session_state.sentence.append(st.session_state.current_gesture)
                    st.rerun()
        
        with button_col3:
            if st.button("🗑️ Clear Sentence", use_container_width=True):
                st.session_state.sentence = []
                st.rerun()
    
    with col2:
        st.subheader("🎯 Current Gesture")
        gesture_display = st.empty()
        confidence_display = st.empty()
        
        st.divider()
        
        st.subheader("💬 Sentence")
        sentence_display = st.empty()
        
        # Speak button
        if st.button("🔊 Speak Sentence", use_container_width=True, type="secondary"):
            if len(st.session_state.sentence) > 0:
                sentence_text = " ".join(st.session_state.sentence)
                st.session_state.translator.speak_text(sentence_text)
                st.success(f"🔊 Speaking: {sentence_text}")
            else:
                st.warning("Sentence is empty!")
        
        st.divider()
        
        # Statistics
        st.subheader("📈 Statistics")
        st.metric("Words in Sentence", len(st.session_state.sentence))
    
    # Camera processing
    if st.session_state.running:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera!")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, gesture, confidence, all_predictions = \
                st.session_state.translator.process_frame(frame)
            
            # Update session state
            st.session_state.current_gesture = gesture
            st.session_state.confidence = confidence
            
            # Check for auto-add
            if gesture and st.session_state.translator.check_gesture_stability(gesture):
                st.session_state.sentence.append(gesture)
            
            # Display frame
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
            
            # Update displays
            if gesture:
                gesture_display.markdown(
                    f'<div class="gesture-display">✋ {gesture}</div>',
                    unsafe_allow_html=True
                )
                confidence_display.progress(float(confidence), text=f"Confidence: {confidence:.1%}")
            else:
                gesture_display.markdown(
                    '<div class="gesture-display">👋 No hand detected</div>',
                    unsafe_allow_html=True
                )
                confidence_display.empty()
            
            # Update sentence display
            if st.session_state.sentence:
                sentence_text = " ".join(st.session_state.sentence)
                sentence_display.markdown(
                    f'<div class="sentence-display">{sentence_text}</div>',
                    unsafe_allow_html=True
                )
            else:
                sentence_display.markdown(
                    '<div class="sentence-display"><i>Sentence will appear here...</i></div>',
                    unsafe_allow_html=True
                )
        
        cap.release()
    else:
        # Display placeholder when camera is off
        video_placeholder.info("📷 Click 'Start Camera' to begin translation")
        
        # Update displays when camera is off
        if st.session_state.current_gesture:
            gesture_display.markdown(
                f'<div class="gesture-display">✋ {st.session_state.current_gesture}</div>',
                unsafe_allow_html=True
            )
        else:
            gesture_display.markdown(
                '<div class="gesture-display">👋 Camera is off</div>',
                unsafe_allow_html=True
            )
        
        if st.session_state.sentence:
            sentence_text = " ".join(st.session_state.sentence)
            sentence_display.markdown(
                f'<div class="sentence-display">{sentence_text}</div>',
                unsafe_allow_html=True
            )
        else:
            sentence_display.markdown(
                '<div class="sentence-display"><i>Sentence will appear here...</i></div>',
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()