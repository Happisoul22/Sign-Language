# 🤟 Sign Language Translator - Streamlit Web App

## Overview
A real-time sign language translation web application with interactive buttons and text-to-speech functionality.

## Features
✅ **Interactive Web Interface** - Beautiful, user-friendly Streamlit UI
✅ **Start/Stop Camera** - Control camera with buttons
✅ **Real-time Recognition** - Instant gesture detection
✅ **Sentence Building** - Auto-build sentences from gestures
✅ **Text-to-Speech** - Speak out recognized sentences
✅ **Live Confidence Display** - See prediction confidence in real-time
✅ **Gesture Statistics** - Track your gestures

## Installation

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

### 2. Make Sure You Have:
- A trained model in the `models/` folder (from training step)
- Python 3.8 or higher
- A working webcam

## How to Run

### Start the Streamlit App:

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser at: `http://localhost:8501`

## Using the Application

### 1️⃣ **Starting the App**
- The app loads your trained model automatically
- Check the sidebar to see loaded gestures

### 2️⃣ **Start Camera**
- Click the **"🎥 Start Camera"** button
- Your webcam will activate
- Show your hand gesture to the camera

### 3️⃣ **Recognition**
- The app detects your hand in real-time
- Current gesture appears in the "Current Gesture" panel
- Confidence level is shown with a progress bar
- Gestures auto-add to sentence when held steady

### 4️⃣ **Building Sentences**
- Hold a gesture steady for ~1 second to auto-add
- OR click **"➕ Add to Sentence"** to manually add
- Your sentence builds in the "Sentence" panel

### 5️⃣ **Text-to-Speech**
- Click **"🔊 Speak Sentence"** button
- The app will speak your sentence out loud!

### 6️⃣ **Controls**
- **Start/Stop Camera** - Toggle camera on/off
- **Add to Sentence** - Manually add current gesture
- **Clear Sentence** - Reset the sentence
- **Speak Sentence** - Text-to-speech output

## Interface Layout

### Left Side (Camera Feed)
- Live video with hand landmarks
- Control buttons

### Right Side (Information)
- Current gesture display
- Confidence meter
- Sentence builder
- Statistics

### Sidebar
- Model information
- Loaded gestures list
- Camera settings
- Instructions

## Troubleshooting

### Camera Not Working?
- Check camera permissions
- Try changing "Camera Index" in sidebar (0, 1, 2...)
- Make sure no other app is using the camera

### Model Not Loading?
- Ensure you have a trained model in `models/` folder
- Check that both `.h5` and `_labels.npy` files exist
- Run `train_model.py` first if no model exists

### Text-to-Speech Not Working?
- Windows: Should work by default
- Mac: Requires `espeak` - install with: `brew install espeak`
- Linux: Install `espeak` - `sudo apt-get install espeak`

## Keyboard Shortcuts (in browser)
- Press `R` to refresh the app
- Press `Ctrl+C` in terminal to stop the server

## Tips for Best Performance
1. **Good Lighting** - Ensure your hand is well-lit
2. **Plain Background** - Use a simple background for better detection
3. **Steady Gestures** - Hold gestures for 1-2 seconds
4. **Camera Distance** - Keep hand 1-2 feet from camera
5. **Single Hand** - Show one hand at a time for best results

## Project Structure
```
sign_language_translator/
├── streamlit_app.py           # Main Streamlit application
├── requirements.txt           # Python dependencies
├── models/                    # Trained models folder
│   ├── sign_language_model_*.h5
│   └── sign_language_model_*_labels.npy
├── sign_language_data/        # Training data
│   ├── hand_landmarks_*.npy
│   └── labels_*.npy
├── collect_data.py           # Data collection script
├── train_model.py            # Model training script
├── manage_dataset.py         # Dataset management
└── test_model.py             # Testing script
```

## Advanced Features

### Auto-Save Sentences
The app keeps your sentence in memory until you clear it.

### Confidence Threshold
Only predictions above 70% confidence are shown (adjustable in code).

### Prediction Smoothing
Uses last 10 predictions for stable results.

## Customization

### Change Text-to-Speech Voice
Edit in `streamlit_app.py`:
```python
self.tts_engine.setProperty('rate', 150)  # Speed
self.tts_engine.setProperty('volume', 0.9)  # Volume
```

### Adjust Camera Resolution
Edit in `streamlit_app.py`:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### Change Confidence Threshold
Edit in `streamlit_app.py`:
```python
self.confidence_threshold = 0.7  # 70%
```

## Next Steps
1. Add more gestures to your dataset
2. Train with more samples for better accuracy
3. Deploy the app online (Streamlit Cloud, Heroku, etc.)
4. Add gesture recording for training directly from the app

## Credits
Built with:
- Streamlit - Web framework
- MediaPipe - Hand detection
- TensorFlow - Machine learning
- pyttsx3 - Text-to-speech
- OpenCV - Computer vision

## License
Free to use for educational and personal projects!

---

Enjoy your Sign Language Translator! 🤟✨
