// Global variables
let cameraActive = false;
let updateInterval = null;
let currentGesture = null;

// Text-to-Speech
const synth = window.speechSynthesis;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Sign Language Translator loaded');
    updateCameraStatus(false);
});

// Start Camera
async function startCamera() {
    try {
        const response = await fetch('/start_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            cameraActive = true;
            updateCameraStatus(true);
            
            // Show video feed
            document.getElementById('videoPlaceholder').style.display = 'none';
            document.getElementById('videoFeed').style.display = 'block';
            document.getElementById('videoFeed').src = '/video_feed?' + new Date().getTime();
            
            // Switch buttons
            document.getElementById('startBtn').style.display = 'none';
            document.getElementById('stopBtn').style.display = 'flex';
            
            // Start updating predictions
            startPredictionUpdates();
            
            showNotification('Camera started successfully!', 'success');
        }
    } catch (error) {
        console.error('Error starting camera:', error);
        showNotification('Failed to start camera', 'error');
    }
}

// Stop Camera
async function stopCamera() {
    try {
        const response = await fetch('/stop_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            cameraActive = false;
            updateCameraStatus(false);
            
            // Hide video feed
            document.getElementById('videoFeed').style.display = 'none';
            document.getElementById('videoPlaceholder').style.display = 'flex';
            
            // Switch buttons
            document.getElementById('stopBtn').style.display = 'none';
            document.getElementById('startBtn').style.display = 'flex';
            
            // Stop updating predictions
            stopPredictionUpdates();
            
            // Reset displays
            resetDisplays();
            
            showNotification('Camera stopped', 'info');
        }
    } catch (error) {
        console.error('Error stopping camera:', error);
        showNotification('Failed to stop camera', 'error');
    }
}

// Update camera status indicator
function updateCameraStatus(active) {
    const statusIndicator = document.getElementById('cameraStatus');
    const statusText = statusIndicator.querySelector('.status-text');
    
    if (active) {
        statusIndicator.classList.add('active');
        statusText.textContent = 'Camera Active';
    } else {
        statusIndicator.classList.remove('active');
        statusText.textContent = 'Camera Off';
    }
}

// Start prediction updates
function startPredictionUpdates() {
    updateInterval = setInterval(updatePredictions, 100); // Update every 100ms
}

// Stop prediction updates
function stopPredictionUpdates() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

// Update predictions from server
async function updatePredictions() {
    try {
        const response = await fetch('/get_prediction');
        const data = await response.json();
        
        // Update current gesture
        if (data.gesture && data.hand_detected) {
            currentGesture = data.gesture;
            updateGestureDisplay(data.gesture, data.confidence);
            updatePredictionsList(data.all_predictions);
        } else {
            currentGesture = null;
            updateGestureDisplay(null, 0);
        }
        
        // Update sentence
        updateSentenceDisplay(data.sentence);
        
    } catch (error) {
        console.error('Error updating predictions:', error);
    }
}

// Update gesture display
function updateGestureDisplay(gesture, confidence) {
    const gestureIcon = document.querySelector('.gesture-icon');
    const gestureName = document.getElementById('gestureName');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    
    if (gesture) {
        // Map gestures to emojis
        const gestureEmojis = {
            'Hello': '👋',
            'ThankYou': '🙏',
            'Yes': '👍',
            'No': '👎',
            'Help': '🆘',
            'HELP': '🆘',
            'OK': '👌',
            'PEACE': '✌️',
            'I LOVE YOU': '🤟',
            'Goodbye': '👋',
            'Please': '🙏'
        };
        
        gestureIcon.textContent = gestureEmojis[gesture] || '✋';
        gestureName.textContent = gesture;
        
        const confidencePercent = Math.round(confidence * 100);
        confidenceFill.style.width = confidencePercent + '%';
        confidenceText.textContent = confidencePercent + '%';
    } else {
        gestureIcon.textContent = '👋';
        gestureName.textContent = 'No hand detected';
        confidenceFill.style.width = '0%';
        confidenceText.textContent = '0%';
    }
}

// Update predictions list
function updatePredictionsList(predictions) {
    const predictionsList = document.getElementById('predictionsList');
    
    if (!predictions || predictions.length === 0) {
        predictionsList.innerHTML = '<p class="no-predictions">No predictions available</p>';
        return;
    }
    
    // Sort by confidence
    predictions.sort((a, b) => b.confidence - a.confidence);
    
    // Show top 5
    const topPredictions = predictions.slice(0, 5);
    
    let html = '';
    topPredictions.forEach(pred => {
        const confidencePercent = Math.round(pred.confidence * 100);
        html += `
            <div class="prediction-item">
                <span class="prediction-name">${pred.label}</span>
                <span class="prediction-confidence">${confidencePercent}%</span>
            </div>
        `;
    });
    
    predictionsList.innerHTML = html;
}

// Update sentence display
function updateSentenceDisplay(sentence) {
    const sentenceText = document.getElementById('sentenceText');
    const wordCount = document.getElementById('wordCount');
    
    if (sentence && sentence.length > 0) {
        sentenceText.textContent = sentence.join(' ');
        sentenceText.classList.remove('empty');
        wordCount.textContent = sentence.length + ' word' + (sentence.length !== 1 ? 's' : '');
    } else {
        sentenceText.textContent = 'Your sentence will appear here...';
        sentenceText.classList.add('empty');
        wordCount.textContent = '0 words';
    }
}

// Add current gesture to sentence
async function addToSentence() {
    if (!currentGesture) {
        showNotification('No gesture detected', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/add_to_sentence', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ gesture: currentGesture })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            updateSentenceDisplay(data.sentence);
            showNotification(`Added "${currentGesture}" to sentence`, 'success');
        }
    } catch (error) {
        console.error('Error adding to sentence:', error);
        showNotification('Failed to add gesture', 'error');
    }
}

// Clear sentence
async function clearSentence() {
    try {
        const response = await fetch('/clear_sentence', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            updateSentenceDisplay([]);
            showNotification('Sentence cleared', 'info');
        }
    } catch (error) {
        console.error('Error clearing sentence:', error);
        showNotification('Failed to clear sentence', 'error');
    }
}

// Speak sentence using Web Speech API
async function speakSentence() {
    try {
        const response = await fetch('/get_sentence');
        const data = await response.json();
        
        if (!data.sentence || data.sentence.length === 0) {
            showNotification('Sentence is empty', 'warning');
            return;
        }
        
        const text = data.sentence.join(' ');
        
        // Cancel any ongoing speech
        synth.cancel();
        
        // Create utterance
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        // Event handlers
        utterance.onstart = function() {
            showNotification('🔊 Speaking: ' + text, 'info');
        };
        
        utterance.onend = function() {
            console.log('Speech finished');
        };
        
        utterance.onerror = function(event) {
            console.error('Speech error:', event);
            showNotification('Speech error occurred', 'error');
        };
        
        // Speak
        synth.speak(utterance);
        
    } catch (error) {
        console.error('Error speaking sentence:', error);
        showNotification('Failed to speak sentence', 'error');
    }
}

// Reset displays
function resetDisplays() {
    updateGestureDisplay(null, 0);
    document.getElementById('predictionsList').innerHTML = '<p class="no-predictions">Start camera to see predictions</p>';
}

// Show notification (simple toast)
function showNotification(message, type) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Style
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.padding = '15px 25px';
    notification.style.borderRadius = '10px';
    notification.style.color = 'white';
    notification.style.fontWeight = '600';
    notification.style.zIndex = '10000';
    notification.style.boxShadow = '0 4px 15px rgba(0,0,0,0.2)';
    notification.style.animation = 'slideIn 0.3s ease-out';
    
    // Colors based on type
    const colors = {
        'success': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        'error': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        'warning': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
        'info': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    };
    
    notification.style.background = colors[type] || colors['info'];
    
    // Add to body
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add CSS for notification animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Space - Start/Stop camera
    if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        if (cameraActive) {
            stopCamera();
        } else {
            startCamera();
        }
    }
    
    // A - Add to sentence
    if (e.code === 'KeyA' && e.ctrlKey) {
        e.preventDefault();
        addToSentence();
    }
    
    // C - Clear sentence
    if (e.code === 'KeyC' && e.ctrlKey && e.shiftKey) {
        e.preventDefault();
        clearSentence();
    }
    
    // S - Speak
    if (e.code === 'KeyS' && e.ctrlKey) {
        e.preventDefault();
        speakSentence();
    }
});

console.log('Keyboard shortcuts enabled:');
console.log('Space - Start/Stop camera');
console.log('Ctrl+A - Add to sentence');
console.log('Ctrl+Shift+C - Clear sentence');
console.log('Ctrl+S - Speak sentence');
