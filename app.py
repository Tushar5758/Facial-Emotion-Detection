from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import base64
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
import json
import logging
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SESSION_FOLDER'] = 'sessions'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create required directories
os.makedirs(app.config['SESSION_FOLDER'], exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to track DeepFace availability
DEEPFACE_AVAILABLE = False

# Try to import DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("✅ DeepFace imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ DeepFace not available: {e}")
    logger.warning("Install with: pip install deepface")
except Exception as e:
    logger.warning(f"⚠️ DeepFace import error: {e}")

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class EmotionAnalyzer:
    def __init__(self):
        self.supported_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
    def decode_base64_image(self, base64_string):
        """Convert base64 string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
                
            return img
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None
    
    def analyze_emotion_with_deepface(self, image):
        """Analyze emotion using DeepFace"""
        try:
            if not DEEPFACE_AVAILABLE:
                raise ImportError("DeepFace not available")
                
            # Use DeepFace to analyze emotions
            result = DeepFace.analyze(
                img_path=image,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            # Extract emotion scores
            if isinstance(result, list):
                emotions = result[0]['emotion']
            else:
                emotions = result['emotion']
            
            # Normalize emotions and convert numpy types
            normalized_emotions = {}
            for emotion in self.supported_emotions:
                if emotion in emotions:
                    # Convert numpy float32 to Python float
                    normalized_emotions[emotion] = round(float(emotions[emotion]), 2)
                else:
                    normalized_emotions[emotion] = 0.0
            
            # Ensure all values are JSON serializable
            normalized_emotions = convert_numpy_types(normalized_emotions)
            
            return {
                'success': True,
                'emotions': normalized_emotions,
                'dominant_emotion': max(normalized_emotions, key=normalized_emotions.get)
            }
            
        except Exception as e:
            logger.error(f"DeepFace analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'emotions': {emotion: 0.0 for emotion in self.supported_emotions}
            }
    
    def analyze_emotion_mock(self, image):
        """Mock emotion analysis for demo purposes"""
        try:
            # Get image properties for semi-realistic mock results
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))  # Convert to Python float
            
            # Generate mock emotions based on image properties
            base_scores = {
                'happy': max(0, min(100, brightness / 2.55 + np.random.normal(0, 10))),
                'neutral': max(0, min(100, 50 + np.random.normal(0, 15))),
                'sad': max(0, min(100, (255 - brightness) / 3 + np.random.normal(0, 8))),
                'angry': max(0, min(100, 15 + np.random.normal(0, 12))),
                'surprise': max(0, min(100, 20 + np.random.normal(0, 10))),
                'fear': max(0, min(100, 10 + np.random.normal(0, 8))),
                'disgust': max(0, min(100, 8 + np.random.normal(0, 6)))
            }
            
            # Normalize to sum to 100 and convert to Python floats
            total = sum(base_scores.values())
            mock_emotions = {}
            for emotion in base_scores:
                # Ensure conversion to Python float
                mock_emotions[emotion] = round(float(base_scores[emotion] / total * 100), 2)
            
            # Convert any remaining numpy types
            mock_emotions = convert_numpy_types(mock_emotions)
            
            return {
                'success': True,
                'emotions': mock_emotions,
                'dominant_emotion': max(mock_emotions, key=mock_emotions.get)
            }
            
        except Exception as e:
            logger.error(f"Mock analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'emotions': {emotion: 0.0 for emotion in self.supported_emotions}
            }
    
    def analyze_emotion(self, image):
        """Main emotion analysis method"""
        if DEEPFACE_AVAILABLE:
            return self.analyze_emotion_with_deepface(image)
        else:
            logger.info("Using mock analysis (DeepFace not available)")
            return self.analyze_emotion_mock(image)

# Initialize analyzer
analyzer = EmotionAnalyzer()

# Routes
@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Emotion Detection API is running',
        'deepface_available': DEEPFACE_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/create-session', methods=['POST'])
def create_session():
    """Create a new session"""
    try:
        session_id = str(uuid.uuid4())
        session_folder = os.path.join(app.config['SESSION_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        session_data = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'frames_count': 0,
            'status': 'created',
            'deepface_available': DEEPFACE_AVAILABLE
        }
        
        with open(os.path.join(session_folder, 'session.json'), 'w') as f:
            json.dump(session_data, f)
        
        logger.info(f"Created session: {session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Session created successfully'
        })
        
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload-frames', methods=['POST'])
def upload_frames():
    """Upload and save captured frames"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
            
        session_id = data.get('session_id')
        frames = data.get('frames', [])
        
        if not session_id or not frames:
            return jsonify({'success': False, 'error': 'Missing session_id or frames'}), 400
        
        session_folder = os.path.join(app.config['SESSION_FOLDER'], session_id)
        if not os.path.exists(session_folder):
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        saved_frames = []
        
        for i, frame_data in enumerate(frames):
            try:
                image = analyzer.decode_base64_image(frame_data['imageData'])
                if image is not None:
                    timestamp_clean = frame_data['timestamp'].replace(':', '-').replace('.', '-')
                    filename = f"frame_{i+1:02d}_{timestamp_clean}.jpg"
                    filepath = os.path.join(session_folder, filename)
                    
                    if cv2.imwrite(filepath, image):
                        saved_frames.append({
                            'frame_id': i + 1,
                            'filename': filename,
                            'filepath': filepath,
                            'timestamp': frame_data['timestamp']
                        })
                        logger.info(f"Saved frame {i+1}: {filename}")
                    
            except Exception as e:
                logger.error(f"Error saving frame {i+1}: {e}")
                continue
        
        # Update session metadata
        session_file = os.path.join(session_folder, 'session.json')
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        session_data.update({
            'frames_count': len(saved_frames),
            'status': 'frames_uploaded',
            'frames': saved_frames,
            'uploaded_at': datetime.now().isoformat()
        })
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Successfully saved {len(saved_frames)} frames',
            'frames_saved': len(saved_frames)
        })
        
    except Exception as e:
        logger.error(f"Frame upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-emotions', methods=['POST'])
def analyze_emotions():
    """Analyze emotions for all frames in a session"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
            
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Missing session_id'}), 400
        
        session_folder = os.path.join(app.config['SESSION_FOLDER'], session_id)
        session_file = os.path.join(session_folder, 'session.json')
        
        if not os.path.exists(session_file):
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        # Load session data
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        frames = session_data.get('frames', [])
        if not frames:
            return jsonify({'success': False, 'error': 'No frames found'}), 400
        
        # Analyze each frame
        analysis_results = []
        
        logger.info(f"Starting emotion analysis for {len(frames)} frames")
        
        for frame_info in frames:
            filepath = frame_info['filepath']
            
            if os.path.exists(filepath):
                try:
                    image = cv2.imread(filepath)
                    if image is None:
                        raise ValueError("Could not load image")
                    
                    emotion_result = analyzer.analyze_emotion(image)
                    
                    # Ensure all data is JSON serializable
                    frame_result = {
                        'frame': frame_info['frame_id'],
                        'timestamp': frame_info['timestamp'],
                        'filename': frame_info['filename'],
                        'emotions': convert_numpy_types(emotion_result['emotions']),
                        'dominant_emotion': emotion_result.get('dominant_emotion', 'neutral'),
                        'success': emotion_result['success']
                    }
                    
                    if not emotion_result['success']:
                        frame_result['error'] = emotion_result.get('error', 'Unknown error')
                    
                    analysis_results.append(frame_result)
                    logger.info(f"Analyzed frame {frame_info['frame_id']}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing frame {frame_info['frame_id']}: {e}")
                    analysis_results.append({
                        'frame': frame_info['frame_id'],
                        'timestamp': frame_info['timestamp'],
                        'filename': frame_info['filename'],
                        'emotions': {emotion: 0.0 for emotion in analyzer.supported_emotions},
                        'dominant_emotion': 'neutral',
                        'success': False,
                        'error': str(e)
                    })
        
        # Calculate average emotions
        successful_analyses = [r for r in analysis_results if r['success']]
        
        if successful_analyses:
            avg_emotions = {}
            for emotion in analyzer.supported_emotions:
                total = sum(float(frame['emotions'][emotion]) for frame in successful_analyses)
                avg_emotions[emotion] = round(total / len(successful_analyses), 2)
            
            dominant_emotion = max(avg_emotions, key=avg_emotions.get)
        else:
            avg_emotions = {emotion: 0.0 for emotion in analyzer.supported_emotions}
            dominant_emotion = 'neutral'
        
        # Ensure all data is JSON serializable
        avg_emotions = convert_numpy_types(avg_emotions)
        analysis_results = convert_numpy_types(analysis_results)
        
        # Update session data
        session_data.update({
            'status': 'analyzed',
            'analysis_results': analysis_results,
            'average_emotions': avg_emotions,
            'dominant_emotion': dominant_emotion,
            'analyzed_at': datetime.now().isoformat()
        })
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'results': analysis_results,
            'average_emotions': avg_emotions,
            'dominant_emotion': dominant_emotion,
            'total_frames': len(frames),
            'successful_analyses': len(successful_analyses),
            'deepface_used': DEEPFACE_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"Emotion analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def calculate_mind_age(emotions, dominant_emotion):
    """Calculate psychological/emotional age based on emotion patterns"""
    try:
        # Base mind age calculation using emotion distribution
        emotion_weights = {
            'happy': 0.15,      # Moderate happiness shows emotional balance
            'neutral': 0.20,    # High neutrality indicates emotional maturity
            'sad': -0.05,       # Sadness can indicate introspection (mature) but also vulnerability
            'angry': -0.15,     # Anger often indicates lower emotional regulation
            'fear': -0.10,      # Fear can show caution (mature) or anxiety (immature)
            'surprise': 0.05,   # Openness to new experiences
            'disgust': -0.08    # Strong disgust reactions can indicate rigidity
        }
        
        # Calculate weighted emotional maturity score
        maturity_score = 0
        for emotion, percentage in emotions.items():
            if emotion in emotion_weights:
                maturity_score += (percentage / 100) * emotion_weights[emotion]
        
        # Base age ranges for different emotional patterns
        age_mapping = {
            'happy': {'base': 25, 'range': (20, 35), 'description': 'Optimistic Young Adult'},
            'neutral': {'base': 35, 'range': (30, 45), 'description': 'Mature Adult'},
            'sad': {'base': 28, 'range': (18, 40), 'description': 'Reflective Individual'},
            'angry': {'base': 22, 'range': (16, 30), 'description': 'Reactive Young Adult'},
            'fear': {'base': 26, 'range': (20, 35), 'description': 'Cautious Individual'},
            'surprise': {'base': 23, 'range': (18, 32), 'description': 'Curious Young Adult'},
            'disgust': {'base': 30, 'range': (25, 40), 'description': 'Critical Adult'}
        }
        
        # Get base mind age from dominant emotion
        dominant_info = age_mapping.get(dominant_emotion, age_mapping['neutral'])
        base_age = dominant_info['base']
        
        # Adjust based on maturity score (-0.3 to +0.3 range typically)
        age_adjustment = maturity_score * 30  # Scale to reasonable age range
        calculated_age = max(16, min(50, base_age + age_adjustment))
        
        # Determine emotional intelligence level
        if maturity_score > 0.1:
            ei_level = "High"
            ei_description = "Shows strong emotional regulation and balance"
        elif maturity_score > -0.05:
            ei_level = "Moderate"
            ei_description = "Demonstrates average emotional awareness"
        else:
            ei_level = "Developing"
            ei_description = "Has room for growth in emotional regulation"
        
        return {
            'mind_age': round(calculated_age),
            'age_range': dominant_info['range'],
            'personality_type': dominant_info['description'],
            'emotional_intelligence': ei_level,
            'ei_description': ei_description,
            'maturity_score': round(maturity_score, 3)
        }
        
    except Exception as e:
        logger.error(f"Mind age calculation error: {e}")
        return {
            'mind_age': 25,
            'age_range': (20, 35),
            'personality_type': 'Balanced Individual',
            'emotional_intelligence': 'Moderate',
            'ei_description': 'Shows typical emotional patterns',
            'maturity_score': 0.0
        }

@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    """Get personalized recommendations based on dominant emotion"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
            
        dominant_emotion = data.get('dominant_emotion', 'neutral')
        emotions = data.get('emotions', {})
        
        # Calculate mind age based on emotions
        mind_age_data = calculate_mind_age(emotions, dominant_emotion)
        
        recommendations = {
            'happy': [
                '🎉 Great mood detected! Share your positivity with others.',
                '📝 Consider journaling about what made you happy today.',
                '🏃‍♂️ Channel this energy into a fun physical activity.',
                '🎵 Listen to upbeat music to maintain your good mood.',
                '🤝 This is a perfect time to connect with friends and family.',
                f'🧠 Your optimistic energy reflects a {mind_age_data["personality_type"].lower()} mindset!'
            ],
            'sad': [
                '🤗 It\'s okay to feel sad. Practice self-compassion.',
                '☎️ Consider reaching out to a trusted friend or family member.',
                '🧘‍♀️ Try mindfulness meditation or deep breathing exercises.',
                '🌿 Spend some time in nature to lift your spirits.',
                '📖 Reading uplifting books or watching feel-good content might help.',
                f'💭 Your reflective state shows depth - use this time for introspection and growth.'
            ],
            'angry': [
                '😤 Take deep breaths and count to ten before reacting.',
                '🏃‍♂️ Try physical exercise to release tension and anger.',
                '📝 Write down your feelings to process them better.',
                '🎧 Listen to calming music or nature sounds.',
                '🧘‍♀️ Practice progressive muscle relaxation techniques.',
                f'⚡ Channel your intense energy constructively - consider anger management techniques.'
            ],
            'surprise': [
                '✨ Embrace the unexpected! New experiences can be enriching.',
                '📚 Use this energy to learn something new and interesting.',
                '🤔 Reflect on what surprised you and why it affected you.',
                '📱 Share interesting discoveries with others.',
                '🎯 Channel this alertness into creative problem-solving.',
                f'🌟 Your openness to surprise shows a curious, adaptable mind!'
            ],
            'fear': [
                '💪 Remember that courage is acting despite fear.',
                '🧘‍♀️ Practice grounding techniques to feel more centered.',
                '👥 Talk to someone you trust about your concerns.',
                '📖 Learn more about what you fear to reduce anxiety.',
                '🌱 Take small steps toward overcoming your fears.',
                f'🛡️ Your caution can be wisdom - balance it with confidence-building activities.'
            ],
            'disgust': [
                '🧘‍♀️ Take a moment to understand the source of this feeling.',
                '🌱 Focus on things that bring you peace and comfort.',
                '🏠 Create a clean, organized environment around you.',
                '😊 Practice gratitude for positive aspects of your day.',
                '🔄 Consider if there are changes you can make to improve the situation.',
                f'🎭 Your strong reactions show you have clear values - use them constructively.'
            ],
            'neutral': [
                '⚖️ Emotional balance is actually quite healthy!',
                '🎯 This is a good time for focused work or planning.',
                '🌟 Consider trying something new to spark interest.',
                '📝 Reflect on your goals and priorities.',
                '🧘‍♀️ Use this calm state for meditation or mindfulness practice.',
                f'🏛️ Your emotional stability reflects mature self-regulation skills!'
            ]
        }
        
        emotion_recommendations = recommendations.get(dominant_emotion, recommendations['neutral'])
        
        return jsonify({
            'success': True,
            'dominant_emotion': dominant_emotion,
            'recommendations': emotion_recommendations,
            'mind_age_analysis': {
                'estimated_mind_age': mind_age_data['mind_age'],
                'age_range': f"{mind_age_data['age_range'][0]}-{mind_age_data['age_range'][1]} years",
                'personality_type': mind_age_data['personality_type'],
                'emotional_intelligence': mind_age_data['emotional_intelligence'],
                'ei_description': mind_age_data['ei_description'],
                'interpretation': f"Based on your emotional patterns, your psychological age appears to be around {mind_age_data['mind_age']} years, suggesting a {mind_age_data['personality_type'].lower()} emotional profile."
            },
            'general_tip': 'Remember that emotions are temporary and provide valuable information about your experiences. Your emotional patterns reveal insights about your psychological maturity and self-awareness.'
        })
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({'success': False, 'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Facial Emotion Detection System...")
    print("📋 Available endpoints:")
    print("   GET  / - Main application")
    print("   GET  /api/health - Health check")
    print("   POST /api/create-session - Create new session")
    print("   POST /api/upload-frames - Upload captured frames")
    print("   POST /api/analyze-emotions - Analyze emotions")
    print("   POST /api/get-recommendations - Get recommendations")
    print(f"\n🔧 DeepFace Status: {'✅ Available' if DEEPFACE_AVAILABLE else '❌ Not Available (using mock analysis)'}")
    print("\n🌐 Open your browser and go to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)