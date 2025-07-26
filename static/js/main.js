class EmotionDetectionSystem {
    constructor() {
        this.initializeElements();
        this.initializeVariables();
        this.initializeEventListeners();
        this.updateSessionInfo();
        this.checkBackendConnection();
    }

    initializeElements() {
        this.videoElement = document.getElementById('videoElement');
        this.cameraPlaceholder = document.getElementById('cameraPlaceholder');
        this.startBtn = document.getElementById('startBtn');
        this.captureBtn = document.getElementById('captureBtn');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.retakeBtn = document.getElementById('retakeBtn');
        this.systemStatus = document.getElementById('systemStatus');
        this.captureStatus = document.getElementById('captureStatus');
        this.progressBar = document.getElementById('progressBar');
        this.progressFill = document.getElementById('progressFill');
        this.framesThumbnails = document.getElementById('framesThumbnails');
        this.resultsSection = document.getElementById('resultsSection');
        this.emotionResults = document.getElementById('emotionResults');
        this.recommendationsSection = document.getElementById('recommendationsSection');
        this.recommendationsList = document.getElementById('recommendationsList');
        this.sessionInfo = document.getElementById('sessionInfo');
        this.sessionId = document.getElementById('sessionId');
        this.frameCount = document.getElementById('frameCount');
        this.sessionStatus = document.getElementById('sessionStatus');
    }

    initializeVariables() {
        this.stream = null;
        this.capturedFrames = [];
        this.sessionID = this.generateSessionID();
        this.captureInterval = null;
        this.currentFrame = 0;
        this.apiUrl = '';  // Empty string means same domain
        this.backendConnected = false;
        this.currentAnalysisData = null; // Store current analysis for recommendations
    }

    generateSessionID() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.captureBtn.addEventListener('click', () => this.beginCapture());
        this.analyzeBtn.addEventListener('click', () => this.analyzeEmotions());
        this.retakeBtn.addEventListener('click', () => this.retakeTest());
    }

    updateSessionInfo() {
        this.sessionId.textContent = this.sessionID;
        this.frameCount.textContent = this.capturedFrames.length;
        this.sessionInfo.style.display = 'block';
    }

    updateStatus(message, type = 'info') {
        this.systemStatus.textContent = message;
        this.systemStatus.className = `status ${type}`;
        this.sessionStatus.textContent = message;
    }

    async checkBackendConnection() {
        try {
            const response = await fetch('/api/health', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            if (data.status === 'healthy') {
                this.backendConnected = true;
                const deepfaceStatus = data.deepface_available ? 'with DeepFace' : 'with mock analysis';
                this.updateStatus(`✅ Connected to backend server (${deepfaceStatus})`, 'info');
            }
        } catch (error) {
            console.warn('Backend connection failed:', error);
            this.backendConnected = false;
            this.updateStatus('❌ Cannot connect to backend server', 'error');
        }
    }

    async startCamera() {
        try {
            this.updateStatus('Starting camera...', 'info');
            
            // Try multiple approaches for browser compatibility
            let getUserMedia = null;
            
            // Modern approach
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
            }
            // Legacy approach 1
            else if (navigator.webkitGetUserMedia) {
                getUserMedia = (constraints) => {
                    return new Promise((resolve, reject) => {
                        navigator.webkitGetUserMedia(constraints, resolve, reject);
                    });
                };
            }
            // Legacy approach 2
            else if (navigator.mozGetUserMedia) {
                getUserMedia = (constraints) => {
                    return new Promise((resolve, reject) => {
                        navigator.mozGetUserMedia(constraints, resolve, reject);
                    });
                };
            }
            // Legacy approach 3
            else if (navigator.getUserMedia) {
                getUserMedia = (constraints) => {
                    return new Promise((resolve, reject) => {
                        navigator.getUserMedia(constraints, resolve, reject);
                    });
                };
            }

            if (!getUserMedia) {
                throw new Error('Camera not supported');
            }
            
            // Start with simple constraints
            let constraints = { video: true };
            
            try {
                this.stream = await getUserMedia(constraints);
            } catch (simpleError) {
                // If simple fails, try more specific constraints
                constraints = {
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    }
                };
                this.stream = await getUserMedia(constraints);
            }
            
            this.videoElement.srcObject = this.stream;
            
            // Simple approach for video ready
            this.videoElement.onloadedmetadata = () => {
                this.videoElement.play().catch(e => {
                    console.log('Autoplay prevented, but video is ready');
                });
            };
            
            // Give video a moment to initialize
            setTimeout(() => {
                this.videoElement.style.display = 'block';
                this.cameraPlaceholder.style.display = 'none';
                
                this.startBtn.disabled = true;
                this.captureBtn.disabled = false;
                
                this.updateStatus('Camera started successfully! Ready to capture frames.', 'info');
            }, 500);
            
        } catch (error) {
            console.error('Camera error:', error);
            
            // Simple error message
            let errorMessage = 'Camera access failed. ';
            
            if (error.name === 'NotAllowedError') {
                errorMessage += 'Please allow camera permissions.';
            } else if (error.name === 'NotFoundError') {
                errorMessage += 'No camera found.';
            } else {
                errorMessage += 'Please check camera availability.';
            }
            
            this.updateStatus(errorMessage, 'error');
        }
    }

    beginCapture() {
        if (!this.backendConnected) {
            this.updateStatus('❌ Backend server not connected. Please refresh and try again.', 'error');
            return;
        }

        this.capturedFrames = [];
        this.framesThumbnails.innerHTML = '';
        this.currentFrame = 0;
        
        this.captureBtn.disabled = true;
        this.progressBar.style.display = 'block';
        this.captureStatus.style.display = 'block';
        this.resultsSection.style.display = 'none';
        
        this.updateStatus('Capturing frames...', 'info');
        this.captureStatus.textContent = `Capturing frame 1 of 10...`;
        
        // Capture first frame immediately
        this.captureFrame();
        
        // Then capture remaining 9 frames at 1-second intervals
        this.captureInterval = setInterval(() => {
            this.captureFrame();
        }, 1000);
    }

    captureFrame() {
        try {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = this.videoElement.videoWidth || 640;
            canvas.height = this.videoElement.videoHeight || 480;
            
            ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            const timestamp = new Date().toISOString();
            
            const frameData = {
                id: this.currentFrame + 1,
                imageData: imageData,
                timestamp: timestamp,
                sessionID: this.sessionID
            };
            
            this.capturedFrames.push(frameData);
            this.currentFrame++;
            
            // Create thumbnail
            const img = document.createElement('img');
            img.src = imageData;
            img.className = 'frame-thumbnail';
            img.title = `Frame ${this.currentFrame} - ${timestamp}`;
            this.framesThumbnails.appendChild(img);
            
            // Update progress
            const progress = (this.currentFrame / 10) * 100;
            this.progressFill.style.width = progress + '%';
            this.frameCount.textContent = this.currentFrame;
            
            if (this.currentFrame < 10) {
                this.captureStatus.textContent = `Capturing frame ${this.currentFrame + 1} of 10...`;
            } else {
                // All frames captured
                clearInterval(this.captureInterval);
                this.captureStatus.textContent = 'All frames captured successfully!';
                this.updateStatus('10 frames captured! Ready for analysis.', 'info');
                this.analyzeBtn.disabled = false;
                this.retakeBtn.disabled = false;
            }
            
            this.updateSessionInfo();
            
        } catch (error) {
            console.error('Error capturing frame:', error);
            this.updateStatus('Error capturing frame. Please try again.', 'error');
        }
    }

    async analyzeEmotions() {
        if (!this.backendConnected) {
            this.updateStatus('❌ Backend server not connected', 'error');
            return;
        }

        this.analyzeBtn.disabled = true;
        this.updateStatus('Creating session and uploading frames...', 'info');
        
        try {
            // Step 1: Create session
            const sessionResponse = await fetch('/api/create-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (!sessionResponse.ok) {
                throw new Error(`Session creation failed: ${sessionResponse.status}`);
            }
            
            const sessionData = await sessionResponse.json();
            if (!sessionData.success) {
                throw new Error(sessionData.error || 'Failed to create session');
            }
            
            const backendSessionId = sessionData.session_id;
            this.updateStatus('Session created. Uploading frames...', 'info');
            
            // Step 2: Upload frames
            const uploadResponse = await fetch('/api/upload-frames', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: backendSessionId,
                    frames: this.capturedFrames
                })
            });
            
            if (!uploadResponse.ok) {
                throw new Error(`Frame upload failed: ${uploadResponse.status}`);
            }
            
            const uploadData = await uploadResponse.json();
            if (!uploadData.success) {
                throw new Error(uploadData.error || 'Failed to upload frames');
            }
            
            this.updateStatus('Frames uploaded. Analyzing emotions...', 'info');
            
            // Step 3: Analyze emotions
            const analysisResponse = await fetch('/api/analyze-emotions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: backendSessionId
                })
            });
            
            if (!analysisResponse.ok) {
                throw new Error(`Analysis failed: ${analysisResponse.status}`);
            }
            
            const analysisData = await analysisResponse.json();
            if (!analysisData.success) {
                throw new Error(analysisData.error || 'Failed to analyze emotions');
            }
            
            // Store analysis data for recommendations
            this.currentAnalysisData = analysisData;
            
            // Step 4: Get recommendations with emotions data
            this.updateStatus('Getting personalized recommendations...', 'info');
            const recResponse = await fetch('/api/get-recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    dominant_emotion: analysisData.dominant_emotion,
                    emotions: analysisData.average_emotions // Pass the emotions data
                })
            });
            
            let recData = { success: false };
            if (recResponse.ok) {
                recData = await recResponse.json();
            }
            
            // Display results
            this.displayResults(analysisData.results, analysisData.average_emotions);
            this.generateRecommendations(recData);
            
            const analysisType = analysisData.deepface_used ? 'DeepFace' : 'mock';
            this.updateStatus(`Analysis complete using ${analysisType}! Processed ${analysisData.successful_analyses}/${analysisData.total_frames} frames.`, 'info');
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.updateStatus(`Analysis failed: ${error.message}`, 'error');
        }
        
        this.analyzeBtn.disabled = false;
    }

    displayResults(results, averageEmotions) {
        this.resultsSection.style.display = 'block';
        this.emotionResults.innerHTML = '';
        
        const emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral'];
        
        // Display results
        const resultHTML = `
            <h4>📈 Average Emotion Distribution (${results.length} frames)</h4>
            <div style="margin: 20px 0;">
                ${emotions.map(emotion => `
                    <div class="emotion-result">
                        <span style="font-weight: bold; text-transform: capitalize;">
                            ${this.getEmotionEmoji(emotion)} ${emotion}
                        </span>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div class="emotion-bar">
                                <div class="emotion-fill" style="width: ${averageEmotions[emotion]}%; background: ${this.getEmotionColor(emotion)};"></div>
                            </div>
                            <span style="font-weight: bold; min-width: 60px;">${averageEmotions[emotion]}%</span>
                        </div>
                    </div>
                `).join('')}
            </div>
            
            <details style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                <summary style="cursor: pointer; font-weight: bold;">📋 Detailed Frame-by-Frame Results</summary>
                <div style="margin-top: 15px;">
                    ${results.map(result => `
                        <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px; border-left: 3px solid #3498db;">
                            <strong>Frame ${result.frame}</strong> - ${new Date(result.timestamp).toLocaleTimeString()}
                            ${result.success ? `
                                <div style="font-size: 0.9em; margin-top: 5px;">
                                    Dominant: ${result.dominant_emotion} | 
                                    ${emotions.map(emotion => 
                                        `${emotion}: ${result.emotions[emotion]}%`
                                    ).join(' | ')}
                                </div>
                            ` : `
                                <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                                    Error: ${result.error || 'Analysis failed'}
                                </div>
                            `}
                        </div>
                    `).join('')}
                </div>
            </details>
        `;
        
        this.emotionResults.innerHTML = resultHTML;
    }

    getEmotionEmoji(emotion) {
        const emojis = {
            happy: '😊',
            sad: '😢',
            angry: '😠',
            surprise: '😮',
            fear: '😨',
            disgust: '🤢',
            neutral: '😐'
        };
        return emojis[emotion] || '😐';
    }

    getEmotionColor(emotion) {
        const colors = {
            happy: '#2ecc71',
            sad: '#3498db',
            angry: '#e74c3c',
            surprise: '#f39c12',
            fear: '#9b59b6',
            disgust: '#1abc9c',
            neutral: '#95a5a6'
        };
        return colors[emotion] || '#95a5a6';
    }

    generateRecommendations(recommendationData) {
        if (!recommendationData.success) {
            this.recommendationsSection.style.display = 'none';
            return;
        }

        const recommendations = recommendationData.recommendations || [];
        const dominantEmotion = recommendationData.dominant_emotion || 'neutral';
        const generalTip = recommendationData.general_tip || 'Practice emotional awareness regularly.';
        const mindAgeData = recommendationData.mind_age_analysis;
        
        this.recommendationsSection.style.display = 'block';
        
        let recommendationsHTML = `
            <p style="margin-bottom: 15px; font-style: italic;">
                Based on your dominant emotion (<strong>${dominantEmotion}</strong>), here are some personalized suggestions:
            </p>
            ${recommendations.map(rec => `
                <div class="recommendation-item">${rec}</div>
            `).join('')}
        `;

        // Add Mind Age Analysis section if available
        if (mindAgeData) {
            recommendationsHTML += `
                <div class="mind-age-analysis" style="margin: 25px 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0 0 15px 0; text-align: center; font-size: 1.3em;">🧠 Psychological Age Analysis</h4>
                    
                    <div class="mind-age-card" style="display: flex; flex-direction: column; gap: 15px;">
                        <div class="age-display" style="text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px; backdrop-filter: blur(10px);">
                            <span class="age-number" style="font-size: 3em; font-weight: bold; display: block; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">${mindAgeData.estimated_mind_age}</span>
                            <span class="age-label" style="font-size: 1.2em; opacity: 0.8;">years old (mentally)</span>
                        </div>
                        
                        <div class="age-details">
                            <div style="margin: 8px 0; padding: 12px; background: rgba(255,255,255,0.1); border-radius: 8px; border-left: 4px solid rgba(255,255,255,0.5);">
                                <strong>📊 Age Range:</strong> ${mindAgeData.age_range}
                            </div>
                            <div style="margin: 8px 0; padding: 12px; background: rgba(255,255,255,0.1); border-radius: 8px; border-left: 4px solid rgba(255,255,255,0.5);">
                                <strong>🎭 Personality Type:</strong> ${mindAgeData.personality_type}
                            </div>
                            <div style="margin: 8px 0; padding: 12px; background: rgba(255,255,255,0.1); border-radius: 8px; border-left: 4px solid rgba(255,255,255,0.5);">
                                <strong>🧩 Emotional Intelligence:</strong> ${mindAgeData.emotional_intelligence}
                            </div>
                            <div style="margin: 8px 0; padding: 12px; background: rgba(255,255,255,0.15); border-radius: 8px; font-style: italic; opacity: 0.9;">
                                ${mindAgeData.ei_description}
                            </div>
                        </div>
                        
                        <div class="interpretation" style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.2); font-weight: 500; line-height: 1.4;">
                            <strong>💭 Analysis:</strong> ${mindAgeData.interpretation}
                        </div>
                    </div>
                </div>
            `;
        }

        recommendationsHTML += `
            <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #f39c12;">
                <strong>💡 General Tip:</strong> ${generalTip}
            </div>
        `;
        
        this.recommendationsList.innerHTML = recommendationsHTML;
    }

    retakeTest() {
        // Reset everything for a new test
        this.capturedFrames = [];
        this.currentFrame = 0;
        this.framesThumbnails.innerHTML = '';
        this.progressFill.style.width = '0%';
        this.progressBar.style.display = 'none';
        this.captureStatus.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.currentAnalysisData = null; // Reset analysis data
        
        // Generate new session ID
        this.sessionID = this.generateSessionID();
        this.updateSessionInfo();
        
        // Reset button states
        this.captureBtn.disabled = false;
        this.analyzeBtn.disabled = true;
        this.retakeBtn.disabled = true;
        
        this.updateStatus('Ready for new test. Click "Capture Frames" to start.', 'info');
        
        // Clear any intervals
        if (this.captureInterval) {
            clearInterval(this.captureInterval);
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.videoElement.style.display = 'none';
        this.cameraPlaceholder.style.display = 'flex';
        this.startBtn.disabled = false;
    }
}

// Initialize the system when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const emotionSystem = new EmotionDetectionSystem();
    
    // Handle page unload to cleanup camera
    window.addEventListener('beforeunload', () => {
        if (emotionSystem.stream) {
            emotionSystem.stopCamera();
        }
    });
});