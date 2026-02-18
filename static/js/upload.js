

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const audioFileInput = document.getElementById('audioFile');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const audioPlayer = document.getElementById('audioPlayer');
    const removeFileBtn = document.getElementById('removeFile');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingState = document.getElementById('loadingState');
    const resultsContainer = document.getElementById('resultsContainer');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    
    let selectedFile = null;
    let isAnalyzing = false; // Prevent multiple simultaneous analyses
    let progressInterval = null; // Track interval so we can clear it
    
    // File input change
    audioFileInput.addEventListener('change', function(e) {
        handleFileSelect(e.target.files[0]);
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // Click handler for upload area - FIXED: Only trigger when truly visible
    uploadArea.addEventListener('click', function(e) {
        // Get computed style to check real visibility
        const isVisible = window.getComputedStyle(uploadArea).display !== 'none' &&
                         uploadArea.style.display !== 'none';
        
        // Only trigger if upload area is actually visible
        if (isVisible && 
            !e.target.closest('label') && 
            !e.target.closest('input') &&
            !isAnalyzing) { // Don't trigger during analysis
            console.log('Upload area clicked - opening file dialog');
            audioFileInput.click();
        } else {
            console.log('Upload area click ignored - not visible or analyzing');
        }
    });
    
    // Remove file - FIXED: Stop propagation
    removeFileBtn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        resetUpload();
    });
    
    // Analyze button - FIXED: Stop propagation to prevent triggering upload
    analyzeBtn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        console.log('Analyze button clicked');
        if (selectedFile) {
            analyzeAudio();
        } else {
            console.warn('No file selected');
        }
    });
    
    // Prevent file info section from triggering upload
    if (fileInfo) {
        fileInfo.addEventListener('click', function(e) {
            e.stopPropagation();
        });
    }
    
    // Prevent audio player from triggering upload
    if (audioPlayer) {
        audioPlayer.addEventListener('click', function(e) {
            e.stopPropagation();
        });
    }
    
    // Analyze another button
    const analyzeAnotherBtn = document.getElementById('analyzeAnother');
    if (analyzeAnotherBtn) {
        analyzeAnotherBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            resetUpload();
        });
    }
    
    // Download report button
    const downloadReportBtn = document.getElementById('downloadReport');
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            generateReport();
        });
    }
    
    // Handle file selection
    function handleFileSelect(file) {
        if (!file) return;
        
        // Validate file type
        const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg'];
        const validExtensions = ['.wav', '.mp3', '.flac', '.ogg'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
            showError('Invalid file format. Please upload WAV, MP3, FLAC, or OGG files.');
            return;
        }
        
        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            showError('File too large. Maximum size is 16MB.');
            return;
        }
        
        selectedFile = file;
        
        // Display file info
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        
        // Load audio into player
        const fileURL = URL.createObjectURL(file);
        audioPlayer.src = fileURL;
        
        // Show file info, hide upload area
        uploadArea.style.display = 'none';
        fileInfo.style.display = 'block';
        hideError();
    }
    
    // Reset upload
    function resetUpload() {
        selectedFile = null;
        isAnalyzing = false;
        audioFileInput.value = '';
        audioPlayer.src = '';
        uploadArea.style.display = 'block';
        fileInfo.style.display = 'none';
        loadingState.style.display = 'none';
        resultsContainer.style.display = 'none';
        hideError();
        console.log('Upload reset');
    }
    
    // Analyze audio
    async function analyzeAudio() {
        if (!selectedFile) {
            console.warn('No file selected for analysis');
            return;
        }
        
        if (isAnalyzing) {
            console.warn('Analysis already in progress');
            return;
        }
        
        isAnalyzing = true;
        console.log('Starting analysis...');
        
        // Hide file info and results, show loading
        fileInfo.style.display = 'none';
        resultsContainer.style.display = 'none';
        loadingState.style.display = 'block';
        hideError();
        
        // Start progress animation (takes ~2 seconds to complete)
        simulateProgress();
        
        // Prepare form data
        const formData = new FormData();
        formData.append('audio_file', selectedFile);
        
        try {
            // Run API call and minimum animation time IN PARALLEL
            // This ensures animation always plays fully regardless of how fast backend responds
            const [response] = await Promise.all([
                fetch('/api/upload', { method: 'POST', body: formData }),
                new Promise(resolve => setTimeout(resolve, 2200)) // Always wait 2.2s for animation
            ]);
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                displayResults(data);
            } else {
                showError(data.error || 'An error occurred during analysis');
                loadingState.style.display = 'none';
                fileInfo.style.display = 'block';
                isAnalyzing = false;
            }
        } catch (error) {
            console.error('Error:', error);
            showError('Network error. Please try again.');
            loadingState.style.display = 'none';
            fileInfo.style.display = 'block';
            isAnalyzing = false;
        }
    }
    
    // Simulate progress animation
    function simulateProgress() {
        const progressFill = document.getElementById('progressFill');
        const steps = document.querySelectorAll('.loading-steps .step');

        // ── Clear any previous interval ──────────────────────────
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }

        // ── Hard-reset progress bar to 0 ─────────────────────────
        progressFill.style.transition = 'none';   // kill transition so reset is instant
        progressFill.style.width = '0%';

        // ── Hard-reset all steps ──────────────────────────────────
        steps.forEach(step => {
            step.classList.remove('active', 'complete');
            const icon = step.querySelector('i');
            if (icon) icon.className = 'fas fa-circle';
        });

        // Force browser to repaint before re-enabling transition
        void progressFill.offsetWidth;
        progressFill.style.transition = '';        // restore transition

        let progress = 0;

        progressInterval = setInterval(() => {
            progress += 5;
            progressFill.style.width = progress + '%';

            if (progress >= 25 && progress < 50) {
                steps[0].classList.remove('active');
                steps[0].classList.add('complete');
                steps[0].querySelector('i').className = 'fas fa-check-circle';
                steps[1].classList.add('active');
            } else if (progress >= 50 && progress < 75) {
                steps[1].classList.remove('active');
                steps[1].classList.add('complete');
                steps[1].querySelector('i').className = 'fas fa-check-circle';
                steps[2].classList.add('active');
            } else if (progress >= 75 && progress < 100) {
                steps[2].classList.remove('active');
                steps[2].classList.add('complete');
                steps[2].querySelector('i').className = 'fas fa-check-circle';
                steps[3].classList.add('active');
            } else if (progress >= 100) {
                steps[3].classList.remove('active');
                steps[3].classList.add('complete');
                steps[3].querySelector('i').className = 'fas fa-check-circle';
                clearInterval(progressInterval);
                progressInterval = null;
            }
        }, 100);
    }
    
    // Display results - UPDATED: Only hybrid results with spectrogram
    function displayResults(data) {
        loadingState.style.display = 'none';
        resultsContainer.style.display = 'block';
        isAnalyzing = false; // ← Reset so next analysis works properly
        
        // Set result badge
        const resultBadge = document.getElementById('resultBadge');
        const resultLabel = document.getElementById('resultLabel');
        
        if (data.is_fake) {
            resultBadge.classList.remove('real');
            resultBadge.classList.add('fake');
            resultLabel.textContent = 'FAKE';
        } else {
            resultBadge.classList.remove('fake');
            resultBadge.classList.add('real');
            resultLabel.textContent = 'REAL';
        }
        
        // Set confidence score with dynamic color coding
        const confidenceValue = document.getElementById('confidenceValue');
        const meterFill = document.getElementById('meterFill');
        
        confidenceValue.textContent = data.confidence_score + '%';
        
        // Animate the confidence meter
        setTimeout(() => {
            meterFill.style.width = data.confidence_score + '%';
            
            // Dynamic color based on prediction and confidence
            if (data.is_fake) {
                // Fake audio - red gradient (higher confidence = darker red)
                if (data.confidence_score >= 80) {
                    meterFill.style.background = 'linear-gradient(90deg, #dc2626, #991b1b)';
                } else if (data.confidence_score >= 60) {
                    meterFill.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
                } else {
                    meterFill.style.background = 'linear-gradient(90deg, #f87171, #ef4444)';
                }
            } else {
                // Real audio - green gradient (higher confidence = darker green)
                if (data.confidence_score >= 80) {
                    meterFill.style.background = 'linear-gradient(90deg, #059669, #047857)';
                } else if (data.confidence_score >= 60) {
                    meterFill.style.background = 'linear-gradient(90deg, #10b981, #059669)';
                } else {
                    meterFill.style.background = 'linear-gradient(90deg, #34d399, #10b981)';
                }
            }
        }, 100);
        
        // Show spectrogram if available
        const spectrogramSection = document.getElementById('spectrogramSection');
        const spectrogramImage = document.getElementById('spectrogramImage');
        const spectrogramAudioPlayer = document.getElementById('spectrogramAudioPlayer');
        
        if (data.spectrogram_path) {
            spectrogramImage.src = data.spectrogram_path;
            spectrogramImage.style.opacity = '0';
            spectrogramSection.style.display = 'block';
            
            // Set audio source for playback with spectrogram
            if (selectedFile) {
                const audioURL = URL.createObjectURL(selectedFile);
                spectrogramAudioPlayer.src = audioURL;
            }
            
            // Fade in when loaded
            spectrogramImage.onload = function() {
                spectrogramImage.style.transition = 'opacity 0.5s ease';
                spectrogramImage.style.opacity = '1';
            };
            
            spectrogramImage.onerror = function() {
                console.warn('Failed to load spectrogram');
                spectrogramSection.style.display = 'none';
            };
            
            // Sync playback indicator with audio
            setupAudioSync(spectrogramAudioPlayer);
        } else {
            spectrogramSection.style.display = 'none';
        }
        
        // Save to history
        saveToHistory(data);
        
        // Scroll to results
        setTimeout(() => {
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);
    }
    
    // Setup audio sync with spectrogram
    function setupAudioSync(audioElement) {
        const indicator = document.getElementById('playbackIndicator');
        const container = document.querySelector('.spectrogram-container');
        
        if (!indicator || !audioElement || !container) return;
        
        // Update indicator position as audio plays
        audioElement.addEventListener('timeupdate', function() {
            if (!audioElement.duration) return;
            
            const progress = (audioElement.currentTime / audioElement.duration);
            const containerWidth = container.offsetWidth;
            const indicatorPosition = (progress * (containerWidth - 20)) + 10; // Account for padding
            
            indicator.style.left = indicatorPosition + 'px';
        });
        
        audioElement.addEventListener('play', function() {
            indicator.classList.add('playing');
        });
        
        audioElement.addEventListener('pause', function() {
            indicator.classList.remove('playing');
        });
        
        audioElement.addEventListener('ended', function() {
            indicator.classList.remove('playing');
            // Reset to start
            indicator.style.left = '10px';
        });
        
        audioElement.addEventListener('seeked', function() {
            // Update position when user seeks
            const progress = (audioElement.currentTime / audioElement.duration);
            const containerWidth = container.offsetWidth;
            const indicatorPosition = (progress * (containerWidth - 20)) + 10;
            indicator.style.left = indicatorPosition + 'px';
        });
    }
    
    // Save analysis to history
    function saveToHistory(data) {
        try {
            // Get existing history
            let history = JSON.parse(localStorage.getItem('analysisHistory') || '[]');
            
            // Create history entry
            const entry = {
                id: Date.now(),
                filename: selectedFile ? selectedFile.name : 'Unknown',
                filesize: selectedFile ? selectedFile.size : 0,
                prediction: data.prediction,
                is_fake: data.is_fake,
                confidence: data.confidence_score,
                timestamp: new Date().toISOString(),
                spectrogram_path: data.spectrogram_path
            };
            
            // Add to beginning of array
            history.unshift(entry);
            
            // Keep only last 50 entries
            if (history.length > 50) {
                history = history.slice(0, 50);
            }
            
            // Save back to localStorage
            localStorage.setItem('analysisHistory', JSON.stringify(history));
            
            console.log('Saved to history:', entry);
        } catch (error) {
            console.error('Error saving to history:', error);
        }
    }
    
    // Generate downloadable report
    function generateReport() {
        const resultLabel = document.getElementById('resultLabel').textContent;
        const confidenceValue = document.getElementById('confidenceValue').textContent;
        
        const reportContent = `
DEEPFAKE AUDIO DETECTION REPORT
================================

File: ${selectedFile ? selectedFile.name : 'Unknown'}
Date: ${new Date().toLocaleString()}

ANALYSIS RESULT
---------------
Prediction: ${resultLabel}
Confidence: ${confidenceValue}

DETECTION METHOD
----------------
Hybrid AI System combining:
- Machine Learning (Random Forest)
- Deep Learning (CNN)
- Ensemble Fusion (Weighted Average)

FEATURES ANALYZED
-----------------
- MFCC Coefficients (13)
- Spectral Features
- Mel Spectrogram Patterns
- Temporal Characteristics

---
Report generated by AuthentiVox AI
`;
        
        // Create blob and download
        const blob = new Blob([reportContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `deepfake_report_${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    // Show error message
    function showError(message) {
        errorText.textContent = message;
        errorMessage.style.display = 'flex';
        setTimeout(() => {
            errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
    }
    
    // Hide error message
    function hideError() {
        errorMessage.style.display = 'none';
    }
    
    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }
});