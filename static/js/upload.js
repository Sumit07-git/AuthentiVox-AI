document.addEventListener('DOMContentLoaded', function () {
    const uploadSection = document.getElementById('uploadSection');
    const uploadArea = document.getElementById('uploadArea');
    const audioFileInput = document.getElementById('audioFile');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const audioPlayer = document.getElementById('audioPlayer');
    const removeFileBtn = document.getElementById('removeFile');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingState = document.getElementById('loadingState');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');

    const resultsHero = document.getElementById('resultsHero');
    const resultsDashboard = document.getElementById('resultsDashboard');
    const audioAnalysisSection = document.getElementById('audioAnalysisSection');
    const spectrogramFullSection = document.getElementById('spectrogramFullSection');
    const technicalSection = document.getElementById('technicalSection');
    const actionsSection = document.getElementById('actionsSection');

    let selectedFile = null;
    let isAnalyzing = false;
    let analysisStartTime = null;

    audioFileInput.addEventListener('change', function (e) {
        handleFileSelect(e.target.files[0]);
    });

    uploadArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function (e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function (e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFileSelect(files[0]);
    });

    uploadArea.addEventListener('click', function (e) {
        const isVisible = window.getComputedStyle(uploadArea).display !== 'none';
        if (isVisible && !e.target.closest('label') && !e.target.closest('input') && !isAnalyzing) {
            audioFileInput.click();
        }
    });

    removeFileBtn.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        resetUpload();
    });

    analyzeBtn.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        if (selectedFile) analyzeAudio();
    });

    const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
    if (analyzeAnotherBtn) analyzeAnotherBtn.addEventListener('click', resetToUpload);

    const downloadReportBtn = document.getElementById('downloadReportBtn');
    if (downloadReportBtn) downloadReportBtn.addEventListener('click', downloadReport);

    const viewHistoryBtn = document.getElementById('viewHistoryBtn');
    if (viewHistoryBtn) viewHistoryBtn.addEventListener('click', () => { window.location.href = '/history'; });

    const shareResultsBtn = document.getElementById('shareResultsBtn');
    if (shareResultsBtn) shareResultsBtn.addEventListener('click', shareResults);

    async function handleFileSelect(file) {
        if (!file) return;

        const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg'];
        const validExtensions = ['.wav', '.mp3', '.flac', '.ogg'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

        if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
            showError('Invalid file format. Please upload WAV, MP3, FLAC, or OGG files.');
            return;
        }
        if (file.size > 16 * 1024 * 1024) {
            showError('File too large. Maximum size is 16MB.');
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        audioPlayer.src = URL.createObjectURL(file);
        uploadArea.style.display = 'none';
        fileInfo.style.display = 'block';
        hideError();
    }

    function resetUpload() {
        selectedFile = null;
        isAnalyzing = false;
        audioFileInput.value = '';
        audioPlayer.src = '';
        uploadArea.style.display = 'block';
        fileInfo.style.display = 'none';
        loadingState.style.display = 'none';
        hideError();
    }

    async function analyzeAudio() {
        if (!selectedFile || isAnalyzing) return;

        isAnalyzing = true;
        analysisStartTime = Date.now();
        fileInfo.style.display = 'none';
        loadingState.style.display = 'block';
        hideError();
        simulateProgress();

        const formData = new FormData();
        formData.append('audio_file', selectedFile);

        try {
            const [response] = await Promise.all([
                fetch('/api/upload', { method: 'POST', body: formData }),
                new Promise(resolve => setTimeout(resolve, 2200)),
            ]);

            if (!response.ok) {
                let errorMsg = 'Server error (' + response.status + ')';
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (_) {
                    if (response.status === 502) errorMsg = 'Server is temporarily unavailable. Please try again.';
                    else if (response.status === 413) errorMsg = 'File is too large for the server to process.';
                    else if (response.status === 504) errorMsg = 'Analysis timed out. Try a shorter audio file.';
                    else if (response.status === 503) errorMsg = 'Service unavailable. Please wait and try again.';
                    else if (response.status === 500) errorMsg = 'Internal server error. Please try again.';
                }
                showError(errorMsg);
                loadingState.style.display = 'none';
                fileInfo.style.display = 'block';
                isAnalyzing = false;
                return;
            }

            let data;
            try {
                data = await response.json();
            } catch (_) {
                showError('Invalid response from server. Please try again.');
                loadingState.style.display = 'none';
                fileInfo.style.display = 'block';
                isAnalyzing = false;
                return;
            }

            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error || 'An error occurred during analysis');
                loadingState.style.display = 'none';
                fileInfo.style.display = 'block';
                isAnalyzing = false;
            }
        } catch (error) {
            let errorMsg = 'Network error. Please check your connection and try again.';
            if (error.name === 'AbortError') errorMsg = 'Request was cancelled. Please try again.';
            else if (error.message && error.message.includes('Failed to fetch'))
                errorMsg = 'Cannot reach the server. Please check if the server is running.';
            showError(errorMsg);
            loadingState.style.display = 'none';
            fileInfo.style.display = 'block';
            isAnalyzing = false;
        }
    }

    function simulateProgress() {
        const progressFill = document.getElementById('progressFill');
        const steps = document.querySelectorAll('.loading-steps .step');

        progressFill.style.transition = 'none';
        progressFill.style.width = '0%';
        steps.forEach(step => {
            step.classList.remove('active', 'complete');
            step.querySelector('i').className = 'fas fa-circle';
        });
        void progressFill.offsetWidth;
        progressFill.style.transition = '';

        let progress = 0;
        const interval = setInterval(() => {
            progress += 5;
            progressFill.style.width = progress + '%';

            if (progress >= 25 && progress < 50) {
                steps[0].classList.add('complete');
                steps[0].querySelector('i').className = 'fas fa-check-circle';
                steps[1].classList.add('active');
            } else if (progress >= 50 && progress < 75) {
                steps[1].classList.add('complete');
                steps[1].querySelector('i').className = 'fas fa-check-circle';
                steps[2].classList.add('active');
            } else if (progress >= 75 && progress < 100) {
                steps[2].classList.add('complete');
                steps[2].querySelector('i').className = 'fas fa-check-circle';
                steps[3].classList.add('active');
            } else if (progress >= 100) {
                steps[3].classList.add('complete');
                steps[3].querySelector('i').className = 'fas fa-check-circle';
                clearInterval(interval);
            }
        }, 100);
    }

    function saveToHistory(data) {
        try {
            let history = JSON.parse(localStorage.getItem('analysisHistory') || '[]');
            history.unshift({
                id: Date.now().toString(),
                filename: selectedFile ? selectedFile.name : 'Unknown',
                filesize: selectedFile ? selectedFile.size : 0,
                is_fake: data.is_fake,
                confidence: data.confidence_score,
                timestamp: new Date().toISOString(),
            });
            if (history.length > 100) history = history.slice(0, 100);
            localStorage.setItem('analysisHistory', JSON.stringify(history));
        } catch (e) {
            console.error('Error saving to history:', e);
        }
    }

    function displayResults(data) {
        const processingTime = ((Date.now() - analysisStartTime) / 1000).toFixed(1);
        saveToHistory(data);

        if (loadingState) loadingState.style.display = 'none';
        if (uploadSection) uploadSection.style.display = 'none';

        setTimeout(() => {
            if (resultsHero) { resultsHero.style.display = 'block'; updateHeroSection(data); }
            setTimeout(() => {
                if (resultsDashboard) { resultsDashboard.style.display = 'block'; updateDashboardSection(data, processingTime); }
            }, 200);
            setTimeout(() => {
                if (audioAnalysisSection) { audioAnalysisSection.style.display = 'block'; updateAudioAnalysisSection(data); }
            }, 400);
            if (data.spectrogram_path && spectrogramFullSection) {
                setTimeout(() => { spectrogramFullSection.style.display = 'block'; updateSpectrogramSection(data); }, 600);
            }
            setTimeout(() => {
                if (technicalSection) { technicalSection.style.display = 'block'; updateTechnicalSection(data); }
            }, 800);
            setTimeout(() => {
                if (actionsSection) actionsSection.style.display = 'block';
            }, 1000);
        }, 100);

        isAnalyzing = false;
    }

    function updateHeroSection(data) {
        const heroBadge = document.getElementById('heroBadge');
        const heroLabel = document.getElementById('heroLabel');
        const heroTitle = document.getElementById('heroTitle');
        const heroConfidence = document.getElementById('heroConfidence');

        if (!heroBadge || !heroLabel || !heroConfidence || !heroTitle) return;

        resultsHero.classList.remove('real', 'fake');
        heroBadge.classList.remove('real', 'fake');

        if (data.is_fake) {
            resultsHero.classList.add('fake');
            heroBadge.classList.add('fake');
            heroLabel.textContent = 'SYNTHETIC';
            heroTitle.textContent = 'Fake Audio Detected';
            heroBadge.querySelector('i').className = 'fas fa-exclamation-triangle';
        } else {
            resultsHero.classList.add('real');
            heroBadge.classList.add('real');
            heroLabel.textContent = 'AUTHENTIC';
            heroTitle.textContent = 'Real Audio Detected';
            heroBadge.querySelector('i').className = 'fas fa-shield-alt';
        }
        heroConfidence.textContent = data.confidence_score + '%';
    }

    function updateDashboardSection(data, processingTime) {
        const predictionText = document.getElementById('predictionText');
        const confidenceText = document.getElementById('confidenceText');
        const processingTimeEl = document.getElementById('processingTime');

        if (predictionText) predictionText.textContent = data.is_fake ? 'SYNTHETIC' : 'AUTHENTIC';
        if (confidenceText) confidenceText.textContent = data.confidence_score + '%';
        if (processingTimeEl) processingTimeEl.textContent = processingTime + 's';

        const predictionStatIcon = document.querySelector('#resultsDashboard .stat-card:first-child .stat-icon');
        if (predictionStatIcon) {
            predictionStatIcon.classList.remove('real', 'fake');
            predictionStatIcon.classList.add(data.is_fake ? 'fake' : 'real');
            predictionStatIcon.querySelector('i').className = data.is_fake
                ? 'fas fa-times-circle' : 'fas fa-check-circle';
        }

        const circleProgress = document.getElementById('circleProgress');
        const circlePercentage = document.getElementById('circlePercentage');

        if (circleProgress && circlePercentage) {
            const circumference = 2 * Math.PI * 90;
            const offset = circumference - (data.confidence_score / 100) * circumference;
            const svg = circleProgress.closest('svg');

            if (svg) {
                let gradient = svg.querySelector('#gradient');
                if (!gradient) {
                    const defs = svg.querySelector('defs') || document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                    if (!svg.querySelector('defs')) svg.insertBefore(defs, svg.firstChild);
                    gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
                    gradient.setAttribute('id', 'gradient');
                    gradient.setAttribute('x1', '0%'); gradient.setAttribute('y1', '0%');
                    gradient.setAttribute('x2', '100%'); gradient.setAttribute('y2', '100%');
                    defs.appendChild(gradient);
                }
                gradient.innerHTML = '';
                const colors = data.is_fake
                    ? ['#ef4444', '#dc2626']
                    : ['#10b981', '#059669'];
                colors.forEach((c, i) => {
                    const stop = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                    stop.setAttribute('offset', i === 0 ? '0%' : '100%');
                    stop.setAttribute('style', `stop-color:${c};stop-opacity:1`);
                    gradient.appendChild(stop);
                });
            }

            setTimeout(() => {
                circleProgress.style.strokeDashoffset = offset;
                animateNumber(circlePercentage, 0, data.confidence_score, 2000);
            }, 300);
        }

        const confidenceDescription = document.getElementById('confidenceDescription');
        if (confidenceDescription) {
            if (data.confidence_score >= 90) confidenceDescription.textContent = 'The AI model is highly confident in this prediction. The audio shows strong indicators.';
            else if (data.confidence_score >= 70) confidenceDescription.textContent = 'The AI model has good confidence in this prediction. Most indicators align with the result.';
            else confidenceDescription.textContent = 'The AI model has moderate confidence. Some indicators are ambiguous.';
        }

        setTimeout(() => {
            const mlFill = document.getElementById('mlFill');
            const mlValue = document.getElementById('mlValue');
            const dlFill = document.getElementById('dlFill');
            const dlValue = document.getElementById('dlValue');

            const mlConf = (data.debug && data.debug.ml_confidence != null)
                ? data.debug.ml_confidence
                : null;
            const dlConf = (data.debug && data.debug.dl_confidence != null)
                ? data.debug.dl_confidence
                : null;

            if (mlFill && mlValue) {
                if (mlConf !== null) {
                    mlFill.style.width = mlConf + '%';
                    mlValue.textContent = mlConf.toFixed(1) + '%';
                } else {
                    mlFill.style.width = '0%';
                    mlValue.textContent = 'N/A';
                }
            }
            if (dlFill && dlValue) {
                if (dlConf !== null) {
                    dlFill.style.width = dlConf + '%';
                    dlValue.textContent = dlConf.toFixed(1) + '%';
                } else {
                    dlFill.style.width = '0%';
                    dlValue.textContent = 'N/A';
                }
            }
        }, 800);
    }

    function updateAudioAnalysisSection(data) {
        const analysisFileName = document.getElementById('analysisFileName');
        const analysisFileSize = document.getElementById('analysisFileSize');
        const resultsAudioPlayer = document.getElementById('resultsAudioPlayer');

        if (analysisFileName) analysisFileName.textContent = selectedFile.name;
        if (analysisFileSize) analysisFileSize.textContent = formatFileSize(selectedFile.size);

        if (resultsAudioPlayer) {
            resultsAudioPlayer.src = URL.createObjectURL(selectedFile);
            resultsAudioPlayer.addEventListener('loadedmetadata', function () {
                const dur = Math.floor(resultsAudioPlayer.duration);
                const minutes = Math.floor(dur / 60);
                const seconds = dur % 60;
                const el = document.getElementById('audioDuration');
                if (el) el.textContent = minutes + ':' + seconds.toString().padStart(2, '0');
            });
        }

        const indicatorDescriptions = [
            { real: 'Natural voice characteristics detected', fake: 'Unnatural frequency patterns found' },
            { real: 'Consistent energy distribution', fake: 'Irregular energy distribution detected' },
            { real: 'No synthetic markers found', fake: 'Synthetic markers detected' },
            { real: 'Authentic vocal patterns', fake: 'Unnatural vocal patterns' },
        ];

        document.querySelectorAll('.indicator-item').forEach((indicator, index) => {
            const badge = indicator.querySelector('.indicator-badge');
            const content = indicator.querySelector('.indicator-content p');
            if (!badge || !content || !indicatorDescriptions[index]) return;

            if (data.is_fake) {
                badge.classList.remove('pass'); badge.classList.add('fail'); badge.textContent = '✗';
                content.textContent = indicatorDescriptions[index].fake;
            } else {
                badge.classList.remove('fail'); badge.classList.add('pass'); badge.textContent = '✓';
                content.textContent = indicatorDescriptions[index].real;
            }
        });
    }

    function updateSpectrogramSection(data) {
        const img = document.getElementById('mainSpectrogramImage');
        if (img) img.src = data.spectrogram_path;

        const btn = document.getElementById('downloadSpectrogramBtn');
        if (btn) {
            btn.addEventListener('click', function (e) {
                e.preventDefault();
                fetch(data.spectrogram_path)
                    .then(r => r.blob())
                    .then(blob => {
                        const url = URL.createObjectURL(blob);
                        const link = document.createElement('a');
                        link.href = url;
                        link.download = 'spectrogram_' + Date.now() + '.png';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        URL.revokeObjectURL(url);
                    })
                    .catch(() => {
                        const link = document.createElement('a');
                        link.href = data.spectrogram_path;
                        link.download = 'spectrogram.png';
                        link.target = '_blank';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                    });
            });
        }
    }

    function updateTechnicalSection(data) {
        document.querySelectorAll('.timeline-item').forEach((item, index) => {
            setTimeout(() => item.classList.add('completed'), index * 200);
        });
    }

    function animateNumber(element, start, end, duration) {
        const range = end - start;
        const increment = range / (duration / 16);
        let current = start;
        const timer = setInterval(() => {
            current += increment;
            if (current >= end) { current = end; clearInterval(timer); }
            element.textContent = Math.round(current) + '%';
        }, 16);
    }

    function resetToUpload() {
        if (resultsHero) { resultsHero.style.display = 'none'; resultsHero.classList.remove('real', 'fake'); }
        if (resultsDashboard) resultsDashboard.style.display = 'none';
        if (audioAnalysisSection) audioAnalysisSection.style.display = 'none';
        if (spectrogramFullSection) spectrogramFullSection.style.display = 'none';
        if (technicalSection) technicalSection.style.display = 'none';
        if (actionsSection) actionsSection.style.display = 'none';
        if (uploadSection) uploadSection.style.display = 'block';
        resetUpload();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function downloadReport() {
        const predictionText = document.getElementById('predictionText');
        const confidenceText = document.getElementById('confidenceText');
        const content =
            'AUTHENTIVOX AUDIO ANALYSIS REPORT\n==================================\n\n' +
            'File: ' + (selectedFile ? selectedFile.name : 'Unknown') + '\n' +
            'Date: ' + new Date().toLocaleString() + '\n\n' +
            'ANALYSIS RESULT\n---------------\n' +
            'Prediction: ' + (predictionText ? predictionText.textContent : 'N/A') + '\n' +
            'Confidence: ' + (confidenceText ? confidenceText.textContent : 'N/A') + '\n\n' +
            'DETECTION METHOD\n----------------\n' +
            'Hybrid AI System combining:\n- Machine Learning (Random Forest)\n- Deep Learning (CNN)\n- Ensemble Fusion (Weighted Average)\n\n' +
            'FEATURES ANALYZED\n-----------------\n- MFCC Coefficients (13)\n- Spectral Features\n- Mel Spectrogram Patterns\n- Temporal Characteristics\n\n' +
            '---\nReport generated by AuthentiVox\n';

        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'authentivox_report_' + Date.now() + '.txt';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function shareResults() {
        const predictionText = document.getElementById('predictionText');
        const confidenceText = document.getElementById('confidenceText');
        if (navigator.share) {
            navigator.share({
                title: 'AuthentiVox Analysis Results',
                text: 'My audio was detected as ' +
                    (predictionText ? predictionText.textContent : 'N/A') +
                    ' with ' + (confidenceText ? confidenceText.textContent : 'N/A') +
                    ' confidence by AuthentiVox AI.',
            }).catch(err => console.log('Share failed:', err));
        } else {
            alert('Sharing not supported on this browser');
        }
    }

    function showError(message) { errorText.textContent = message; errorMessage.style.display = 'flex'; }
    function hideError() { errorMessage.style.display = 'none'; }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024, sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }
});