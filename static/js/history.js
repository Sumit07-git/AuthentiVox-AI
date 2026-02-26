
document.addEventListener('DOMContentLoaded', function() {
    
    const historyContainer = document.getElementById('historyContainer');
    const historyList = document.getElementById('historyList');
    const emptyState = document.getElementById('emptyState');
    const filterResult = document.getElementById('filterResult');
    const sortBy = document.getElementById('sortBy');
    const clearHistoryBtn = document.getElementById('clearHistory');
    const downloadAllBtn = document.getElementById('downloadAllReports');
    const pagination = document.getElementById('pagination');
    const prevPageBtn = document.getElementById('prevPage');
    const nextPageBtn = document.getElementById('nextPage');
    const pageInfo = document.getElementById('pageInfo');
    
    
    const reportModal = document.getElementById('reportModal');
    const modalOverlay = document.getElementById('modalOverlay');
    const modalClose = document.getElementById('modalClose');
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    const modalBody = document.getElementById('modalBody');
    const downloadThisReport = document.getElementById('downloadThisReport');
    
    let currentPage = 1;
    const itemsPerPage = 10;
    let currentHistory = [];
    let selectedReport = null;
    
    
    loadHistory();
    
    
    filterResult.addEventListener('change', loadHistory);
    sortBy.addEventListener('change', loadHistory);
    clearHistoryBtn.addEventListener('click', clearAllHistory);
    downloadAllBtn.addEventListener('click', downloadAllReports);
    prevPageBtn.addEventListener('click', () => changePage(-1));
    nextPageBtn.addEventListener('click', () => changePage(1));
    
    
    modalClose.addEventListener('click', closeModal);
    modalCloseBtn.addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', closeModal);
    downloadThisReport.addEventListener('click', downloadCurrentReport);
    
    
    function loadHistory() {
        try {
            
            let history = JSON.parse(localStorage.getItem('analysisHistory') || '[]');
            
            
            const filter = filterResult.value;
            if (filter === 'real') {
                history = history.filter(item => !item.is_fake);
            } else if (filter === 'fake') {
                history = history.filter(item => item.is_fake);
            }
            
            
            const sort = sortBy.value;
            history.sort((a, b) => {
                switch(sort) {
                    case 'newest':
                        return new Date(b.timestamp) - new Date(a.timestamp);
                    case 'oldest':
                        return new Date(a.timestamp) - new Date(b.timestamp);
                    case 'confidence-high':
                        return b.confidence - a.confidence;
                    case 'confidence-low':
                        return a.confidence - b.confidence;
                    default:
                        return 0;
                }
            });
            
            currentHistory = history;
            currentPage = 1;
            
            displayHistory();
        } catch (error) {
            console.error('Error loading history:', error);
            showEmptyState();
        }
    }
    
    
    function displayHistory() {
        if (currentHistory.length === 0) {
            showEmptyState();
            return;
        }
        
        hideEmptyState();
        
        
        const totalPages = Math.ceil(currentHistory.length / itemsPerPage);
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        const pageItems = currentHistory.slice(startIndex, endIndex);
        
        
        historyList.innerHTML = '';
        
        
        pageItems.forEach(item => {
            const historyItem = createHistoryItem(item);
            historyList.appendChild(historyItem);
        });
        
        
        updatePagination(totalPages);
    }
    
    
    function createHistoryItem(item) {
        const div = document.createElement('div');
        div.className = 'history-item';
        
        const resultClass = item.is_fake ? 'fake' : 'real';
        const resultIcon = item.is_fake ? 'fa-times-circle' : 'fa-check-circle';
        const resultText = item.is_fake ? 'FAKE' : 'REAL';
        
        const date = new Date(item.timestamp);
        const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        
        div.innerHTML = `
            <div class="history-item-icon ${resultClass}">
                <i class="fas ${resultIcon}"></i>
            </div>
            <div class="history-item-info">
                <h4>${item.filename}</h4>
                <div class="history-meta">
                    <span><i class="fas fa-clock"></i> ${formattedDate}</span>
                    <span><i class="fas fa-file"></i> ${formatFileSize(item.filesize)}</span>
                </div>
            </div>
            <div class="history-item-result">
                <span class="result-badge ${resultClass}">${resultText}</span>
                <span class="confidence-badge">${item.confidence}%</span>
            </div>
            <div class="history-item-actions">
                <button class="btn-icon view-report" data-id="${item.id}" title="View Report">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="btn-icon download-report" data-id="${item.id}" title="Download Report">
                    <i class="fas fa-download"></i>
                </button>
                <button class="btn-icon delete-report" data-id="${item.id}" title="Delete">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        
        
        div.querySelector('.view-report').addEventListener('click', () => viewReport(item));
        div.querySelector('.download-report').addEventListener('click', () => downloadReport(item));
        div.querySelector('.delete-report').addEventListener('click', () => deleteReport(item.id));
        
        return div;
    }
    
    
    function viewReport(item) {
        selectedReport = item;
        
        const resultClass = item.is_fake ? 'fake' : 'real';
        const resultText = item.is_fake ? 'FAKE' : 'REAL';
        const date = new Date(item.timestamp);
        
        modalBody.innerHTML = `
            <div class="report-detail">
                <div class="report-header">
                    <h3>${item.filename}</h3>
                    <span class="result-badge-large ${resultClass}">${resultText}</span>
                </div>
                
                <div class="report-info">
                    <div class="report-row">
                        <span class="label">Analysis Date:</span>
                        <span class="value">${date.toLocaleString()}</span>
                    </div>
                    <div class="report-row">
                        <span class="label">File Size:</span>
                        <span class="value">${formatFileSize(item.filesize)}</span>
                    </div>
                    <div class="report-row">
                        <span class="label">Prediction:</span>
                        <span class="value ${resultClass}">${resultText}</span>
                    </div>
                    <div class="report-row">
                        <span class="label">Confidence Score:</span>
                        <span class="value">${item.confidence}%</span>
                    </div>
                </div>
                
                <div class="confidence-meter">
                    <div class="meter-label">
                        <span>Confidence Level</span>
                        <span>${item.confidence}%</span>
                    </div>
                    <div class="meter-bar">
                        <div class="meter-fill ${resultClass}" style="width: ${item.confidence}%"></div>
                    </div>
                </div>
            </div>
        `;
        
        reportModal.style.display = 'block';
    }
    
    
    function closeModal() {
        reportModal.style.display = 'none';
        selectedReport = null;
    }
    
    
    function downloadReport(item) {
        const resultText = item.is_fake ? 'FAKE' : 'REAL';
        const date = new Date(item.timestamp);
        
        const reportContent = `
DEEPFAKE AUDIO DETECTION REPORT
================================

File: ${item.filename}
Size: ${formatFileSize(item.filesize)}
Analysis Date: ${date.toLocaleString()}

ANALYSIS RESULT
---------------
Prediction: ${resultText}
Confidence: ${item.confidence}%

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
Report generated by DeepGuard AI
Report ID: ${item.id}
`;
        
        const blob = new Blob([reportContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${item.id}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    
    function downloadCurrentReport() {
        if (selectedReport) {
            downloadReport(selectedReport);
        }
    }
    
    
    function downloadAllReports() {
        if (currentHistory.length === 0) {
            alert('No reports to download');
            return;
        }
        
        let allReports = 'DEEPFAKE AUDIO DETECTION - ALL REPORTS\n';
        allReports += '=' .repeat(50) + '\n\n';
        
        currentHistory.forEach((item, index) => {
            const resultText = item.is_fake ? 'FAKE' : 'REAL';
            const date = new Date(item.timestamp);
            
            allReports += `REPORT #${index + 1}\n`;
            allReports += '-'.repeat(50) + '\n';
            allReports += `File: ${item.filename}\n`;
            allReports += `Date: ${date.toLocaleString()}\n`;
            allReports += `Result: ${resultText}\n`;
            allReports += `Confidence: ${item.confidence}%\n`;
            allReports += '\n';
        });
        
        allReports += '\n---\nGenerated by DeepGuard AI\n';
        
        const blob = new Blob([allReports], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `all_reports_${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    
    function deleteReport(id) {
        if (!confirm('Are you sure you want to delete this report?')) {
            return;
        }
        
        try {
            let history = JSON.parse(localStorage.getItem('analysisHistory') || '[]');
            history = history.filter(item => item.id !== id);
            localStorage.setItem('analysisHistory', JSON.stringify(history));
            loadHistory();
        } catch (error) {
            console.error('Error deleting report:', error);
        }
    }
    
    
    function clearAllHistory() {
        if (!confirm('Are you sure you want to delete ALL history? This cannot be undone.')) {
            return;
        }
        
        localStorage.removeItem('analysisHistory');
        loadHistory();
    }
    
    
    function updatePagination(totalPages) {
        if (totalPages <= 1) {
            pagination.style.display = 'none';
            return;
        }
        
        pagination.style.display = 'flex';
        pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
        
        prevPageBtn.disabled = currentPage === 1;
        nextPageBtn.disabled = currentPage === totalPages;
    }
    
    function changePage(delta) {
        currentPage += delta;
        displayHistory();
    }
    
    
    function showEmptyState() {
        emptyState.style.display = 'block';
        historyList.innerHTML = '';
        pagination.style.display = 'none';
    }
    
    function hideEmptyState() {
        emptyState.style.display = 'none';
    }
    
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }
});