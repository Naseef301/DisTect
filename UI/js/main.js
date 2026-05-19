




// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const recordBtn = document.getElementById('recordBtn');
const recordBtnText = document.getElementById('recordBtnText');
const predictBtn = document.getElementById('predictBtn');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const audioPlayer = document.getElementById('audioPlayer');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const resultsContainer = document.getElementById('resultsContainer');
const predictionResults = document.getElementById('predictionResults');
const distressCard = document.getElementById('distressCard');
const distressLabel = document.getElementById('distressLabel');
const distressProbability = document.getElementById('distressProbability');
const emotionBars = document.getElementById('emotionBars');
const primaryEmotion = document.getElementById('primaryEmotion');

// State
let selectedFile = null;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let audioURL = null;

// Use relative URL
const API_URL = '';

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
    setTimeout(() => { errorMessage.style.display = 'none'; }, 7000);
}

function hideError() {
    errorMessage.style.display = 'none';
}

function showLoading() {
    loading.style.display = 'block';
    predictBtn.disabled = true;
}

function hideLoading() {
    loading.style.display = 'none';
    predictBtn.disabled = false;
}

function updateFileInfo(file) {
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.style.display = 'block';

    if (audioURL) URL.revokeObjectURL(audioURL);
    audioURL = URL.createObjectURL(file);
    audioPlayer.src = audioURL;
    audioPlayer.style.display = 'block';

    predictBtn.disabled = false;
}

function clearFileInfo() {
    selectedFile = null;
    fileInfo.style.display = 'none';
    audioPlayer.style.display = 'none';
    predictBtn.disabled = true;

    if (audioURL) {
        URL.revokeObjectURL(audioURL);
        audioURL = null;
    }
}

uploadBtn.addEventListener('click', () => {
    if (isRecording) stopRecording();
    fileInput.click();
});

fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];

    if (file) {
        const allowedTypes = [
            'audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg',
            'audio/flac', 'audio/x-m4a', 'audio/mp4', 'audio/webm',
            'audio/x-wav', 'audio/wave'
        ];
        const allowedExts = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();

        if (!allowedTypes.includes(file.type) && !allowedExts.includes(ext)) {
            showError('Unsupported file type. Please upload WAV, MP3, OGG, FLAC, M4A, or WEBM.');
            event.target.value = '';
            return;
        }

        selectedFile = file;
        updateFileInfo(file);
        hideError();
        predictionResults.style.display = 'none';
    }

    event.target.value = '';
});

recordBtn.addEventListener('click', async () => {
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : MediaRecorder.isTypeSupported('audio/webm')
                ? 'audio/webm'
                : '';

        mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const type = mediaRecorder.mimeType || 'audio/webm';
            const ext = type.includes('ogg') ? 'ogg' : 'webm';
            const audioBlob = new Blob(audioChunks, { type });
            const recordedFile = new File([audioBlob], `recorded-audio.${ext}`, { type });

            selectedFile = recordedFile;
            updateFileInfo(recordedFile);

            stream.getTracks().forEach(track => track.stop());
            hideError();
        };

        mediaRecorder.start();
        isRecording = true;
        recordBtn.classList.add('recording');
        recordBtnText.textContent = 'Stop Recording';
        uploadBtn.disabled = true;
        predictBtn.disabled = true;
        clearFileInfo();

    } catch (error) {
        console.error('Microphone error:', error);
        showError('Cannot access microphone. Please allow microphone permission and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        isRecording = false;
        recordBtn.classList.remove('recording');
        recordBtnText.textContent = 'Record Audio';
        uploadBtn.disabled = false;
    }
}

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please upload or record audio first.');
        return;
    }

    hideError();
    showLoading();
    predictionResults.style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `Server error: ${response.status}`);
        }

        displayResults(data);

    } catch (error) {
        console.error('Prediction error:', error);

        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            showError('Cannot reach the server. Make sure Flask is running (python app.py).');
        } else {
            showError(error.message || 'Prediction failed. Please try again.');
        }
    } finally {
        hideLoading();
    }
});

function displayResults(data) {
    const welcomeIllustration = document.querySelector('.welcome-illustration');
    if (welcomeIllustration) welcomeIllustration.style.display = 'none';

    predictionResults.style.display = 'block';

    const isDistress = data.distress.label === 'Distress';
    distressCard.className = 'distress-card ' + (isDistress ? 'distress' : 'non-distress');
    distressLabel.textContent = (isDistress ? '🔴 ' : '🟢 ') + data.distress.label;
    distressProbability.textContent = data.distress.percentage.toFixed(1) + '%';

    emotionBars.innerHTML = '';
    const emotionEntries = Object.entries(data.emotions).sort((a, b) => b[1].percentage - a[1].percentage);

    emotionEntries.forEach(([emotion, values]) => {
        const emotionBar = document.createElement('div');
        emotionBar.className = `emotion-bar emotion-${emotion}`;
        emotionBar.innerHTML = `
            <div class="emotion-bar-header">
                <span class="emotion-name">${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</span>
                <span class="emotion-percentage">${values.percentage.toFixed(1)}%</span>
            </div>
            <div class="emotion-bar-fill-container">
                <div class="emotion-bar-fill" style="width: ${values.percentage}%"></div>
            </div>`;
        emotionBars.appendChild(emotionBar);
    });


}

async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();

        if (data.status !== 'healthy') {
            showError('Backend not ready. Please wait a moment and refresh.');
        }
    } catch {
        showError('Cannot connect to server. Make sure Flask is running: python app.py');
    }
}

function init() {
    console.log('Voice Distress Detection App Initialized');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        recordBtn.disabled = true;
        recordBtn.title = 'Recording not supported in this browser';
    }

    checkAPIHealth();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

window.addEventListener('beforeunload', () => {
    if (audioURL) URL.revokeObjectURL(audioURL);
    if (isRecording) stopRecording();
});

