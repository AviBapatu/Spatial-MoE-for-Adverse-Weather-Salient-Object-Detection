const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

// UI Elements
const originalImage = document.getElementById('original-image');
const predictedMask = document.getElementById('predicted-mask');
const statTime = document.getElementById('stat-time');
const statOrigRes = document.getElementById('stat-orig-res');
const statConf = document.getElementById('stat-conf');

// Event Listeners for Drag and Drop
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    // Show preview immediately
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImage.src = e.target.result;
        uploadImage(file);
    };
    reader.readAsDataURL(file);
}

async function uploadImage(file) {
    // UI State: Loading
    dropZone.style.display = 'none';
    results.style.display = 'none';
    loading.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Prediction failed');
        }

        const data = await response.json();
        
        // Update UI
        predictedMask.src = `data:image/png;base64,${data.mask_b64}`;
        statTime.textContent = `${data.inference_time_ms} ms`;
        statOrigRes.textContent = data.original_resolution;
        statConf.textContent = `${(data.max_confidence * 100).toFixed(1)}%`;

        // Show Results
        loading.style.display = 'none';
        results.style.display = 'block';
        dropZone.style.display = 'block';
        
    } catch (error) {
        alert(`Error: ${error.message}`);
        loading.style.display = 'none';
        dropZone.style.display = 'block';
    }
}
