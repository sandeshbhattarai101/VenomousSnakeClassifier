/* Snake Vision — frontend logic */
'use strict';

const API_URL = '/api/predict';

// ── DOM refs ────────────────────────────────────────────────────────────────
const uploadZone   = document.getElementById('uploadZone');
const previewZone  = document.getElementById('previewZone');
const resultZone   = document.getElementById('resultZone');
const errorZone    = document.getElementById('errorZone');
const previewImg   = document.getElementById('previewImg');
const fileInput    = document.getElementById('fileInput');
const cameraInput  = document.getElementById('cameraInput');
const clearBtn     = document.getElementById('clearBtn');
const analyseBtn   = document.getElementById('analyseBtn');
const retryBtn     = document.getElementById('retryBtn');
const errorRetry   = document.getElementById('errorRetryBtn');
const spinner      = document.getElementById('spinner');

// Result elements
const resultBadge  = document.getElementById('resultBadge');
const resultIcon   = document.getElementById('resultIcon');
const resultLabel  = document.getElementById('resultLabel');
const confBar      = document.getElementById('confBar');
const confPct      = document.getElementById('confPct');
const vPct         = document.getElementById('vPct');
const sPct         = document.getElementById('sPct');
const warningBox   = document.getElementById('warningBox');
const warnIcon     = document.getElementById('warnIcon');
const warningText  = document.getElementById('warningText');
const inferenceRow = document.getElementById('inferenceRow');
const errorText    = document.getElementById('errorText');

let currentFile = null;

// ── State helpers ────────────────────────────────────────────────────────────
function showOnly(...zones) {
  [uploadZone, previewZone, resultZone, errorZone].forEach(z => {
    z.classList.toggle('hidden', !zones.includes(z));
  });
}

function resetToUpload() {
  currentFile = null;
  previewImg.src = '';
  fileInput.value = '';
  cameraInput.value = '';
  showOnly(uploadZone);
}

// ── File handling ────────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showError('Please select a valid image file (JPG, PNG, WEBP).');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showError('Image is too large. Please use an image under 20 MB.');
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    showOnly(previewZone);
  };
  reader.readAsDataURL(file);
}

fileInput.addEventListener('change',  () => handleFile(fileInput.files[0]));
cameraInput.addEventListener('change', () => handleFile(cameraInput.files[0]));
clearBtn.addEventListener('click', resetToUpload);

// ── Drag-and-drop ────────────────────────────────────────────────────────────
uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
['dragleave', 'dragend'].forEach(evt =>
  uploadZone.addEventListener(evt, () => uploadZone.classList.remove('drag-over'))
);
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  handleFile(e.dataTransfer.files[0]);
});

// Paste from clipboard (desktop convenience)
document.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      handleFile(item.getAsFile());
      break;
    }
  }
});

// ── Prediction ───────────────────────────────────────────────────────────────
analyseBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  // Loading state
  analyseBtn.disabled = true;
  analyseBtn.querySelector('.btn-text').textContent = 'Analysing…';
  spinner.classList.remove('hidden');

  try {
    const form = new FormData();
    form.append('file', currentFile);

    const res = await fetch(API_URL, { method: 'POST', body: form });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    showResult(data);

  } catch (err) {
    if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
      showError('Cannot reach the server. Make sure app.py is running on port 8000.');
    } else {
      showError(err.message || 'Unknown error. Please try again.');
    }
  } finally {
    analyseBtn.disabled = false;
    analyseBtn.querySelector('.btn-text').textContent = 'Analyse Snake';
    spinner.classList.add('hidden');
  }
});

// ── Show result ───────────────────────────────────────────────────────────────
function showResult(data) {
  const isVenomous = data.label === 'venomous';

  // Badge
  resultBadge.className = 'result-badge ' + (isVenomous ? 'venomous-result' : 'safe-result');
  resultIcon.textContent  = isVenomous ? '☠️' : '✅';
  resultLabel.textContent = data.display;

  // Confidence bar (animate after render)
  confBar.className = 'confidence-bar-fill ' + (isVenomous ? 'venomous-fill' : 'safe-fill');
  confPct.textContent = data.confidence + '%';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { confBar.style.width = data.confidence + '%'; });
  });

  // Probabilities
  vPct.textContent = data.venomous_probability + '%';
  sPct.textContent = data.safe_probability + '%';

  // Warning
  warningBox.className = 'warning-box ' + (isVenomous ? 'danger' : 'safe');
  warnIcon.textContent  = isVenomous ? '🚨' : '🛡️';
  warningText.textContent = data.warning;

  // Inference time
  if (data.inference_ms) {
    inferenceRow.textContent = `Inference time: ${data.inference_ms} ms`;
  }

  showOnly(resultZone);
}

// ── Show error ────────────────────────────────────────────────────────────────
function showError(msg) {
  errorText.textContent = msg;
  showOnly(errorZone);
}

retryBtn.addEventListener('click', resetToUpload);
errorRetry.addEventListener('click', resetToUpload);
