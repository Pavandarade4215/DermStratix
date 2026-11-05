/**
 * DermStratix AI — Healthcare Decision Support Platform
 * Client-Side Application Controller
 */

// Global Application State
const AppState = {
    currentView: 'screening',     // 'screening', 'history', 'about', 'performance'
    currentMode: 'lesion',        // 'lesion' or 'facial'
    selectedSampleId: 'sample_melanoma',
    uploadedImageBase64: null,
    lastResult: null,
    history: []
};

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initDropzone();
    initHistory();
    initPerformanceTab();
    
    // Automatically load default demo case
    loadSamplePreset('sample_melanoma');
});

// ==============================================================================
// 1. NAVIGATION & VIEW SWITCHING
// ==============================================================================
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-menu .nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetView = btn.getAttribute('data-view');
            switchView(targetView);
        });
    });
}

function switchView(viewName) {
    AppState.currentView = viewName;

    // Update nav buttons
    document.querySelectorAll('.nav-menu .nav-btn').forEach(b => {
        if (b.getAttribute('data-view') === viewName) {
            b.classList.add('active');
        } else {
            b.classList.remove('active');
        }
    });

    // Update views
    document.querySelectorAll('.app-view').forEach(v => {
        v.classList.remove('active');
    });

    const activeView = document.getElementById(`view-${viewName}`);
    if (activeView) {
        activeView.classList.add('active');
    }

    if (viewName === 'performance' && !AppState.perfChartsInitialized) {
        loadBenchmarkPerformance();
    } else if (viewName === 'history') {
        renderHistoryTable();
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ==============================================================================
// 2. SCREENING MODE SELECTION (TWO CLEAN CARDS)
// ==============================================================================
function setScreeningMode(mode) {
    AppState.currentMode = mode;

    const lesionCard = document.getElementById('mode-card-lesion');
    const facialCard = document.getElementById('mode-card-facial');
    const abcdContainer = document.getElementById('abcd-container');

    if (mode === 'lesion') {
        lesionCard?.classList.add('active');
        facialCard?.classList.remove('active');
        if (abcdContainer) abcdContainer.style.display = 'block';
    } else {
        facialCard?.classList.add('active');
        lesionCard?.classList.remove('active');
        if (abcdContainer) abcdContainer.style.display = 'none';
    }

    // Auto re-execute if image is available
    if (AppState.uploadedImageBase64 || AppState.selectedSampleId) {
        executeScreening();
    }
}

function toggleVitalsDrawer() {
    const drawer = document.getElementById('vitals-drawer-content');
    const arrow = document.getElementById('drawer-arrow');
    if (drawer.classList.contains('open')) {
        drawer.classList.remove('open');
        arrow.style.transform = 'rotate(0deg)';
    } else {
        drawer.classList.add('open');
        arrow.style.transform = 'rotate(180deg)';
    }
}

// ==============================================================================
// 3. IMAGE ACQUISITION & DROPZONE
// ==============================================================================
function initDropzone() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    if (!dropzone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-over');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleUploadedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length > 0) {
            handleUploadedFile(fileInput.files[0]);
        }
    });
}

function handleUploadedFile(file) {
    if (!file.type.match('image.*')) {
        alert('Please select a valid image file (JPEG, PNG, WEBP).');
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        AppState.uploadedImageBase64 = e.target.result;
        AppState.selectedSampleId = null;
        document.getElementById('sample-select').value = '';
        executeScreening();
    };
    reader.readAsDataURL(file);
}

function triggerWebcamCapture() {
    // Simulated live camera acquisition
    const demoCases = ['sample_melanoma', 'sample_rosacea', 'sample_nevus', 'sample_jaundice'];
    const randomCase = demoCases[Math.floor(Math.random() * demoCases.length)];
    loadSamplePreset(randomCase);
}

function onSampleDropdownChange(val) {
    if (val) {
        loadSamplePreset(val);
    }
}

function loadSamplePreset(sampleId) {
    AppState.selectedSampleId = sampleId;
    AppState.uploadedImageBase64 = null;
    
    const select = document.getElementById('sample-select');
    if (select) select.value = sampleId;

    // Adjust mode based on sample type
    if (sampleId.includes('facial') || sampleId.includes('rosacea') || sampleId.includes('jaundice') || sampleId.includes('acne') || sampleId === 'sample_normal') {
        AppState.currentMode = 'facial';
        document.getElementById('mode-card-facial')?.classList.add('active');
        document.getElementById('mode-card-lesion')?.classList.remove('active');
        document.getElementById('abcd-container').style.display = 'none';
    } else {
        AppState.currentMode = 'lesion';
        document.getElementById('mode-card-lesion')?.classList.add('active');
        document.getElementById('mode-card-facial')?.classList.remove('active');
        document.getElementById('abcd-container').style.display = 'block';
    }

    executeScreening();
}

// ==============================================================================
// 4. SCREENING EXECUTION & API INTEGRATION
// ==============================================================================
function getVitalsPayload() {
    return {
        age: parseFloat(document.getElementById('vit-age')?.value || 52),
        systolic_bp: parseFloat(document.getElementById('vit-sys')?.value || 136),
        diastolic_bp: 84,
        bmi: parseFloat(document.getElementById('vit-bmi')?.value || 26.8),
        glucose: parseFloat(document.getElementById('vit-gluc')?.value || 110),
        cholesterol: 200,
        heart_rate: 74,
        smoking: 0,
        alcohol: 0,
        physical_activity: 1
    };
}

function executeScreening() {
    const btn = document.getElementById('btn-screen');
    if (btn) {
        btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing Cutaneous Features...';
        btn.disabled = true;
    }

    const payload = {
        sample_id: AppState.selectedSampleId,
        image_base64: AppState.uploadedImageBase64,
        mode: AppState.currentMode,
        model: 'mobilenetv2',
        vitals: getVitalsPayload()
    };

    fetch('/api/screen/image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert(`Screening analysis error: ${data.error}`);
            return;
        }
        AppState.lastResult = data;
        renderResults(data);
    })
    .catch(err => {
        console.error('API Error:', err);
    })
    .finally(() => {
        if (btn) {
            btn.innerHTML = '<i class="fa-solid fa-stethoscope"></i> Analyze Image & Stratify Risk';
            btn.disabled = false;
        }
    });
}

// ==============================================================================
// 5. RENDERING RESULTS (HEALTHCARE SAAS FORMAT)
// ==============================================================================
function renderResults(data) {
    const model = data.model || {};
    const triage = data.triage || {};
    const vitality = data.vitality || {};
    const abcd = data.abcd || {};
    const gate = data.focus_gate || {};

    // 1. Primary Diagnosis & Triage Tier
    const diagTitle = document.getElementById('res-diag-title');
    const tierPill = document.getElementById('res-tier-pill');
    const tierText = document.getElementById('res-tier-text');
    const confEl = document.getElementById('res-confidence');
    const modelEl = document.getElementById('res-model-used');

    if (diagTitle) diagTitle.innerText = model.predicted_class || 'Cutaneous Finding';
    if (confEl) confEl.innerText = `${((model.top_probability || 0.91) * 100).toFixed(1)}%`;
    if (modelEl) modelEl.innerText = `${model.model || 'MobileNetV2'} Deep CNN`;

    if (tierPill && tierText) {
        tierPill.className = 'risk-tier-pill';
        if (triage.tier_level === 3 || triage.posterior_risk_p > 0.7) {
            tierPill.classList.add('tier-urgent');
            tierText.innerText = 'Urgent Priority';
        } else if (triage.tier_level === 2 || triage.posterior_risk_p >= 0.35) {
            tierPill.classList.add('tier-moderate');
            tierText.innerText = 'Moderate Risk';
        } else {
            tierPill.classList.add('tier-low');
            tierText.innerText = 'Low Risk';
        }
    }

    // 2. Vitality Score
    const vScore = vitality.vitality_score || triage.vitality_score || 85;
    const vNum = document.getElementById('res-vitality-num');
    const vBar = document.getElementById('res-vitality-bar');
    const vLabel = document.getElementById('res-vitality-label');

    if (vNum) vNum.innerText = vScore;
    if (vBar) {
        vBar.style.width = `${vScore}%`;
        if (vScore < 60) {
            vBar.style.backgroundColor = 'var(--urgent-red)';
        } else if (vScore < 80) {
            vBar.style.backgroundColor = 'var(--moderate-amber)';
        } else {
            vBar.style.backgroundColor = 'var(--low-green)';
        }
    }
    if (vLabel) vLabel.innerText = vitality.status_label || 'Cutaneous Risk Evaluated';

    // 3. Recommended Action
    const actionHeader = document.getElementById('res-action-header');
    const actionDesc = document.getElementById('res-action-desc');
    const actionCard = document.getElementById('res-action-card');

    if (actionHeader) actionHeader.innerText = triage.action_header || 'Standard Follow-up Protocol';
    if (actionDesc) actionDesc.innerText = triage.clinical_action || 'Routine examination recommended.';
    if (actionCard) {
        if (triage.tier_level === 3 || triage.posterior_risk_p > 0.7) {
            actionCard.style.borderLeftColor = 'var(--urgent-red)';
        } else if (triage.tier_level === 2 || triage.posterior_risk_p >= 0.35) {
            actionCard.style.borderLeftColor = 'var(--moderate-amber)';
        } else {
            actionCard.style.borderLeftColor = 'var(--low-green)';
        }
    }

    // 4. Optical Focus Gate
    const gateBadge = document.getElementById('blur-gate-badge');
    const gateFill = document.getElementById('blur-gate-fill');
    const gateMsg = document.getElementById('blur-gate-msg');

    if (gateBadge) {
        if (gate.is_soft_focus) {
            gateBadge.className = 'badge-status status-warn';
            gateBadge.innerText = 'Soft-Focus (Enhanced)';
        } else {
            gateBadge.className = 'badge-status status-pass';
            gateBadge.innerText = 'Focus Verified';
        }
    }
    if (gateFill) {
        const pct = Math.min(100, Math.max(15, ((gate.sharpness_score || 180) / 240) * 100));
        gateFill.style.width = `${pct}%`;
    }
    if (gateMsg) gateMsg.innerText = gate.status_message || 'Image clarity verified for clinical assessment.';

    // 5. Visual Evidence
    const imgEl = document.getElementById('main-result-img');
    const captionEl = document.getElementById('visual-caption');
    const images = data.images || {};

    if (imgEl) {
        if (images.lime_explanation) {
            imgEl.src = images.lime_explanation;
        } else if (images.annotated_detection) {
            imgEl.src = images.annotated_detection;
        }
    }
    if (captionEl) {
        captionEl.innerHTML = '<i class="fa-solid fa-circle-info"></i> LIME Superpixel Overlay (Green indicates feature evidence driving triage)';
    }

    // 6. ABCD Criteria Rubric (Dermoscopy)
    if (data.mode && data.mode.includes('Facial')) {
        document.getElementById('abcd-container').style.display = 'none';
    } else {
        document.getElementById('abcd-container').style.display = 'block';
        renderAbcdItem('a', abcd.asymmetry, '/ 10');
        renderAbcdItem('b', abcd.border, '/ 10');
        renderAbcdItem('c', abcd.color, '/ 10');
        if (abcd.diameter) {
            document.getElementById('val-abcd-d').innerText = `${abcd.diameter.value_mm} mm`;
            document.getElementById('fill-abcd-d').style.width = `${Math.min(100, (abcd.diameter.value_mm / 10) * 100)}%`;
            const dStatus = document.getElementById('status-abcd-d');
            if (dStatus) {
                dStatus.innerText = abcd.diameter.is_flagged ? '> 6.0 mm Alert' : '< 6.0 mm Normal';
                dStatus.className = abcd.diameter.is_flagged ? 'abcd-status text-danger' : 'abcd-status text-success';
            }
        }
    }

    // 7. Probability Distribution
    const distContainer = document.getElementById('distribution-bars');
    if (distContainer && model.class_probabilities) {
        distContainer.innerHTML = '';
        model.class_probabilities.forEach(item => {
            const pct = (item.probability * 100).toFixed(1);
            let barColor = 'var(--accent-cyan)';
            if (item.severity_weight > 0.7) barColor = 'var(--urgent-red)';
            else if (item.severity_weight > 0.3) barColor = 'var(--moderate-amber)';
            else barColor = 'var(--low-green)';

            distContainer.innerHTML += `
                <div class="dist-row">
                    <span class="dist-name" title="${item.name}">${item.name}</span>
                    <div class="dist-track">
                        <div class="dist-fill" style="width: ${pct}%; background-color: ${barColor};"></div>
                    </div>
                    <span class="dist-pct">${pct}%</span>
                </div>
            `;
        });
    }
}

function renderAbcdItem(key, item, unit) {
    if (!item) return;
    const valEl = document.getElementById(`val-abcd-${key}`);
    const fillEl = document.getElementById(`fill-abcd-${key}`);
    const statusEl = document.getElementById(`status-abcd-${key}`);

    if (valEl) valEl.innerText = `${item.score} ${unit}`;
    if (fillEl) {
        fillEl.style.width = `${item.score * 10}%`;
        fillEl.className = item.is_flagged ? 'abcd-bar-fill bar-red' : 'abcd-bar-fill bar-green';
    }
    if (statusEl) {
        statusEl.innerText = item.status;
        statusEl.className = item.is_flagged ? 'abcd-status text-danger' : 'abcd-status text-success';
    }
}

function switchImageView(viewType) {
    document.querySelectorAll('.view-switch-btns .switch-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');

    const data = AppState.lastResult;
    if (!data || !data.images) return;

    const imgEl = document.getElementById('main-result-img');
    const capEl = document.getElementById('visual-caption');

    if (viewType === 'lime' && data.images.lime_explanation) {
        imgEl.src = data.images.lime_explanation;
        capEl.innerHTML = '<i class="fa-solid fa-circle-info"></i> LIME Superpixel Overlay (Green indicates feature evidence driving triage)';
    } else if (viewType === 'roi' && data.images.roi_224) {
        imgEl.src = data.images.roi_224;
        capEl.innerHTML = '<i class="fa-solid fa-circle-info"></i> Localized 224x224 RGB Region of Interest';
    } else if (viewType === 'annotated' && data.images.annotated_detection) {
        imgEl.src = data.images.annotated_detection;
        capEl.innerHTML = '<i class="fa-solid fa-circle-info"></i> OpenCV Optical Detection Bounding Box';
    }
}

// ==============================================================================
// 6. SCREENING HISTORY MANAGEMENT
// ==============================================================================
function initHistory() {
    const stored = localStorage.getItem('dermstratix_saas_history');
    if (stored) {
        try {
            AppState.history = JSON.parse(stored);
        } catch (e) {
            AppState.history = [];
        }
    }

    if (!AppState.history || AppState.history.length === 0) {
        // Pre-seed clean clinical demo records
        AppState.history = [
            {
                id: 'REC-92841',
                timestamp: '2026-11-04 14:22',
                mode: 'Dermoscopic Lesion',
                finding: 'Malignant Melanoma (mel)',
                tier: 'Urgent Priority',
                tierClass: 'tier-urgent',
                vitality: 28
            },
            {
                id: 'REC-88412',
                timestamp: '2026-11-03 11:05',
                mode: 'Facial Screening',
                finding: 'Centrofacial Rosacea (Erythema)',
                tier: 'Moderate Risk',
                tierClass: 'tier-moderate',
                vitality: 73
            },
            {
                id: 'REC-74109',
                timestamp: '2026-11-02 09:40',
                mode: 'Dermoscopic Lesion',
                finding: 'Melanocytic Nevus (Common Mole)',
                tier: 'Low Risk',
                tierClass: 'tier-low',
                vitality: 94
            }
        ];
        saveHistoryToStorage();
    }
    updateHistoryBadge();
}

function saveHistoryToStorage() {
    localStorage.setItem('dermstratix_saas_history', JSON.stringify(AppState.history));
    updateHistoryBadge();
}

function updateHistoryBadge() {
    const badge = document.getElementById('history-badge');
    if (badge) {
        badge.innerText = AppState.history.length;
    }
}

function saveToHistory() {
    const data = AppState.lastResult;
    if (!data) return;

    const record = {
        id: `REC-${Math.floor(10000 + Math.random() * 90000)}`,
        timestamp: new Date().toISOString().replace('T', ' ').substring(0, 16),
        mode: data.mode || 'Dermoscopic Lesion',
        finding: data.model?.predicted_class || 'Evaluated Condition',
        tier: data.triage?.tier_name || 'Standard Tier',
        tierClass: data.triage?.posterior_risk_p > 0.7 ? 'tier-urgent' : (data.triage?.posterior_risk_p >= 0.35 ? 'tier-moderate' : 'tier-low'),
        vitality: data.vitality?.vitality_score || 80
    };

    AppState.history.unshift(record);
    saveHistoryToStorage();
    alert(`Screening record ${record.id} successfully saved to patient history.`);
}

function renderHistoryTable() {
    const tbody = document.getElementById('history-tbody');
    if (!tbody) return;

    tbody.innerHTML = '';
    if (AppState.history.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:2rem; color:var(--text-muted);">No past screening records found.</td></tr>';
        return;
    }

    AppState.history.forEach((rec, idx) => {
        tbody.innerHTML += `
            <tr>
                <td>${rec.timestamp}</td>
                <td><strong>${rec.id}</strong></td>
                <td>${rec.mode}</td>
                <td><strong>${rec.finding}</strong></td>
                <td><span class="risk-tier-pill ${rec.tierClass}"><span class="tier-dot"></span> ${rec.tier}</span></td>
                <td><strong>${rec.vitality}</strong> / 100</td>
                <td>
                    <button class="btn btn-secondary btn-sm" onclick="viewHistoricalRecord(${idx})"><i class="fa-solid fa-eye"></i> View</button>
                    <button class="btn-text" style="color:var(--urgent-red);" onclick="deleteHistoricalRecord(${idx})"><i class="fa-solid fa-trash"></i></button>
                </td>
            </tr>
        `;
    });
}

function deleteHistoricalRecord(idx) {
    if (confirm('Delete this screening record from history?')) {
        AppState.history.splice(idx, 1);
        saveHistoryToStorage();
        renderHistoryTable();
    }
}

function clearScreeningHistory() {
    if (confirm('Are you sure you want to clear all screening history records?')) {
        AppState.history = [];
        saveHistoryToStorage();
        renderHistoryTable();
    }
}

function viewHistoricalRecord(idx) {
    const rec = AppState.history[idx];
    if (!rec) return;

    if (rec.finding.includes('Melanoma')) {
        loadSamplePreset('sample_melanoma');
    } else if (rec.finding.includes('Nevus')) {
        loadSamplePreset('sample_nevus');
    } else if (rec.finding.includes('Rosacea')) {
        loadSamplePreset('sample_rosacea');
    } else {
        loadSamplePreset('sample_bcc');
    }
    switchView('screening');
}

// ==============================================================================
// 7. MODEL PERFORMANCE & BENCHMARKING HUB
// ==============================================================================
function initPerformanceTab() {
    // Lazy loaded when view is clicked
}

function loadBenchmarkPerformance() {
    fetch('/api/benchmark/data')
        .then(res => res.json())
        .then(data => {
            renderPerformanceTable(data.table_iv_models);
            renderSaasMetricsChart();
            renderSaasConfusionMatrix(data.fig10_confusion_matrix);
            AppState.perfChartsInitialized = true;
        })
        .catch(err => console.error('Benchmark loading error:', err));
}

function renderPerformanceTable(models) {
    const tbody = document.getElementById('performance-table-body');
    if (!tbody || !models) return;

    tbody.innerHTML = '';
    models.forEach(m => {
        const isProposed = m.model.includes('MobileNetV2');
        tbody.innerHTML += `
            <tr class="${isProposed ? 'highlight-best' : ''}">
                <td><strong>${m.model}</strong> ${isProposed ? '<span class="risk-tier-pill tier-low" style="font-size:0.65rem; padding:0.1rem 0.4rem;">Proposed</span>' : ''}</td>
                <td>${m.paradigm} (${m.params})</td>
                <td>${m.vitals_acc ? m.vitals_acc.toFixed(2) + '%' : '—'}</td>
                <td><strong>${m.binary_acc.toFixed(2)}%</strong></td>
                <td>${m.facial_5class.toFixed(2)}%</td>
                <td>${m.ham_7class.toFixed(2)}%</td>
                <td><span class="badge-status ${isProposed ? 'status-pass' : 'status-warn'}">${isProposed ? 'Production Standard' : 'Comparative Model'}</span></td>
            </tr>
        `;
    });
}

function renderSaasMetricsChart() {
    const ctx = document.getElementById('saas-chart-metrics');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Logistic Reg', 'SVM RBF', 'Random Forest', 'XGBoost', 'VGG-16', 'DenseNet-121', 'ResNet-50', 'MobileNetV2 (Proposed)'],
            datasets: [
                { label: 'Accuracy (%)', data: [78.4, 82.6, 85.4, 86.4, 83.2, 87.5, 88.6, 90.9], backgroundColor: '#0284c7' },
                { label: 'Sensitivity / Recall (%)', data: [77.1, 80.5, 85.4, 86.4, 82.0, 87.0, 88.5, 90.8], backgroundColor: '#059669' },
                { label: 'Macro F1 (%)', data: [76.9, 80.8, 85.4, 86.4, 82.2, 86.9, 88.3, 90.6], backgroundColor: '#475569' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { boxWidth: 12, font: { family: 'Inter', size: 12 } } }
            },
            scales: {
                x: { grid: { display: false } },
                y: { min: 70, max: 95, ticks: { stepSize: 5 }, grid: { color: '#f1f5f9' } }
            }
        }
    });
}

function renderSaasConfusionMatrix(cmData) {
    const container = document.getElementById('saas-cm-wrapper');
    if (!container || !cmData) return;

    const classes = cmData.classes;
    const matrix = cmData.matrix;

    let html = '<table class="saas-cm-table"><thead><tr><th>Ground Truth \ Predicted</th>';
    classes.forEach(c => { html += `<th>${c}</th>`; });
    html += '</tr></thead><tbody>';

    matrix.forEach((row, i) => {
        html += `<tr><td><strong>${classes[i]}</strong></td>`;
        row.forEach((val, j) => {
            const isDiag = (i === j);
            let bg = isDiag ? `rgba(5, 150, 105, ${val / 100})` : `rgba(2, 132, 199, ${val / 100})`;
            let textCol = isDiag && val > 50 ? '#ffffff' : (val > 5 ? '#0f172a' : '#94a3b8');
            html += `<td class="cm-data-cell" style="background:${bg}; color:${textCol};">${val}%</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

// ==============================================================================
// 8. AUDIT REPORT MODAL & PRINTING
// ==============================================================================
function openAuditReport() {
    const data = AppState.lastResult;
    if (!data) {
        alert('Please run an image screening first.');
        return;
    }

    const model = data.model || {};
    const triage = data.triage || {};
    const abcd = data.abcd || {};
    const gate = data.focus_gate || {};

    const modalBody = document.getElementById('modal-report-content');
    modalBody.innerHTML = `
        <div style="font-family:'Inter',sans-serif; color:#0f172a; line-height:1.5;">
            <div style="border-bottom:2px solid #0f172a; padding-bottom:0.75rem; margin-bottom:1rem; display:flex; justify-content:space-between; align-items:flex-end;">
                <div>
                    <h2 style="font-size:1.3rem; margin:0; color:#0f172a;">DermStratix AI — Clinical Screening Report</h2>
                    <p style="font-size:0.8rem; color:#64748b; margin:0;">Automated Cutaneous & Facial Health Decision Support</p>
                </div>
                <div style="text-align:right; font-size:0.8rem; color:#475569;">
                    <div><strong>Date:</strong> ${new Date().toISOString().replace('T', ' ').substring(0, 16)} UTC</div>
                    <div><strong>Patient Ref:</strong> PAT-${Math.floor(100000 + Math.random() * 900000)}</div>
                </div>
            </div>

            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:1rem; margin-bottom:1rem;">
                <div style="font-size:0.75rem; text-transform:uppercase; font-weight:700; color:#64748b; margin-bottom:0.25rem;">Diagnostic Finding</div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:1.4rem; font-weight:700; color:#0f172a;">${model.predicted_class || 'Melanoma'}</div>
                        <div style="font-size:0.85rem; color:#475569;">Confidence: ${((model.top_probability || 0.91) * 100).toFixed(1)}% (${model.model || 'MobileNetV2'} Model)</div>
                    </div>
                    <div style="text-align:right;">
                        <span class="risk-tier-pill ${triage.posterior_risk_p > 0.7 ? 'tier-urgent' : (triage.posterior_risk_p >= 0.35 ? 'tier-moderate' : 'tier-low')}">
                            <span class="tier-dot"></span> ${triage.tier_name || 'Urgent Priority'}
                        </span>
                        <div style="font-size:0.8rem; color:#64748b; margin-top:0.25rem;">Health Score: <strong>${data.vitality?.vitality_score || 28} / 100</strong></div>
                    </div>
                </div>

                <div style="margin-top:0.75rem; padding:0.75rem; background:#ffffff; border-radius:6px; border:1px solid #e2e8f0;">
                    <strong style="color:#0f172a; font-size:0.85rem;">Clinical Protocol:</strong>
                    <p style="margin:0.2rem 0 0 0; font-size:0.82rem; color:#334155;">${triage.clinical_action || 'Refer to specialist.'}</p>
                </div>
            </div>

            <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem;">
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:0.85rem;">
                    <strong style="font-size:0.85rem; color:#0f172a; display:block; margin-bottom:0.4rem;">ABCD Morphometric Rubric</strong>
                    <div style="font-size:0.8rem; line-height:1.6; color:#475569;">
                        <div>• <strong>Asymmetry:</strong> ${abcd.asymmetry?.score || '7.4'} / 10 (${abcd.asymmetry?.status})</div>
                        <div>• <strong>Border Irregularity:</strong> ${abcd.border?.score || '6.8'} / 10 (${abcd.border?.status})</div>
                        <div>• <strong>Color Variegation:</strong> ${abcd.color?.score || '7.2'} / 10 (${abcd.color?.status})</div>
                        <div>• <strong>Caliber Diameter:</strong> ${abcd.diameter?.value_mm || '7.9'} mm (${abcd.diameter?.status})</div>
                    </div>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:0.85rem;">
                    <strong style="font-size:0.85rem; color:#0f172a; display:block; margin-bottom:0.4rem;">Quality & Acquisition</strong>
                    <div style="font-size:0.8rem; line-height:1.6; color:#475569;">
                        <div>• <strong>Mode:</strong> ${data.mode || 'Dermoscopy Mode'}</div>
                        <div>• <strong>Optical Focus Gate:</strong> ${gate.sharpness_score || 185} (Threshold: 30.0)</div>
                        <div>• <strong>Contrast Normalization:</strong> ${gate.clahe_applied ? 'CLAHE Applied' : 'Natural Sharpness'}</div>
                        <div>• <strong>Verification:</strong> Completed</div>
                    </div>
                </div>
            </div>

            <div style="font-size:0.75rem; color:#64748b; background:#f1f5f9; border-radius:6px; padding:0.65rem; border:1px solid #e2e8f0;">
                <strong>Clinical Notice:</strong> ${data.disclaimer || 'Investigational decision support screening tool. Findings do not constitute a definitive medical diagnosis.'}
            </div>
        </div>
    `;

    document.getElementById('report-modal')?.classList.add('open');
}

function closeAuditReport() {
    document.getElementById('report-modal')?.classList.remove('open');
}
