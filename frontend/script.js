// script.js

const API = 'http://127.0.0.1:5000/api';

// Stores current patient data
let currentPatient = null;
let featureRanges  = null;

// Friendly display names for features
const FEATURE_LABELS = {
  age:      'Age',
  sex:      'Sex (0=F, 1=M)',
  cp:       'Chest Pain Type',
  trestbps: 'Resting BP',
  chol:     'Cholesterol',
  fbs:      'Fasting Blood Sugar',
  restecg:  'Resting ECG',
  thalach:  'Max Heart Rate',
  exang:    'Exercise Angina',
  oldpeak:  'ST Depression',
  slope:    'Slope',
  ca:       'Major Vessels',
  thal:     'Thal'
};

const FEATURE_DESCRIPTIONS = {
  age:      'Patient age in years (dataset range: 29–77)',
  sex:      '0 = Female, 1 = Male',
  cp:       '0=No pain, 1=Typical angina, 2=Atypical angina, 3=Non-anginal',
  trestbps: 'Resting blood pressure in mmHg (normal: 90–120)',
  chol:     'Serum cholesterol in mg/dl (normal: <200, high: >240)',
  fbs:      'Fasting blood sugar >120 mg/dl: 0=No, 1=Yes',
  restecg:  '0=Normal, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy',
  thalach:  'Maximum heart rate achieved during exercise test',
  exang:    'Exercise-induced angina: 0=No, 1=Yes',
  oldpeak:  'ST depression induced by exercise relative to rest',
  slope:    '0=Upsloping, 1=Flat, 2=Downsloping (peak exercise ST segment)',
  ca:       'Number of major vessels colored by fluoroscopy (0–3)',
  thal:     '0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Unknown'
};


// ── Step 1: Load a random patient ────────────────────────────────────
async function loadSample() {
  const btn = document.getElementById('btn-sample');
  btn.disabled = true;
  btn.textContent = '⏳ Loading...';

  // Also fetch feature ranges if not loaded yet
  if (!featureRanges) {
    const r = await fetch(`${API}/features`);
    featureRanges = await r.json();
  }

  const res  = await fetch(`${API}/sample`);
  const json = await res.json();
  currentPatient = json.data;

  // Build the patient grid
  const grid = document.getElementById('patient-grid');
  grid.innerHTML = '';

  for (const [key, val] of Object.entries(currentPatient)) {
    const item = document.createElement('div');
    item.className = 'grid-item';
    item.innerHTML = `
      <div class="label">${FEATURE_LABELS[key] || key}</div>
      <div class="value">${val}</div>
    `;
    grid.appendChild(item);
  }

  // Show patient section, hide old results
  show('section-patient');
  hide('section-prediction');
  hide('section-whatif');
  hide('section-comparison');

  btn.disabled = false;
  btn.textContent = '🔀 Load Random Patient';
}


// ── Step 2: Predict outcome ───────────────────────────────────────────
async function predictOutcome() {
  const btn = document.getElementById('btn-predict');
  btn.disabled = true;
  btn.textContent = '⏳ Predicting...';

  const res  = await fetch(`${API}/predict`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ data: currentPatient })
  });
  const result = await res.json();

  // Show prediction box
  const box = document.getElementById('prediction-box');
  const isDisease = result.prediction === 1;

  box.className  = `result-box ${isDisease ? 'disease' : 'no-disease'}`;
  box.innerHTML  = `
    ${isDisease ? '❤️‍🩹' : '✅'} <strong>${result.label}</strong>
    <br/>
    <span style="font-size:0.85rem; font-weight:400; opacity:0.8;">
      Confidence: ${result.confidence}%
    </span>
  `;

  show('section-prediction');
  show('section-whatif');

  btn.disabled = false;
  btn.textContent = '🔍 Predict Outcome';
}


// ── Step 3: Feature dropdown changed ─────────────────────────────────
function onFeatureChange() {
  const feature = document.getElementById('feature-select').value;
  if (!feature) return;

  const control = document.getElementById('value-control');
  const label   = document.getElementById('value-label');
  const input   = document.getElementById('new-value');
  const current = document.getElementById('current-val');

  const currentVal = currentPatient[feature];
  const range      = featureRanges[feature];

  label.textContent = `New value for "${FEATURE_LABELS[feature]}":`;
  input.min         = range.min;
  input.max         = range.max;
  input.value       = currentVal;

  current.textContent =
    `Current: ${currentVal} | Range: ${range.min} – ${range.max}`;

  // Show medical description
  let desc = document.getElementById('feature-desc');
  if (!desc) {
    desc = document.createElement('div');
    desc.id = 'feature-desc';
    desc.style.cssText = 'font-size:0.78rem;color:#4b6a99;margin-top:4px;';
    current.parentNode.appendChild(desc);
  }
  desc.textContent = FEATURE_DESCRIPTIONS[feature] || '';

  control.classList.remove('hidden');
  document.getElementById('btn-cf').classList.remove('hidden');
}


// ── Step 4: Generate counterfactual ──────────────────────────────────
async function generateCounterfactual() {
  const feature  = document.getElementById('feature-select').value;
  const newValue = parseFloat(document.getElementById('new-value').value);
  const btn      = document.getElementById('btn-cf');

  if (!feature) { alert('Please select a feature first.'); return; }

  btn.disabled    = true;
  btn.textContent = '⏳ Generating...';

  const res    = await fetch(`${API}/counterfactual`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      data:            currentPatient,
      changed_feature: feature,
      new_value:       newValue
    })
  });
  const result = await res.json();

  // ── Original box ──────────────────────────────────────────────────
  const orig    = result.original;
  const origBox = document.getElementById('comp-original');
  const isOrigDisease = orig.prediction === 1;
  origBox.innerHTML = `
    <h3>Original Patient</h3>
    <div class="pred-label ${isOrigDisease ? 'disease' : 'no-disease'}">
      ${isOrigDisease ? '❤️‍🩹' : '✅'} ${orig.label}
    </div>
    <div class="conf">Confidence: ${orig.confidence}%</div>
    <div class="conf-bar-wrap">
      <div class="conf-bar ${isOrigDisease ? 'disease' : 'no-disease'}"
           style="width:${orig.confidence}%"></div>
    </div>
  `;

  // ── Counterfactual box ────────────────────────────────────────────
  const cf    = result.counterfactual;
  const cfBox = document.getElementById('comp-cf');
  const isCfDisease = cf.prediction === 1;
  cfBox.innerHTML = `
    <h3>After Changing ${FEATURE_LABELS[feature]}</h3>
    <div class="pred-label ${isCfDisease ? 'disease' : 'no-disease'}">
      ${isCfDisease ? '❤️‍🩹' : '✅'} ${cf.label}
    </div>
    <div class="conf">Confidence: ${cf.confidence}%</div>
    <div class="conf-bar-wrap">
      <div class="conf-bar ${isCfDisease ? 'disease' : 'no-disease'}"
           style="width:${cf.confidence}%"></div>
    </div>
  `;

  // ── Outcome message ───────────────────────────────────────────────
  const msg = document.getElementById('outcome-message');
  if (result.outcome_changed) {
    msg.className   = 'changed';
    msg.textContent = `✅ Outcome changed! Modifying "${FEATURE_LABELS[feature]}" to ${newValue} made a difference.`;
  } else {
    msg.className   = 'unchanged';
    msg.textContent = `⚠️ Outcome unchanged with this value. See suggestions below for what will work.`;
  }

  // ── Changes list ──────────────────────────────────────────────────
  const changesList = document.getElementById('changes-list');
  let html = '<strong>What changed:</strong><br/>';
  for (const [key, val] of Object.entries(result.changes)) {
    html += `${FEATURE_LABELS[key]}: ${val.from}
             → <span class="changed-val">${val.to}</span><br/>`;
  }

  // ── Smart suggestions ─────────────────────────────────────────────
  if (!result.outcome_changed && result.suggestions.length > 0) {
    html += `<br/><strong>💡 These changes WILL flip the outcome:</strong><br/>`;
    for (const s of result.suggestions) {
      html += `
        • Change <span class="changed-val">${FEATURE_LABELS[s.feature]}</span>
          from ${s.from_value} → <span class="changed-val">${s.change_to}</span>
          → Result: ${s.new_label} (${s.confidence}% confidence)
        <br/>
      `;
    }
  } else if (result.outcome_changed) {
    html += `<br/><span style="color:#4ade80">
             ✅ This single change was enough to flip the prediction!
             </span>`;
  }

  changesList.innerHTML = html;

  // Show low confidence warning
  const minConf = Math.min(orig.confidence, cf.confidence);
  if (minConf < 70) {
    const warn = document.createElement('div');
    warn.style.cssText = `
      background:#16100a; border:1px solid #ca8a0444;
      border-left:3px solid #ca8a04; color:#fbbf24;
      font-size:0.82rem; padding:10px 14px;
      border-radius:8px; margin-bottom:12px;
      line-height:1.6;
    `;
    warn.innerHTML = `⚠️ <strong>Low confidence prediction (${minConf}%)</strong>
      — This patient's profile is near the model's decision boundary.
      Results should be interpreted with caution.`;
    document.getElementById('section-comparison')
            .insertBefore(warn, document.getElementById('outcome-message'));
  }
  show('section-comparison');

  btn.disabled    = false;
  btn.textContent = '⚡ Generate Counterfactual';
}


// ── Helpers ───────────────────────────────────────────────────────────
function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }

// ── Toggle manual form visibility ─────────────────────────────────────
function toggleManualForm() {
  const form = document.getElementById('manual-form');
  const isHidden = form.style.display === 'none';
  form.style.display = isHidden ? 'block' : 'none';
}


// ── Read and validate manual inputs ──────────────────────────────────
function submitManual() {

  // Map of input id → feature name
  const fields = {
    'inp-age':      'age',
    'inp-sex':      'sex',
    'inp-cp':       'cp',
    'inp-trestbps': 'trestbps',
    'inp-chol':     'chol',
    'inp-fbs':      'fbs',
    'inp-restecg':  'restecg',
    'inp-thalach':  'thalach',
    'inp-exang':    'exang',
    'inp-oldpeak':  'oldpeak',
    'inp-slope':    'slope',
    'inp-ca':       'ca',
    'inp-thal':     'thal'
  };

  const errorBox = document.getElementById('form-error');
  const data     = {};
  const missing  = [];

  // Read each field
  for (const [id, feature] of Object.entries(fields)) {
    const val = document.getElementById(id).value;
    if (val === '' || val === null || val === undefined) {
      missing.push(FEATURE_LABELS[feature]);
    } else {
      data[feature] = parseFloat(val);
    }
  }

  // Validation
  if (missing.length > 0) {
    errorBox.style.display = 'block';
    errorBox.textContent   = `Please fill in: ${missing.join(', ')}`;
    return;
  }

  // Extra sanity checks
  const checks = [
    [data.age < 1 || data.age > 120,         'Age must be between 1 and 120'],
    [data.trestbps < 50 || data.trestbps > 250, 'Blood pressure seems invalid (50–250)'],
    [data.chol < 100 || data.chol > 600,     'Cholesterol seems invalid (100–600)'],
    [data.thalach < 50 || data.thalach > 250,'Max heart rate seems invalid (50–250)'],
    [data.oldpeak < 0 || data.oldpeak > 10,  'ST Depression must be between 0 and 10'],
  ];

  for (const [condition, message] of checks) {
    if (condition) {
      errorBox.style.display = 'block';
      errorBox.textContent   = '⚠️ ' + message;
      return;
    }
  }

  // All good — hide error, set as current patient
  errorBox.style.display = 'none';
  currentPatient         = data;

  // Build the patient grid (same as loadSample)
  const grid = document.getElementById('patient-grid');
  grid.innerHTML = '';

  for (const [key, val] of Object.entries(currentPatient)) {
    const item = document.createElement('div');
    item.className = 'grid-item';
    item.innerHTML = `
      <div class="label">${FEATURE_LABELS[key] || key}</div>
      <div class="value">${val}</div>
    `;
    grid.appendChild(item);
  }

  // Hide manual form, show patient section
  document.getElementById('manual-form').style.display = 'none';
  hide('section-prediction');
  hide('section-whatif');
  hide('section-comparison');
  show('section-patient');

  // Tag this as manually entered
  const tag = document.createElement('div');
  tag.style.cssText = `
    display:inline-block; background:#1e1a2e;
    border:1px solid #6d28d955; color:#a78bfa;
    font-size:0.75rem; padding:4px 10px;
    border-radius:20px; margin-bottom:12px;
  `;
  tag.textContent = '✏️ Manually entered values';
  grid.parentNode.insertBefore(tag, grid);
}