/**
 * EcoSort – Frontend Controller v2.0
 * Bugs fixed: tab panel visibility, spinner/icon toggle, drag-leave child,
 * camera live hidden-attr handling, btn-classify state, confidence track aria.
 */
'use strict';

(function () {
  if (window.EcoSortInitialized) return;
  window.EcoSortInitialized = true;
  /* ── Constants ─────────────────────────────────────────────────────── */
  const MAX_BYTES      = 5 * 1024 * 1024;          // 5 MB
  const ALLOWED_TYPES  = ['image/jpeg', 'image/png', 'image/webp'];
  const ALLOWED_EXTS   = /\.(jpe?g|png|webp)$/i;
  const API_PREDICT    = '/predict';

  /* ── Category icons (SVG strings, no innerHTML on user data) ───────── */
  const CAT_ICONS = {
    'Plastic': '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 19H4.815a1.83 1.83 0 0 1-1.57-.881 1.785 1.785 0 0 1-.004-1.784L7.196 9.5"/><path d="M11 19h8.2a1.8 1.8 0 0 0 1.7-1.1 1.8 1.8 0 0 0-.1-2l-4.2-7.1"/><path d="m13 4-2.5 4.5"/><path d="m14 14 3-5"/><circle cx="12" cy="12" r="1.5"/></svg>',
    'Paper':   '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    'Metal':   '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',
    'Glass':   '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 22h8"/><path d="M12 15v7"/><path d="M12 15a5 5 0 0 0 5-5V4H7v6a5 5 0 0 0 5 5z"/></svg>',
    'Organic Waste': '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>',
    'Electronic Waste': '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    'Unknown / Uncertain': '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
  };

  /* Fallback disposal guidance (used when backend is not yet running) */
  const GUIDANCE_FALLBACK = {
    'Plastic': [
      'Rinse containers and remove residual liquids before recycling.',
      'Check resin codes (1–7) against your local curbside programme.',
      'Drop soft plastic films at supermarket collection points.'
    ],
    'Paper': [
      'Recycle clean, dry paper and flattened cardboard.',
      'Do not recycle greasy food-stained paper.',
      'Remove plastic sleeves and tape before recycling.'
    ],
    'Metal': [
      'Rinse cans; aluminium is infinitely recyclable.',
      'Wrap sharp lids safely or keep inside the crushed can.',
      'Take heavy scrap metal to a municipal drop-off facility.'
    ],
    'Glass': [
      'Rinse bottles and jars; remove caps and lids.',
      'Do not mix Pyrex, ceramics, or mirrors with container glass.',
      'Wrap broken glass in paper before disposal.'
    ],
    'Organic Waste': [
      'Compost at home or use municipal organic waste bins.',
      'Avoid adding meat, dairy, or cooking oil to home compost.',
      'Composting diverts methane-generating waste from landfill.'
    ],
    'Electronic Waste': [
      'NEVER bin batteries or electronics in ordinary recycling.',
      'Use certified e-waste collection centres or take-back schemes.',
      'Tape exposed battery terminals before transport.'
    ],
    'Unknown / Uncertain': [
      'The model could not identify this item confidently.',
      'Re-photograph with a single item, clear background, good lighting.',
      'Check the item label or consult your local waste authority.'
    ]
  };

  /* ── Application state ──────────────────────────────────────────────── */
  const S = {
    file:         null,    // File | null
    previewURL:   null,    // object URL | null
    stream:       null,    // MediaStream | null
    classifying:  false,
    activeTab:    'upload',
    facingMode:   'environment'
  };

  /* ── DOM refs ────────────────────────────────────────────────────────── */
  function $id(id) { return document.getElementById(id); }

  const el = {
    header:         $id('site-header'),
    hamburger:      $id('hamburger'),
    mainNav:        $id('main-nav'),
    navLinks:       document.querySelectorAll('.nav-link'),

    tabUpload:      $id('tab-upload'),
    tabCamera:      $id('tab-camera'),
    panelUpload:    $id('panel-upload'),
    panelCamera:    $id('panel-camera'),

    dropzone:       $id('dropzone'),
    fileInput:      $id('file-input'),

    cameraPlaceholder: $id('camera-placeholder'),
    cameraLive:     $id('camera-live'),
    cameraVideo:    $id('camera-video'),
    cameraCanvas:   $id('camera-canvas'),
    btnStartCamera: $id('btn-start-camera'),
    btnStopCamera:  $id('btn-stop-camera'),
    btnCapture:     $id('btn-capture'),
    btnFlipCamera:  $id('btn-flip-camera'),

    alertBox:       $id('alert-error'),
    alertMsg:       $id('alert-message'),
    btnDismiss:     $id('btn-dismiss-error'),

    previewPanel:   $id('preview-panel'),
    previewImg:     $id('preview-img'),
    previewName:    $id('preview-name'),
    previewSize:    $id('preview-size'),
    btnRemove:      $id('btn-remove-image'),

    btnClassify:    $id('btn-classify'),
    classifySpinner:$id('classify-spinner'),
    classifyIcon:   $id('classify-icon'),
    classifyLabel:  $id('classify-label'),

    resultCard:     $id('result-card'),
    resultIconWrap: $id('result-icon-wrap'),
    resultCategory: $id('result-category'),
    resultBadge:    $id('result-badge'),
    resultBadgeText:$id('result-badge-text'),

    confidenceBox:  $id('confidence-box'),
    confidencePct:  $id('confidence-pct'),
    confidenceFill: $id('confidence-fill'),
    confidenceTrack:$id('confidence-track'),

    uncertaintyBox: $id('uncertainty-box'),
    disposalList:   $id('disposal-list'),
    btnReset:       $id('btn-reset')
  };

  /* ── Utilities ───────────────────────────────────────────────────────── */
  function fmtBytes(b) {
    if (!b) return '0 B';
    const u = ['B','KB','MB','GB'];
    const i = Math.floor(Math.log(b) / Math.log(1024));
    return (b / Math.pow(1024, i)).toFixed(1) + ' ' + u[i];
  }

  function showError(msg) {
    el.alertMsg.textContent = msg;          // textContent — safe, no XSS
    el.alertBox.hidden = false;
    el.alertBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function clearError() {
    el.alertBox.hidden = true;
    el.alertMsg.textContent = '';
  }

  function validateFile(file) {
    if (!file) { showError('Please select or capture an image first.'); return false; }
    const typeOK = ALLOWED_TYPES.includes(file.type) || ALLOWED_EXTS.test(file.name);
    if (!typeOK) { showError('Unsupported format. Please upload a JPG, PNG, or WEBP image.'); return false; }
    if (file.size > MAX_BYTES) { showError(`File is too large (${fmtBytes(file.size)}). Maximum allowed is 5 MB.`); return false; }
    clearError();
    return true;
  }

  /* ── File / Preview ──────────────────────────────────────────────────── */
  function useFile(file) {
    if (!validateFile(file)) return;

    // Revoke any previous object URL to avoid memory leak
    if (S.previewURL) { URL.revokeObjectURL(S.previewURL); S.previewURL = null; }

    S.file       = file;
    S.previewURL = URL.createObjectURL(file);

    el.previewImg.src        = S.previewURL;
    el.previewName.textContent = file.name || 'captured_image.jpg';
    el.previewSize.textContent = fmtBytes(file.size);

    // FIX: hide previous result and reset button state before showing preview
    el.resultCard.hidden    = true;
    el.previewPanel.hidden  = false;

    // Reset button to normal state (in case a previous classify was done)
    setClassifyLoading(false);
    clearError();

    el.previewPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function resetAll() {
    S.file = null;
    if (S.previewURL) { URL.revokeObjectURL(S.previewURL); S.previewURL = null; }

    el.fileInput.value     = '';
    el.previewImg.src      = '';
    el.previewPanel.hidden = true;
    el.resultCard.hidden   = true;
    setClassifyLoading(false);
    clearError();

    $id('classifier-shell').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  /* ── Tab switching ───────────────────────────────────────────────────── */
  function setTab(tab) {
    if (S.activeTab === tab) return;
    S.activeTab = tab;

    const goUpload = tab === 'upload';

    el.tabUpload.classList.toggle('active', goUpload);
    el.tabUpload.setAttribute('aria-selected', goUpload);
    // FIX: set hidden attribute directly — the CSS rule `.tab-panel[hidden]{display:none!important}`
    // ensures no other CSS rule can accidentally show it.
    el.panelUpload.hidden = !goUpload;

    el.tabCamera.classList.toggle('active', !goUpload);
    el.tabCamera.setAttribute('aria-selected', !goUpload);
    el.panelCamera.hidden = goUpload;

    if (goUpload) stopCamera();
  }

  /* ── Camera ──────────────────────────────────────────────────────────── */
  async function startCamera() {
    clearError();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError('Camera not supported by this browser, or the page is not served over HTTPS / localhost.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: S.facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });

      S.stream = stream;
      el.cameraVideo.srcObject = stream;
      el.cameraPlaceholder.hidden = true;
      el.cameraLive.hidden        = false;

      // Show flip button if multiple cameras are available
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        if (devices.filter(d => d.kind === 'videoinput').length > 1) {
          el.btnFlipCamera.style.visibility = 'visible';
        }
      } catch (_) { /* ignore */ }

    } catch (err) {
      const msg = {
        NotAllowedError:      'Camera permission denied. Please allow camera access in your browser settings.',
        PermissionDeniedError:'Camera permission denied.',
        NotFoundError:        'No camera found on this device.',
        DevicesNotFoundError: 'No camera found on this device.',
        NotReadableError:     'Camera is in use by another application.',
        TrackStartError:      'Camera is in use by another application.',
      }[err.name] || `Camera error: ${err.message || 'unknown error.'}`;
      showError(msg);
    }
  }

  function stopCamera() {
    if (S.stream) {
      S.stream.getTracks().forEach(t => t.stop());
      S.stream = null;
    }
    if (el.cameraVideo) el.cameraVideo.srcObject = null;
    el.cameraLive.hidden        = true;
    el.cameraPlaceholder.hidden = false;
    el.btnFlipCamera.style.visibility = 'hidden';
  }

  async function flipCamera() {
    S.facingMode = S.facingMode === 'environment' ? 'user' : 'environment';
    stopCamera();
    await startCamera();
  }

  function capturePhoto() {
    if (!S.stream || !el.cameraVideo) return;

    const video = el.cameraVideo;
    const canvas = el.cameraCanvas;
    const w = video.videoWidth  || 640;
    const h = video.videoHeight || 480;

    canvas.width  = w;
    canvas.height = h;
    canvas.getContext('2d').drawImage(video, 0, 0, w, h);

    canvas.toBlob(blob => {
      if (!blob) { showError('Failed to capture the photo. Please try again.'); return; }
      const ts   = new Date().toISOString().replace(/[:.]/g, '-');
      const file = new File([blob], `ecosort_${ts}.jpg`, { type: 'image/jpeg' });
      stopCamera();
      useFile(file);
    }, 'image/jpeg', 0.92);
  }

  /* ── Classify button loading state ──────────────────────────────────── */
  // FIX: toggle both the spinner AND the search icon so the button
  // never shows both at the same time.
  function setClassifyLoading(loading) {
    el.btnClassify.disabled           = loading;
    el.classifySpinner.hidden         = !loading;
    el.classifyIcon.style.display     = loading ? 'none' : '';    // hide/show icon
    el.classifyLabel.textContent      = loading ? 'Analysing image...' : 'Analyse with AI';
  }

  /* ── Inference ───────────────────────────────────────────────────────── */
  async function runClassify() {
    if (S.classifying || !S.file) return;
    S.classifying = true;
    clearError();
    setClassifyLoading(true);

    const form = new FormData();
    form.append('image', S.file);

    try {
      const res  = await fetch(API_PREDICT, { method: 'POST', body: form });
      let   data = null;

      try { data = await res.json(); } catch (_) { /* non-JSON response */ }

      if (!res.ok) {
        const msg = data?.error || `Server returned ${res.status}. Check that app.py is running and the model is trained.`;
        throw new Error(msg);
      }

      if (!data || data.success !== true) {
        throw new Error(data?.error || 'Unexpected response from the server.');
      }

      showResult(data);

    } catch (err) {
      const raw = err.message || '';
      if (raw.includes('Failed to fetch') || raw.includes('NetworkError') || raw.includes('Load failed')) {
        showError('Cannot reach the backend. Make sure you have started "python app.py" (Flask server) and trained the model first.');
      } else {
        showError(raw || 'An unexpected error occurred. Please try again.');
      }
    } finally {
      S.classifying = false;
      setClassifyLoading(false);
    }
  }

  /* ── Render result ───────────────────────────────────────────────────── */
  function showResult(data) {
    const cat     = (typeof data.category === 'string' && data.category) ? data.category : 'Unknown / Uncertain';
    const pct     = typeof data.confidence_percent === 'number'
                    ? Math.round(Math.min(100, Math.max(0, data.confidence_percent)))
                    : (typeof data.confidence === 'number' ? Math.round(data.confidence * 100) : 0);
    const uncertain = Boolean(data.uncertain) || cat === 'Unknown / Uncertain';

    // Category icon — use innerHTML only with our OWN icon SVG strings (not user data)
    el.resultIconWrap.innerHTML = CAT_ICONS[cat] || CAT_ICONS['Unknown / Uncertain'];

    // FIX: use data-cat attribute (single-word key) so CSS selector without spaces works
    el.resultCard.setAttribute('data-cat', cat);

    // Category name — textContent, safe
    el.resultCategory.textContent = cat;

    // Badge
    el.resultBadgeText.textContent = uncertain ? 'Uncertain' : 'High Confidence';
    el.resultBadge.classList.toggle('uncertain', uncertain);

    // Confidence
    el.confidencePct.textContent = pct + '%';
    // Force reflow so CSS transition animates from 0
    el.confidenceFill.style.width = '0%';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { el.confidenceFill.style.width = pct + '%'; });
    });
    el.confidenceTrack.setAttribute('aria-valuenow', pct);

    // Uncertainty notice
    el.uncertaintyBox.hidden = !uncertain;

    // Disposal list — built with safe textContent only
    const items = Array.isArray(data.disposal_guidance) && data.disposal_guidance.length
                  ? data.disposal_guidance
                  : (GUIDANCE_FALLBACK[cat] || GUIDANCE_FALLBACK['Unknown / Uncertain']);

    el.disposalList.innerHTML = '';
    items.forEach(text => {
      const li   = document.createElement('li');
      li.className = 'disposal-item';

      const ico  = document.createElement('span');
      ico.className = 'disposal-item-icon';
      ico.setAttribute('aria-hidden', 'true');
      ico.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';

      const txt  = document.createElement('span');
      txt.textContent = text;   // SAFE — no innerHTML on server data

      li.appendChild(ico);
      li.appendChild(txt);
      el.disposalList.appendChild(li);
    });

    el.resultCard.hidden = false;
    el.resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /* ── Event wiring ────────────────────────────────────────────────────── */
  function init() {

    /* ── Header: scroll shadow ── */
    window.addEventListener('scroll', () => {
      el.header?.classList.toggle('scrolled', window.scrollY > 20);
    }, { passive: true });

    /* ── Mobile nav ── */
    if (el.hamburger && el.mainNav) {
      el.hamburger.addEventListener('click', () => {
        const open = el.mainNav.classList.toggle('open');
        el.hamburger.setAttribute('aria-expanded', open);
      });
      // Close on link click
      el.navLinks.forEach(a => a.addEventListener('click', () => {
        el.mainNav.classList.remove('open');
        el.hamburger.setAttribute('aria-expanded', 'false');
      }));
      // Close on outside click
      document.addEventListener('click', e => {
        if (!el.mainNav.contains(e.target) && !el.hamburger.contains(e.target)) {
          el.mainNav.classList.remove('open');
          el.hamburger.setAttribute('aria-expanded', 'false');
        }
      });
    }

    /* ── Tabs ── */
    el.tabUpload?.addEventListener('click', () => setTab('upload'));
    el.tabCamera?.addEventListener('click', () => setTab('camera'));

    /* ── Drop zone ── */
    if (el.dropzone && el.fileInput) {
      el.dropzone.addEventListener('click',   () => el.fileInput.click());
      el.dropzone.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.fileInput.click(); }
      });
      el.fileInput.addEventListener('change', e => {
        if (e.target.files?.[0]) useFile(e.target.files[0]);
      });

      // Drag & drop — FIX: use counter to handle child element dragenter/dragleave correctly
      let dragDepth = 0;
      el.dropzone.addEventListener('dragenter', e => {
        e.preventDefault();
        dragDepth++;
        el.dropzone.classList.add('is-over');
      });
      el.dropzone.addEventListener('dragleave', () => {
        dragDepth--;
        if (dragDepth <= 0) { dragDepth = 0; el.dropzone.classList.remove('is-over'); }
      });
      el.dropzone.addEventListener('dragover', e => e.preventDefault());
      el.dropzone.addEventListener('drop', e => {
        e.preventDefault();
        dragDepth = 0;
        el.dropzone.classList.remove('is-over');
        const file = e.dataTransfer?.files?.[0];
        if (file) useFile(file);
      });
    }

    /* ── Camera controls ── */
    el.btnStartCamera?.addEventListener('click', startCamera);
    el.btnStopCamera?.addEventListener('click',  stopCamera);
    el.btnCapture?.addEventListener('click',     capturePhoto);
    el.btnFlipCamera?.addEventListener('click',  flipCamera);

    // Release camera when page is unloaded
    window.addEventListener('beforeunload', stopCamera);
    // Release camera when tab is hidden (mobile power saving)
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) stopCamera();
    });

    /* ── Classification ── */
    el.btnClassify?.addEventListener('click',     runClassify);
    el.btnRemove?.addEventListener('click',       resetAll);
    el.btnReset?.addEventListener('click',        resetAll);
    el.btnDismiss?.addEventListener('click',      clearError);

    /* ── Smooth scroll offset for sticky header ── */
    document.querySelectorAll('a[href^="#"]').forEach(a => {
      a.addEventListener('click', e => {
        const id = a.getAttribute('href').slice(1);
        const target = document.getElementById(id);
        if (!target) return;
        e.preventDefault();
        const top = target.getBoundingClientRect().top + window.scrollY - (parseInt(getComputedStyle(document.documentElement).getPropertyValue('--header-h')) || 68) - 16;
        window.scrollTo({ top, behavior: 'smooth' });
      });
    });
  }

  /* ── Bootstrap ───────────────────────────────────────────────────────── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
