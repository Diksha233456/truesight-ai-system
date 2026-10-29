/**
 * TRUSIGHT Voice Analysis Client Module
 * Supports both Live Microphone recording and Audio File Upload.
 * Strictly renames Raw Model Score -> Deception Score.
 * Displays all measured Acoustic Observations in a clear, structured format.
 */

class VoiceAnalysisModule {
  constructor() {
    // Mode tabs
    this.tabLive = document.getElementById('tabVoiceLive');
    this.tabUpload = document.getElementById('tabVoiceUpload');
    this.paneLive = document.getElementById('paneVoiceLive');
    this.paneUpload = document.getElementById('paneVoiceUpload');

    // Live controls
    this.btnStartLive = document.getElementById('btnStartVoiceLive');
    this.btnStopLive = document.getElementById('btnStopVoiceLive');
    this.pillsDuration = document.querySelectorAll('.win-pill');
    this.liveCountdownVal = document.getElementById('liveCountdownVal');
    this.waveformCanvas = document.getElementById('waveformCanvas');

    // Upload controls
    this.dropzone = document.getElementById('voiceDropzone');
    this.fileInput = document.getElementById('voiceFileInput');
    this.uploadFilename = document.getElementById('uploadFilename');
    this.btnAnalyzeFile = document.getElementById('btnAnalyzeFile');

    // Metrics DOM
    this.deceptionScoreVal = document.getElementById('voiceDeceptionScoreVal');
    this.deceptionGauge = document.getElementById('voiceDeceptionGauge');
    this.confidenceVal = document.getElementById('voiceConfidenceVal');
    this.confidenceFill = document.getElementById('voiceConfidenceFill');
    this.speechStatusBadge = document.getElementById('voiceSpeechStatusBadge');
    this.verdictPill = document.getElementById('voiceVerdictPill');

    // Acoustic Observations DOM
    this.obsF0 = document.getElementById('obsF0');
    this.obsF0Desc = document.getElementById('obsF0Desc');
    this.obsTempo = document.getElementById('obsTempo');
    this.obsTempoDesc = document.getElementById('obsTempoDesc');
    this.obsPauses = document.getElementById('obsPauses');
    this.obsPausesDesc = document.getElementById('obsPausesDesc');
    this.obsStability = document.getElementById('obsStability');
    this.obsStabilityDesc = document.getElementById('obsStabilityDesc');
    this.obsIntensity = document.getElementById('obsIntensity');
    this.obsIntensityDesc = document.getElementById('obsIntensityDesc');
    this.structuredObsList = document.getElementById('structuredObsList');

    // History Table & Timeline Chart
    this.historyTableBody = document.getElementById('voiceHistoryTableBody');
    this.timelineCanvas = document.getElementById('voiceTimelineChart');

    // State
    this.selectedWindow = 15; // default 15s window per user requirement
    this.isLiveRecording = false;
    this.audioContext = null;
    this.analyser = null;
    this.micStream = null;
    this.mediaRecorder = null;
    this.recordedChunks = [];
    this.animFrameId = null;
    this.countdownTimer = null;
    this.segmentTimeout = null;
    this.currentSegNum = 0;
    this.timelineChart = null;
    this.segmentsHistory = [];
    this.selectedUploadFile = null;

    this.currentState = {
      ready: false,
      timestamp: null,
      verdict: 'WAITING',
      deceptionScore: null,
      confidence: null,
      speechDetected: false,
      speechStatus: 'WAITING',
      observations: [],
      features: null,
      window_seconds: this.selectedWindow,
    };


    this.initModeTabs();
    this.initDurationPills();
    this.initLiveControls();
    this.initUploadControls();
    this.initTimelineChart();
    this.connectWebSocket();
  }

  initModeTabs() {
    if (!this.tabLive || !this.tabUpload) return;
    this.tabLive.addEventListener('click', () => {
      this.tabLive.classList.add('active');
      this.tabUpload.classList.remove('active');
      this.paneLive.classList.add('active');
      this.paneUpload.classList.remove('active');
    });

    this.tabUpload.addEventListener('click', () => {
      this.tabUpload.classList.add('active');
      this.tabLive.classList.remove('active');
      this.paneUpload.classList.add('active');
      this.paneLive.classList.remove('active');
    });
  }

  initDurationPills() {
    this.pillsDuration.forEach(pill => {
      pill.addEventListener('click', () => {
        this.pillsDuration.forEach(p => p.classList.remove('selected'));
        pill.classList.add('selected');
        this.selectedWindow = parseInt(pill.dataset.sec, 10) || 30;
        this.currentState.window_seconds = this.selectedWindow;
      });
    });
  }

  initLiveControls() {
    if (this.btnStartLive) {
      this.btnStartLive.addEventListener('click', () => this.startLiveSession());
    }
    if (this.btnStopLive) {
      this.btnStopLive.addEventListener('click', () => this.stopLiveSession());
    }
  }

  initUploadControls() {
    if (!this.dropzone || !this.fileInput) return;

    this.dropzone.addEventListener('click', () => this.fileInput.click());

    this.dropzone.addEventListener('dragover', e => {
      e.preventDefault();
      this.dropzone.classList.add('dragover');
    });

    this.dropzone.addEventListener('dragleave', () => {
      this.dropzone.classList.remove('dragover');
    });

    this.dropzone.addEventListener('drop', e => {
      e.preventDefault();
      this.dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        this.handleFileSelected(e.dataTransfer.files[0]);
      }
    });

    this.fileInput.addEventListener('change', e => {
      if (e.target.files.length > 0) {
        this.handleFileSelected(e.target.files[0]);
      }
    });

    if (this.btnAnalyzeFile) {
      this.btnAnalyzeFile.addEventListener('click', () => this.analyzeSelectedFile());
    }
  }

  handleFileSelected(file) {
    this.selectedUploadFile = file;
    if (this.uploadFilename) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
      this.uploadFilename.innerHTML = `<strong>${escapeHtml(file.name)}</strong> (${sizeMb} MB)`;
    }
    if (this.btnAnalyzeFile) {
      this.btnAnalyzeFile.disabled = false;
    }
  }

  async analyzeSelectedFile() {
    if (!this.selectedUploadFile) return;

    const formData = new FormData();
    formData.append('audio', this.selectedUploadFile);

    try {
      if (this.btnAnalyzeFile) {
        this.btnAnalyzeFile.disabled = true;
        this.btnAnalyzeFile.textContent = 'Analyzing Voice...';
      }

      const res = await fetch('/api/voice/analyze?include_features=true', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail || `Upload failed with status ${res.status}`);
      }

      const data = await res.json();
      this.renderVoiceResult(data);
      this.recordHistoryEntry({
        segment: 'FILE',
        time: new Date().toLocaleTimeString(),
        verdict: data.speech_detected ? (data.verdict === 'Deceptive' ? 'Lie-class prediction' : data.verdict === 'Truthful' ? 'Truth-class prediction' : 'Inconclusive') : 'No Speech Detected',
        score: data.speech_detected ? data.score : null,
        confidence: data.speech_detected ? data.confidence : null,
        speech_detected: data.speech_detected,
      });
    } catch (err) {
      alert(`Voice Analysis error: ${err.message}`);
    } finally {
      if (this.btnAnalyzeFile) {
        this.btnAnalyzeFile.disabled = false;
        this.btnAnalyzeFile.textContent = '⚡ Analyze Audio File';
      }
    }
  }

  /* ── Live Microphone Implementation ────────────────────────────── */

  async startLiveSession() {
    try {
      this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.initWaveformVisualizer(this.micStream);

      // Start live session on backend with validated window
      let startRes;
      try {
        startRes = await fetch('/api/voice/live/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ window_seconds: this.selectedWindow }),
        });
      } catch (netErr) {
        throw new Error(`Cannot reach voice service: ${netErr.message}`);
      }

      if (!startRes.ok) {
        if (startRes.status === 409) {
          // Session already running on backend from previous run: cleanly stop and restart
          await fetch('/api/voice/live/stop', { method: 'POST' }).catch(() => {});
          const retryRes = await fetch('/api/voice/live/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ window_seconds: this.selectedWindow }),
          });
          if (!retryRes.ok) {
            let detail = `HTTP ${retryRes.status}`;
            try {
              const j = await retryRes.json();
              if (j && j.detail) detail = j.detail;
            } catch (_) {}
            throw new Error(`Voice session initialization failed: ${detail}`);
          }
        } else {
          let detail = `HTTP ${startRes.status}`;
          try {
            const j = await startRes.json();
            if (j && j.detail) detail = j.detail;
          } catch (_) {}
          throw new Error(`Voice service error (${detail})`);
        }
      }

      this.isLiveRecording = true;
      this.btnStartLive.style.display = 'none';
      this.btnStopLive.style.display = 'inline-flex';
      this.currentSegNum = 0;

      this.startSegmentRecording();
    } catch (err) {
      this.isLiveRecording = false;
      if (this.micStream) {
        this.micStream.getTracks().forEach(t => t.stop());
        this.micStream = null;
      }
      this.stopWaveformVisualizer();
      this.btnStartLive.style.display = 'inline-flex';
      this.btnStopLive.style.display = 'none';
      this.renderVoiceError(err.message);
      alert(`Voice recording start error: ${err.message}`);
    }
  }

  startSegmentRecording() {
    if (!this.isLiveRecording || !this.micStream) return;

    if (this.segmentTimeout) {
      clearTimeout(this.segmentTimeout);
      this.segmentTimeout = null;
    }

    this.recordedChunks = [];
    this.currentSegNum++;
    const segNum = this.currentSegNum;

    const mimeType = this.getSupportedMimeType();
    const options = mimeType ? { mimeType } : {};

    try {
      this.mediaRecorder = new MediaRecorder(this.micStream, options);
    } catch (e) {
      this.mediaRecorder = new MediaRecorder(this.micStream);
    }

    this.mediaRecorder.ondataavailable = e => {
      if (e.data && e.data.size > 0) {
        this.recordedChunks.push(e.data);
      }
    };

    this.mediaRecorder.onstop = async () => {
      // Capture available chunks before resetting buffer
      const chunks = this.recordedChunks;
      this.recordedChunks = [];
      const hasChunks = chunks.length > 0 && chunks.some(c => c.size > 0);

      // Start the next recording window immediately if live to prevent gaps
      if (this.isLiveRecording) {
        this.startSegmentRecording();
      }

      try {
        if (hasChunks) {
          const blob = new Blob(chunks, { type: mimeType || 'audio/webm' });
          await this.uploadLiveSegment(blob, segNum);
        }
      } finally {
        if (this._stopResolve) {
          this._stopResolve();
          this._stopResolve = null;
        }
      }
    };

    // Buffer audio continuously every 1000ms so early-stop captures partial audio
    try {
      this.mediaRecorder.start(1000);
    } catch (_) {
      this.mediaRecorder.start();
    }
    this.startCountdown(this.selectedWindow);

    this.segmentTimeout = setTimeout(() => {
      if (this.mediaRecorder && this.mediaRecorder.state === 'recording' && this.isLiveRecording) {
        try {
          this.mediaRecorder.stop();
        } catch (_) {}
      }
    }, this.selectedWindow * 1000);
  }

  async uploadLiveSegment(blob, segNum) {
    if (!blob || blob.size === 0) return;

    if (this.liveCountdownVal && this.isLiveRecording) {
      this.liveCountdownVal.textContent = 'Processing...';
    }

    const ext = blob.type.includes('ogg') ? 'ogg' : 'webm';
    const formData = new FormData();
    formData.append('audio', blob, `live_segment_${segNum}.${ext}`);

    try {
      const res = await fetch('/api/voice/live/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let errMessage = `HTTP ${res.status}`;
        try {
          const errData = await res.json();
          if (errData && errData.detail) {
            errMessage = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
          }
        } catch (_) {}
        this.renderVoiceError(errMessage, res.status);
        return;
      }

      const data = await res.json();
      this.renderVoiceResult(data);

      this.recordHistoryEntry({
        segment: `#${segNum}`,
        time: new Date().toLocaleTimeString(),
        verdict: data.speech_detected ? (data.verdict === 'Deceptive' ? 'Lie-class prediction' : data.verdict === 'Truthful' ? 'Truth-class prediction' : 'Inconclusive') : 'No Speech Detected',
        score: data.speech_detected ? data.score : null,
        confidence: data.speech_detected ? data.confidence : null,
        speech_detected: data.speech_detected,
      });
    } catch (err) {
      console.warn('Live segment upload failed:', err);
      this.renderVoiceError(err.message || 'Network error connecting to voice service');
    }
  }

  async stopLiveSession() {
    this.isLiveRecording = false;

    if (this.segmentTimeout) {
      clearTimeout(this.segmentTimeout);
      this.segmentTimeout = null;
    }

    if (this.countdownTimer) {
      clearInterval(this.countdownTimer);
      this.countdownTimer = null;
    }

    if (this.liveCountdownVal) {
      this.liveCountdownVal.textContent = 'Stopping...';
    }

    // Stop recorder safely; await onstop and final chunk upload before signaling backend stop
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      try {
        const stopPromise = new Promise(resolve => {
          this._stopResolve = resolve;
        });
        if (this.mediaRecorder.state === 'recording') {
          this.mediaRecorder.requestData();
        }
        this.mediaRecorder.stop();
        await Promise.race([
          stopPromise,
          new Promise(r => setTimeout(r, 8000)),
        ]);
      } catch (e) {
        console.warn('Error during mediaRecorder stop:', e);
      }
    }

    if (this.micStream) {
      this.micStream.getTracks().forEach(t => t.stop());
      this.micStream = null;
    }

    this.stopWaveformVisualizer();

    if (this.liveCountdownVal) {
      this.liveCountdownVal.textContent = 'Stopped';
    }

    this.btnStartLive.style.display = 'inline-flex';
    this.btnStopLive.style.display = 'none';

    await fetch('/api/voice/live/stop', { method: 'POST' }).catch(() => {});
  }

  renderVoiceError(errMessage, statusCode = null) {
    this.currentState.ready = false;
    this.currentState.timestamp = null;
    this.currentState.speechDetected = false;
    this.currentState.speechStatus = 'ERROR';
    this.currentState.verdict = 'ANALYSIS ERROR';

    const statusPrefix = statusCode ? `[HTTP ${statusCode}] ` : '';
    const safeMsg = errMessage ? String(errMessage).replace(/<[^>]+>/g, '') : 'Analysis error';

    if (this.verdictPill) {
      this.verdictPill.textContent = 'ANALYSIS ERROR';
      this.verdictPill.className = 'metric-badge uncertain';
    }

    if (this.speechStatusBadge) {
      this.speechStatusBadge.textContent = 'ERROR';
      this.speechStatusBadge.className = 'metric-badge uncertain';
    }

    if (this.deceptionScoreVal) {
      this.deceptionScoreVal.textContent = '—';
    }
    if (this.deceptionGauge) {
      this.deceptionGauge.style.strokeDashoffset = '188.5';
    }
    if (this.confidenceVal) {
      this.confidenceVal.textContent = '—';
    }
    if (this.confidenceFill) {
      this.confidenceFill.style.width = '0%';
    }

    if (this.liveCountdownVal) {
      this.liveCountdownVal.textContent = 'Error';
    }

    if (this.structuredObsList) {
      this.structuredObsList.innerHTML = `<li class="structured-obs-item" style="color:var(--status-uncertain);">${statusPrefix}${safeMsg}</li>`;
    }
  }

  startCountdown(seconds) {
    if (this.countdownTimer) clearInterval(this.countdownTimer);
    let rem = seconds;
    if (this.liveCountdownVal) this.liveCountdownVal.textContent = `${rem}s`;

    this.countdownTimer = setInterval(() => {
      rem--;
      if (rem <= 0) {
        clearInterval(this.countdownTimer);
        this.countdownTimer = null;
        if (this.liveCountdownVal) this.liveCountdownVal.textContent = 'Processing...';
      } else {
        if (this.liveCountdownVal) this.liveCountdownVal.textContent = `${rem}s`;
      }
    }, 1000);
  }

  getSupportedMimeType() {
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg'];
    for (const t of types) {
      if (MediaRecorder.isTypeSupported(t)) return t;
    }
    return '';
  }

  /* ── Audio Waveform Visualizer ─────────────────────────────────── */

  initWaveformVisualizer(stream) {
    if (!this.waveformCanvas) return;
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = this.audioContext.createMediaStreamSource(stream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 64;
      source.connect(this.analyser);

      const ctx = this.waveformCanvas.getContext('2d');
      const bufferLength = this.analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const draw = () => {
        if (!this.isLiveRecording) return;
        this.animFrameId = requestAnimationFrame(draw);
        this.analyser.getByteFrequencyData(dataArray);

        const width = this.waveformCanvas.width;
        const height = this.waveformCanvas.height;
        ctx.clearRect(0, 0, width, height);

        const barCount = 32;
        const barWidth = (width / barCount) - 2;
        let x = 0;

        for (let i = 0; i < barCount; i++) {
          const val = dataArray[i % bufferLength] || 0;
          const barHeight = Math.max(3, (val / 255) * height);
          const y = height - barHeight;

          const hue = 195 + (i * 2); // Cyan to electric blue
          ctx.fillStyle = `hsl(${hue}, 85%, ${40 + (val / 255) * 30}%)`;
          ctx.fillRect(x, y, barWidth, barHeight);

          x += barWidth + 2;
        }
      };

      draw();
    } catch (err) {
      console.warn('Waveform visualizer init error:', err);
    }
  }

  stopWaveformVisualizer() {
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.waveformCanvas) {
      const ctx = this.waveformCanvas.getContext('2d');
      ctx.clearRect(0, 0, this.waveformCanvas.width, this.waveformCanvas.height);
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      try { this.audioContext.close(); } catch (e) {}
    }
  }

  /* ── WebSocket Integration ──────────────────────────────────────── */

  connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/voice/live`;

    try {
      const ws = new WebSocket(wsUrl);
      ws.onmessage = evt => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === 'segment_result' && msg.data) {
            this.renderVoiceResult(msg.data);
          }
        } catch (e) {}
      };
      ws.onclose = () => {
        setTimeout(() => this.connectWebSocket(), 4000);
      };
    } catch (e) {}
  }

  /* ── Rendering Voice Results & Acoustic Observations ────────────── */

  renderVoiceResult(data) {
    if (!data) return;

    this.currentState.ready = true;
    const rawTs = (data.timestamp !== undefined && data.timestamp !== null) ? data.timestamp : null;
    const numTs = (rawTs !== null && typeof rawTs === 'number' && Number.isFinite(rawTs) && rawTs > 0)
      ? rawTs
      : (typeof rawTs === 'string' && !isNaN(Number(rawTs)) && Number.isFinite(Number(rawTs)) && Number(rawTs) > 0)
        ? Number(rawTs)
        : null;
    this.currentState.timestamp = numTs;
    this.currentState.speechDetected = Boolean(data.speech_detected);

    this.currentState.speechStatus = data.speech_status || '';

    if (!data.speech_detected) {
      // Condition: When speech is absent, clear current score and show "NO SPEECH DETECTED"
      this.currentState.verdict = 'NO SPEECH DETECTED';
      this.currentState.mappedClassLabel = 'NO SPEECH DETECTED';
      this.currentState.deceptionScore = null;
      this.currentState.confidence = null;

      if (this.verdictPill) {
        this.verdictPill.textContent = 'NO SPEECH DETECTED';
        this.verdictPill.className = 'metric-badge uncertain';
      }

      if (this.speechStatusBadge) {
        this.speechStatusBadge.textContent = 'NO SPEECH DETECTED';
        this.speechStatusBadge.className = 'metric-badge uncertain';
      }

      if (this.deceptionScoreVal) {
        this.deceptionScoreVal.textContent = '—';
      }

      if (this.deceptionGauge) {
        this.deceptionGauge.style.strokeDashoffset = '188.5';
      }

      if (this.confidenceVal) {
        this.confidenceVal.textContent = '—';
      }
      if (this.confidenceFill) {
        this.confidenceFill.style.width = '0%';
      }
    } else {
      // Speech detected: map verdict to explicit model prediction statuses
      const score = (data.score !== null && data.score !== undefined) ? Number(data.score) : null;
      let displayVerdict = 'INCONCLUSIVE';
      let verdictClass = 'uncertain';

      if (score !== null && score >= 0.45 && score <= 0.55) {
        displayVerdict = 'INCONCLUSIVE';
        verdictClass = 'uncertain';
      } else if (data.verdict === 'Deceptive' || (score !== null && score > 0.55)) {
        displayVerdict = 'MODEL PREDICTS LIE CLASS';
        verdictClass = 'lie';
      } else {
        displayVerdict = 'MODEL PREDICTS TRUTH CLASS';
        verdictClass = 'truth';
      }

      this.currentState.verdict = displayVerdict;
      this.currentState.mappedClassLabel = displayVerdict;
      this.currentState.deceptionScore = data.score;
      this.currentState.confidence = data.confidence;

      // 1. Verdict & Status
      if (this.verdictPill) {
        this.verdictPill.textContent = displayVerdict;
        this.verdictPill.className = 'metric-badge ' + verdictClass;
      }

      if (this.speechStatusBadge) {
        this.speechStatusBadge.textContent = data.speech_status || 'VOICE DETECTED';
        this.speechStatusBadge.className = 'metric-badge truth';
      }

      // 2. Acoustic Model Score (Raw Classifier Probability P[Class 1])
      const rawScore = data.score;
      if (this.deceptionScoreVal) {
        if (rawScore !== null && rawScore !== undefined) {
          this.deceptionScoreVal.textContent = `${Number(rawScore).toFixed(2)}`;
        } else {
          this.deceptionScoreVal.textContent = '—';
        }
      }

      if (this.deceptionGauge) {
        if (rawScore !== null && rawScore !== undefined) {
          const offset = 188.5 * (1.0 - Math.min(1.0, Math.max(0.0, rawScore)));
          this.deceptionGauge.style.strokeDashoffset = `${offset}`;
          this.deceptionGauge.style.stroke = rawScore > 0.5 ? 'var(--lie)' : 'var(--truth)';
        } else {
          this.deceptionGauge.style.strokeDashoffset = '188.5';
        }
      }

      // 3. Model Confidence (Uncalibrated Model Score)
      const conf = data.confidence;
      if (this.confidenceVal) {
        this.confidenceVal.textContent = conf !== null && conf !== undefined ? `${Math.round(conf * 100)}%` : '—';
      }
      if (this.confidenceFill) {
        this.confidenceFill.style.width = conf !== null && conf !== undefined ? `${Math.round(conf * 100)}%` : '0%';
      }
    }

    // 4. Acoustic Observations (Pitch, Tempo, Pauses, Stability, Intensity)
    const features = data.features || {};
    this.currentState.features = features;

    // Pitch (F0)
    if (features.pitch_mean) {
      if (this.obsF0) this.obsF0.textContent = `${features.pitch_mean.toFixed(1)} Hz`;
      if (this.obsF0Desc) this.obsF0Desc.textContent = `Std: ±${features.pitch_std ? features.pitch_std.toFixed(1) : '0'} Hz (Range: ${features.pitch_min ? features.pitch_min.toFixed(0) : 0}-${features.pitch_max ? features.pitch_max.toFixed(0) : 0} Hz)`;
    } else {
      if (this.obsF0) this.obsF0.textContent = 'Normative';
      if (this.obsF0Desc) this.obsF0Desc.textContent = 'Conversational fundamental frequency';
    }

    // Speech Tempo (Onsets/sec)
    if (features.speech_tempo) {
      if (this.obsTempo) this.obsTempo.textContent = `${features.speech_tempo.toFixed(1)} /s`;
      if (this.obsTempoDesc) this.obsTempoDesc.textContent = features.speech_tempo < 1.5 ? 'Deliberate cadence' : features.speech_tempo > 3.2 ? 'Elevated speech tempo' : 'Typical conversational tempo';
    } else {
      if (this.obsTempo) this.obsTempo.textContent = 'Standard';
      if (this.obsTempoDesc) this.obsTempoDesc.textContent = 'Conversational articulation pace';
    }

    // Pauses & Hesitation
    if (features.pause_ratio !== undefined) {
      const pct = Math.round(features.pause_ratio * 100);
      if (this.obsPauses) this.obsPauses.textContent = `${pct}% Silence`;
      if (this.obsPausesDesc) this.obsPausesDesc.textContent = `${features.pause_count || 0} pauses (${(features.silence_duration || 0).toFixed(1)}s total)`;
    } else {
      if (this.obsPauses) this.obsPauses.textContent = 'Moderate';
      if (this.obsPausesDesc) this.obsPausesDesc.textContent = 'Standard pause distribution';
    }

    // Vocal Stability (Jitter & Shimmer)
    if (features.jitter_local !== undefined) {
      const jPct = (features.jitter_local * 100).toFixed(2);
      const sPct = (features.shimmer_local * 100).toFixed(2);
      if (this.obsStability) this.obsStability.textContent = `${jPct}% Jitter`;
      if (this.obsStabilityDesc) this.obsStabilityDesc.textContent = `Shimmer: ${sPct}% (Glottal periodicity)`;
    } else {
      if (this.obsStability) this.obsStability.textContent = 'Stable';
      if (this.obsStabilityDesc) this.obsStabilityDesc.textContent = 'Normative vocal periodicity';
    }

    // Intensity (RMS Energy)
    if (features.rms_energy_mean !== undefined) {
      if (this.obsIntensity) this.obsIntensity.textContent = `${features.rms_energy_mean.toFixed(3)}`;
      if (this.obsIntensityDesc) this.obsIntensityDesc.textContent = `Std: ±${(features.rms_energy_std || 0).toFixed(3)} RMS`;
    }

    // 5. Structured Explanation List
    const structExpl = data.structured_explanation || {};
    const obsList = structExpl.acoustic_observations || data.explanation || [];
    this.currentState.observations = obsList;

    if (this.structuredObsList) {
      if (obsList.length > 0) {
        this.structuredObsList.innerHTML = obsList
          .map(obs => `<li class="structured-obs-item">${escapeHtml(obs)}</li>`)
          .join('');
      } else {
        this.structuredObsList.innerHTML = '<li class="structured-obs-item">Awaiting audio input for acoustic extraction...</li>';
      }

      // Add quality warnings if present
      const warnings = structExpl.quality_warnings || data.quality_warnings || [];
      if (warnings.length > 0) {
        this.structuredObsList.innerHTML += warnings
          .map(w => `<li class="structured-obs-item warning">⚠️ ${escapeHtml(w)}</li>`)
          .join('');
      }
    }
  }

  /* ── Timeline Chart & History Table ────────────────────────────── */

  initTimelineChart() {
    if (!this.timelineCanvas) return;
    try {
      const ctx = this.timelineCanvas.getContext('2d');
      this.timelineChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: 'Deception Score',
              data: [],
              borderColor: '#0EA5E9',
              backgroundColor: 'rgba(14, 165, 233, 0.1)',
              borderWidth: 2,
              tension: 0.3,
              fill: true,
              pointBackgroundColor: '#0EA5E9',
              pointRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              min: 0,
              max: 100,
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#64748B', callback: val => `${val}%` },
            },
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#64748B' },
            },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => `Deception Score: ${ctx.parsed.y}%`,
              },
            },
          },
        },
      });
    } catch (e) {
      console.warn('Timeline chart init failed:', e);
    }
  }

  recordHistoryEntry(entry) {
    this.segmentsHistory.unshift(entry);
    if (this.segmentsHistory.length > 25) this.segmentsHistory.pop();

    // Update History Table
    if (this.historyTableBody) {
      this.historyTableBody.innerHTML = this.segmentsHistory
        .map(s => {
          const scoreDisplay = s.score !== null && s.score !== undefined ? `${Number(s.score).toFixed(2)}` : '—';
          const confDisplay = s.confidence !== null && s.confidence !== undefined ? `${Math.round(s.confidence * 100)}%` : '—';
          const badgeClass = s.verdict.includes('Truth') ? 'truth' : s.verdict.includes('Lie') ? 'lie' : 'uncertain';
          return `
            <tr>
              <td><strong>${escapeHtml(s.segment)}</strong></td>
              <td>${escapeHtml(s.time)}</td>
              <td><span class="metric-badge ${badgeClass}">${escapeHtml(s.verdict)}</span></td>
              <td><strong>${scoreDisplay}</strong></td>
              <td>${confDisplay}</td>
              <td>${s.speech_detected ? 'Yes' : 'No'}</td>
            </tr>
          `;
        })
        .join('');
    }

    // Update Chart
    if (this.timelineChart && entry.score !== null && entry.score !== undefined) {
      const labels = this.timelineChart.data.labels;
      const data = this.timelineChart.data.datasets[0].data;

      labels.push(entry.segment);
      data.push(Math.round(entry.score * 100));

      if (labels.length > 15) {
        labels.shift();
        data.shift();
      }
      this.timelineChart.update();
    }
  }

  getState() {
    return {
      ...this.currentState,
      isLiveRecording: this.isLiveRecording,
      window_seconds: this.selectedWindow,
      speech_detected: this.currentState.speechDetected,
    };
  }
}

window.VoiceAnalysisModule = VoiceAnalysisModule;
