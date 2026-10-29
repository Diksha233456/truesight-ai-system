/**
 * TRUSIGHT Face Analysis Client Module
 * Manages live webcam video feed with native OpenCV overlays,
 * telemetry polling from /api/face/live, AU intensity bars, and camera lifecycle.
 */

class FaceAnalysisModule {
  constructor() {
    this.streamImg = document.getElementById('webcamFeed');
    this.cameraStatusPill = document.getElementById('faceEnginePill');
    this.btnToggleCam = document.getElementById('btnToggleCamera');
    this.btnSnapshot = document.getElementById('btnSnapshot');

    // Telemetry DOM elements
    this.verdictBadge = document.getElementById('faceVerdictBadge');
    this.truthScoreVal = document.getElementById('faceTruthScoreVal');
    this.truthMeterFill = document.getElementById('faceTruthMeterFill');
    this.lieProbVal = document.getElementById('faceLieProbVal');
    this.reasonsList = document.getElementById('faceReasonsList');
    this.sentimentBadge = document.getElementById('faceSentimentBadge');
    this.sentimentVal = document.getElementById('faceSentimentVal');
    this.sentimentFill = document.getElementById('faceSentimentFill');

    // Head pose
    this.posePitch = document.getElementById('posePitch');
    this.poseYaw = document.getElementById('poseYaw');
    this.poseRoll = document.getElementById('poseRoll');

    // HUD overlays
    this.hudFps = document.getElementById('hudFps');
    this.hudFaceId = document.getElementById('hudFaceId');
    this.faceActiveTag = document.getElementById('faceActiveTag');

    // Action unit elements map
    this.auBars = {
      AU01: document.getElementById('au01_fill'),
      AU02: document.getElementById('au02_fill'),
      AU04: document.getElementById('au04_fill'),
      AU12: document.getElementById('au12_fill'),
      AU15: document.getElementById('au15_fill'),
      AU25: document.getElementById('au25_fill'),
      AU26: document.getElementById('au26_fill'),
      AU45: document.getElementById('au45_fill'),
    };

    this.auVals = {
      AU01: document.getElementById('au01_val'),
      AU02: document.getElementById('au02_val'),
      AU04: document.getElementById('au04_val'),
      AU12: document.getElementById('au12_val'),
      AU15: document.getElementById('au15_val'),
      AU25: document.getElementById('au25_val'),
      AU26: document.getElementById('au26_val'),
      AU45: document.getElementById('au45_val'),
    };

    this.isCameraRunning = true;
    this.pollInterval = null;
    this.currentState = {
      ready: false,
      timestamp: null,
      verdict: 'WAITING',
      truthScore: null,
      pLie: null,
      sentiment: null,
      reasons: [],
      fps: 0,
    };

    this.initEventListeners();
    this.startPolling();
  }

  initEventListeners() {
    if (this.btnToggleCam) {
      this.btnToggleCam.addEventListener('click', () => this.toggleCamera());
    }

    if (this.btnSnapshot) {
      this.btnSnapshot.addEventListener('click', () => this.takeSnapshot());
    }

    // Handle video feed errors and auto-recover
    this._streamConnected = false;
    this._retryTimer = null;
    this._lastStreamReconnect = 0;
    if (this.streamImg) {
      this.streamImg.onerror = () => {
        this._streamConnected = false;
        if (this._retryTimer) clearTimeout(this._retryTimer);
        this._retryTimer = setTimeout(() => {
          if (this.isCameraRunning && this.streamImg) {
            this.streamImg.src = `/api/face/video_feed?t=${Date.now()}`;
          }
        }, 1500);
      };
      this.streamImg.onload = () => {
        this._streamConnected = true;
        this.updateCameraStatus(true);
      };
    }
  }

  startPolling() {
    this.fetchTelemetry();
    this.pollInterval = setInterval(() => this.fetchTelemetry(), 350);
  }

  stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  async fetchTelemetry() {
    if (this._isFetching) return;
    this._isFetching = true;
    try {
      const res = await fetch('/api/face/live');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      this.renderTelemetry(data);
    } catch (err) {
      // Backend may be momentarily busy or restarting
      this.updateCameraStatus(false);
    } finally {
      this._isFetching = false;
    }
  }

  renderTelemetry(data) {
    if (!data) return;

    this.currentState.ready = Boolean(data.ready);
    // If camera is reporting active but video stream is disconnected, auto-reconnect
    if (data.camera_running && !this._streamConnected && this.streamImg) {
      const now = Date.now();
      if (!this._lastStreamReconnect || (now - this._lastStreamReconnect > 2000)) {
        this._lastStreamReconnect = now;
        this.streamImg.src = `/api/face/video_feed?t=${now}`;
      }
    }
    const deception = data.deception || {};
    const rawTs = (data.timestamp !== undefined && data.timestamp !== null)
      ? data.timestamp
      : (deception.timestamp !== undefined && deception.timestamp !== null ? deception.timestamp : null);
    const numTs = (rawTs !== null && typeof rawTs === 'number' && Number.isFinite(rawTs) && rawTs > 0)
      ? rawTs
      : (typeof rawTs === 'string' && !isNaN(Number(rawTs)) && Number.isFinite(Number(rawTs)) && Number(rawTs) > 0)
        ? Number(rawTs)
        : null;
    this.currentState.timestamp = numTs;
    const emotion = data.emotion || {};
    const faceCues = data.face_cues || {};

    // 1. FPS and HUD
    const fps = data.fps || 0;
    this.currentState.fps = fps;
    if (this.hudFps) this.hudFps.textContent = `${fps.toFixed(1)} FPS`;
    if (this.hudFaceId) {
      const id = faceCues.face_id !== undefined ? faceCues.face_id : 1;
      this.hudFaceId.textContent = `ID: ${String(id).padStart(2, '0')}`;
    }

    // 2. Truth / Deception Score
    const hasData = Boolean(data.ready && deception && deception.label && deception.label !== 'WAITING');
    const pLie = (deception.p_lie !== null && deception.p_lie !== undefined) ? Number(deception.p_lie) : null;

    if (!hasData) {
      const isCalibrating = Boolean(data.ready && (!deception || deception.label === 'WAITING'));
      const statusLabel = isCalibrating ? 'INSUFFICIENT FACIAL DATA' : 'MODEL UNAVAILABLE';
      this.currentState.verdict = statusLabel;
      this.currentState.mappedClassLabel = statusLabel;
      this.currentState.pLie = null;
      this.currentState.classProbability = null;
      this.currentState.truthScore = null;
      this.currentState.signalQuality = isCalibrating ? 'CALIBRATING' : 'UNAVAILABLE';

      if (this.verdictBadge) {
        this.verdictBadge.textContent = statusLabel;
        this.verdictBadge.className = 'metric-badge uncertain';
      }

      if (this.truthScoreVal) {
        this.truthScoreVal.textContent = '—';
      }

      if (this.truthMeterFill) {
        this.truthMeterFill.style.width = '0%';
        this.truthMeterFill.className = 'meter-fill uncertain';
      }

      if (this.lieProbVal) {
        this.lieProbVal.textContent = isCalibrating ? 'Awaiting stable face window' : 'Camera offline';
      }
    } else {
      let verdictLabel = 'INCONCLUSIVE';
      let badgeClass = 'uncertain';
      let predProb = pLie;
      let scoreText = '—';
      let meaningText = '';

      if (pLie !== null) {
        const pTruth = Math.max(0.0, Math.min(1.0, 1.0 - pLie));
        if (pLie >= 0.45 && pLie <= 0.55) {
          verdictLabel = 'INCONCLUSIVE';
          badgeClass = 'uncertain';
          predProb = pLie;
          scoreText = `P(class): ${pLie.toFixed(3)}`;
          meaningText = `Threshold band: P(Lie)=${pLie.toFixed(3)}, P(Truth)=${pTruth.toFixed(3)}`;
        } else if (pLie > 0.55 || deception.label === 'LIE') {
          verdictLabel = 'MODEL PREDICTS LIE CLASS';
          badgeClass = 'lie';
          predProb = pLie;
          scoreText = `P(Lie): ${pLie.toFixed(3)}`;
          meaningText = `P(class 1 Lie): ${pLie.toFixed(3)} (uncalibrated)`;
        } else {
          verdictLabel = 'MODEL PREDICTS TRUTH CLASS';
          badgeClass = 'truth';
          predProb = pTruth;
          scoreText = `P(Truth): ${pTruth.toFixed(3)}`;
          meaningText = `P(class 0 Truth): ${pTruth.toFixed(3)} (P[Lie]=${pLie.toFixed(3)})`;
        }
      } else {
        verdictLabel = 'INCONCLUSIVE';
        badgeClass = 'uncertain';
        meaningText = 'No probability returned for current window';
      }

      this.currentState.verdict = verdictLabel;
      this.currentState.mappedClassLabel = verdictLabel;
      this.currentState.pLie = pLie;
      this.currentState.classProbability = predProb;
      this.currentState.truthScore = (pLie !== null ? (1.0 - pLie) : 0.5);
      this.currentState.signalQuality = 'USABLE_OBSERVATIONS';

      if (this.verdictBadge) {
        this.verdictBadge.textContent = verdictLabel;
        this.verdictBadge.className = 'metric-badge ' + badgeClass;
      }

      if (this.truthScoreVal) {
        this.truthScoreVal.textContent = scoreText;
      }

      if (this.truthMeterFill) {
        const pct = predProb !== null ? Math.round(predProb * 100) : 50;
        this.truthMeterFill.style.width = `${pct}%`;
        this.truthMeterFill.className = 'meter-fill ' + badgeClass;
      }

      if (this.lieProbVal) {
        this.lieProbVal.textContent = meaningText;
      }
    }

    // 3. Reasons List
    const reasons = deception.reasons || [];
    this.currentState.reasons = reasons;
    if (this.reasonsList) {
      if (reasons.length > 0) {
        this.reasonsList.innerHTML = reasons
          .slice(0, 3)
          .map(r => `<li class="reason-item">${escapeHtml(r)}</li>`)
          .join('');
      } else {
        this.reasonsList.innerHTML = '<li class="reason-item" style="color:var(--text-muted);">Calibrating facial cues baseline...</li>';
      }
    }

    // 4. Sentiment / Emotion (Distinguish genuine predictions from missing/uncalibrated state)
    const hasEmotion = Boolean(emotion && emotion.label && emotion.label.trim() !== '');
    const emoLabel = hasEmotion ? emotion.label.trim().toUpperCase() : (data.face_present ? 'CALIBRATING' : '—');
    const emoScore = (hasEmotion && emotion.score !== null && emotion.score !== undefined) ? emotion.score : null;
    this.currentState.sentiment = hasEmotion ? emoLabel : null;
    this.currentState.emotion = {
      label: hasEmotion ? emoLabel : '',
      score: emoScore,
      timestamp: (hasEmotion && emotion.timestamp) ? emotion.timestamp : null,
    };

    if (this.sentimentBadge) {
      this.sentimentBadge.textContent = emoLabel;
    }
    if (this.sentimentVal) {
      this.sentimentVal.textContent = emoScore !== null ? `${Math.round(emoScore * 100)}%` : '—';
    }
    if (this.sentimentFill) {
      this.sentimentFill.style.width = emoScore !== null ? `${Math.round(emoScore * 100)}%` : '0%';
    }

    // 5. Head Pose (Pitch, Yaw, Roll)
    const hp = faceCues.head_pose;
    if (hp) {
      if (this.posePitch) this.posePitch.textContent = `${hp.pitch > 0 ? '+' : ''}${hp.pitch}°`;
      if (this.poseYaw) this.poseYaw.textContent = `${hp.yaw > 0 ? '+' : ''}${hp.yaw}°`;
      if (this.poseRoll) this.poseRoll.textContent = `${hp.roll > 0 ? '+' : ''}${hp.roll}°`;
    }

    // 6. Action Units (FACS)
    const aus = faceCues.action_units || {};
    for (const [code, bar] of Object.entries(this.auBars)) {
      if (bar) {
        const val = aus[code] || 0.0;
        const pct = Math.min(100, Math.round(val * 100));
        bar.style.width = `${pct}%`;
        if (this.auVals[code]) {
          this.auVals[code].textContent = `${val.toFixed(2)}`;
        }
      }
    }

    this.updateCameraStatus(true);
  }

  updateCameraStatus(running) {
    this.isCameraRunning = running;
    if (this.cameraStatusPill) {
      const dot = this.cameraStatusPill.querySelector('.pill-dot');
      const text = this.cameraStatusPill.querySelector('.pill-text');
      if (dot) dot.className = 'pill-dot ' + (running ? 'online' : 'offline');
      if (text) text.textContent = running ? `FACE: ONLINE (${this.currentState.fps.toFixed(0)} FPS)` : 'FACE: STANDBY';
    }

    if (this.faceActiveTag) {
      this.faceActiveTag.textContent = running ? 'WEBCAM ACTIVE' : 'CAMERA STANDBY';
    }

    if (!running && this.hudFps) {
      this.hudFps.textContent = 'OFFLINE';
    }

    if (this.btnToggleCam) {
      if (running) {
        this.btnToggleCam.innerHTML = '🛑 Stop Camera';
        this.btnToggleCam.className = 'btn-tactical active-stop';
      } else {
        this.btnToggleCam.innerHTML = '📷 Start Camera';
        this.btnToggleCam.className = 'btn-tactical active-start';
      }
    }
  }

  async toggleCamera() {
    const endpoint = this.isCameraRunning ? '/api/face/camera/stop' : '/api/face/camera/start';
    try {
      if (this.btnToggleCam) this.btnToggleCam.disabled = true;
      const res = await fetch(endpoint, { method: 'POST' });
      const data = await res.json();
      this.updateCameraStatus(data.camera_running);

      if (data.camera_running && this.streamImg) {
        // Refresh stream source with cachebuster
        this.streamImg.src = `/api/face/video_feed?t=${Date.now()}`;
      }
    } catch (err) {
      console.error('Camera toggle failed:', err);
    } finally {
      if (this.btnToggleCam) this.btnToggleCam.disabled = false;
    }
  }

  takeSnapshot() {
    if (!this.streamImg) return;
    try {
      const canvas = document.createElement('canvas');
      canvas.width = this.streamImg.naturalWidth || 640;
      canvas.height = this.streamImg.naturalHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(this.streamImg, 0, 0, canvas.width, canvas.height);

      const link = document.createElement('a');
      link.download = `trusight_face_evidence_${Date.now()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    } catch (err) {
      console.warn('Snapshot capture failed (CORS stream):', err);
    }
  }

  getState() {
    return { ...this.currentState };
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

window.FaceAnalysisModule = FaceAnalysisModule;
