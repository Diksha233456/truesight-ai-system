/**
 * TRUSIGHT Master Dashboard Coordinator
 * Coordinates Face and Voice modules, updates the Combined Analysis Overview,
 * handles health monitoring, and manages global session controls.
 */

class TrusightDashboard {
  constructor() {
    this.faceModule = null;
    this.voiceModule = null;

    // Header elements
    this.systemStatusBadge = document.getElementById('systemStatusBadge');
    this.systemStatusDot = document.getElementById('systemStatusDot');
    this.systemStatusText = document.getElementById('systemStatusText');
    this.sessionTimerVal = document.getElementById('sessionTimerVal');
    this.facePill = document.getElementById('faceEnginePill');
    this.voicePill = document.getElementById('voiceEnginePill');

    // Overview elements
    this.overviewFaceVerdict = document.getElementById('overviewFaceVerdict');
    this.overviewFaceScore = document.getElementById('overviewFaceScore');
    this.overviewFaceDesc = document.getElementById('overviewFaceDesc');

    this.overviewVoiceVerdict = document.getElementById('overviewVoiceVerdict');
    this.overviewVoiceScore = document.getElementById('overviewVoiceScore');
    this.overviewVoiceDesc = document.getElementById('overviewVoiceDesc');

    this.overviewCorrelationText = document.getElementById('overviewCorrelationText');

    // Global action buttons
    this.btnStartAll = document.getElementById('btnStartAll');
    this.btnPauseAll = document.getElementById('btnPauseAll');
    this.btnResetAll = document.getElementById('btnResetAll');

    this.sessionSeconds = 0;
    this.timerInterval = null;
    this.syncInterval = null;
    this.healthInterval = null;

    this.init();
  }

  init() {
    // 1. Initialize submodules
    if (window.FaceAnalysisModule) {
      this.faceModule = new FaceAnalysisModule();
    }
    if (window.VoiceAnalysisModule) {
      this.voiceModule = new VoiceAnalysisModule();
    }

    // 2. Start session clock
    this.startSessionTimer();

    // 3. Start health checks & sync loop
    this.checkHealth();
    this.healthInterval = setInterval(() => this.checkHealth(), 3000);
    this.syncInterval = setInterval(() => this.syncCombinedOverview(), 500);

    // 4. Global button bindings
    if (this.btnStartAll) {
      this.btnStartAll.addEventListener('click', () => this.startAll());
    }
    if (this.btnPauseAll) {
      this.btnPauseAll.addEventListener('click', () => this.pauseAll());
    }
    if (this.btnResetAll) {
      this.btnResetAll.addEventListener('click', () => this.resetAll());
    }
  }

  startSessionTimer() {
    this.timerInterval = setInterval(() => {
      this.sessionSeconds++;
      const mm = String(Math.floor(this.sessionSeconds / 60)).padStart(2, '0');
      const ss = String(this.sessionSeconds % 60).padStart(2, '0');
      if (this.sessionTimerVal) {
        this.sessionTimerVal.textContent = `${mm}:${ss}`;
      }
    }, 1000);
  }

  async checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('Gateway unreachable');
      const data = await res.json();

      // Face Engine health
      const faceOk = data.face_engine && data.face_engine.status === 'ok';
      if (this.facePill) {
        const dot = this.facePill.querySelector('.pill-dot');
        const text = this.facePill.querySelector('.pill-text');
        if (dot) dot.className = 'pill-dot ' + (faceOk ? 'online' : 'offline');
        if (text) text.textContent = faceOk ? 'FACE: 8002 ONLINE' : 'FACE: 8002 STANDBY';
      }

      // Voice Engine health
      const voiceOk = data.voice_engine && data.voice_engine.status === 'running';
      if (this.voicePill) {
        const dot = this.voicePill.querySelector('.pill-dot');
        const text = this.voicePill.querySelector('.pill-text');
        if (dot) dot.className = 'pill-dot ' + (voiceOk ? 'online' : 'offline');
        if (text) {
          const ver = data.voice_engine.model_version || 'v1.0.0';
          text.textContent = voiceOk ? `VOICE: 8001 (${ver})` : 'VOICE: 8001 OFFLINE';
        }
      }

      // Global status
      if (this.systemStatusDot && this.systemStatusText) {
        if (faceOk && voiceOk) {
          this.systemStatusDot.className = 'status-dot';
          this.systemStatusText.textContent = 'SYSTEM ACTIVE — DUAL MODALITIES SYNCHRONIZED';
        } else if (faceOk || voiceOk) {
          this.systemStatusDot.className = 'status-dot';
          this.systemStatusText.textContent = 'PARTIAL MODE — 1 ENGINE ACTIVE';
        } else {
          this.systemStatusDot.className = 'status-dot offline';
          this.systemStatusText.textContent = 'CONNECTING TO ENGINES...';
        }
      }
    } catch (err) {
      if (this.systemStatusDot && this.systemStatusText) {
        this.systemStatusDot.className = 'status-dot offline';
        this.systemStatusText.textContent = 'GATEWAY CONNECTION ERROR';
      }
    }
  }

  syncCombinedOverview() {
    const faceState = this.faceModule ? this.faceModule.getState() : {};
    const voiceState = this.voiceModule ? this.voiceModule.getState() : {};

    // 1. Sync Face Overview Card
    if (this.overviewFaceVerdict) {
      const v = faceState.verdict || 'WAITING';
      this.overviewFaceVerdict.textContent = v;
      this.overviewFaceVerdict.className = 'overview-verdict ' + (v === 'TRUTH' ? 'truth' : v === 'LIE' ? 'lie' : 'uncertain');
    }

    if (this.overviewFaceScore) {
      if (faceState.truthScore !== null && faceState.truthScore !== undefined) {
        this.overviewFaceScore.textContent = `Truth: ${Math.round(faceState.truthScore * 100)}%`;
      } else {
        this.overviewFaceScore.textContent = 'Calibrating...';
      }
    }

    if (this.overviewFaceDesc) {
      const sentiment = faceState.sentiment ? `Sentiment: ${faceState.sentiment}` : 'Sentiment: Normal';
      const reason = faceState.reasons && faceState.reasons.length > 0 ? faceState.reasons[0] : 'Normative facial symmetry';
      this.overviewFaceDesc.textContent = `${sentiment} • ${reason}`;
    }

    // 2. Sync Voice Overview Card
    if (this.overviewVoiceVerdict) {
      const v = voiceState.verdict || 'WAITING';
      this.overviewVoiceVerdict.textContent = v;
      this.overviewVoiceVerdict.className = 'overview-verdict ' + (v === 'Truthful' ? 'truth' : v === 'Deceptive' ? 'lie' : 'uncertain');
    }

    if (this.overviewVoiceScore) {
      if (voiceState.deceptionScore !== null && voiceState.deceptionScore !== undefined) {
        this.overviewVoiceScore.textContent = `Deception: ${Math.round(voiceState.deceptionScore * 100)}%`;
      } else {
        this.overviewVoiceScore.textContent = voiceState.speechStatus || 'Waiting for speech';
      }
    }

    if (this.overviewVoiceDesc) {
      const obs = voiceState.observations && voiceState.observations.length > 0
        ? voiceState.observations[0]
        : (voiceState.speechDetected ? 'Acoustic cues within baseline' : 'Speak into microphone or upload audio to evaluate');
      this.overviewVoiceDesc.textContent = obs;
    }

    // 3. Cross-Modal Forensic Narrative (Honest, evidence-based correlation without artificial fusion)
    if (this.overviewCorrelationText) {
      const faceVerdict = faceState.verdict || 'UNCERTAIN';
      const voiceVerdict = voiceState.verdict || 'Inconclusive';

      let narrative = '';
      if (faceState.ready && voiceState.ready) {
        if (faceVerdict === 'TRUTH' && voiceVerdict === 'Truthful') {
          narrative = 'Consensus Truthful Baseline: Facial micro-cues and vocal acoustic parameters both exhibit low deception likelihood with normative conversational pacing.';
        } else if (faceVerdict === 'LIE' && voiceVerdict === 'Deceptive') {
          narrative = 'Dual-Modality Elevated Risk: Both facial cue analysis and vocal acoustic measurements indicate atypical perturbation patterns consistent with deception criteria.';
        } else if (faceVerdict === 'LIE' && voiceVerdict === 'Truthful') {
          narrative = 'Modality Divergence: Facial tracking detected micro-tension cues, whereas vocal frequency and pause dynamics remain within normative ranges. Correlate with subject context.';
        } else if (faceVerdict === 'TRUTH' && voiceVerdict === 'Deceptive') {
          narrative = 'Modality Divergence: Vocal acoustic model indicated elevated deception probability, while facial expression and gaze alignment remain calm and baseline.';
        } else {
          narrative = 'Observation State: Models active. Facial cues and vocal acoustic measurements are being logged independently in real time.';
        }
      } else if (faceState.ready) {
        narrative = `Live facial telemetry active (${faceVerdict}). Record or upload voice sample to complete dual-modality cross-evaluation.`;
      } else if (voiceState.ready) {
        narrative = `Voice analysis active (${voiceVerdict}). Ensure webcam is active for concurrent facial cue tracking.`;
      } else {
        narrative = 'System awaiting live inputs. Position face in front of webcam and speak into microphone to begin multimodal deception analysis.';
      }

      this.overviewCorrelationText.textContent = narrative;
    }
  }

  async startAll() {
    if (this.faceModule && !this.faceModule.isCameraRunning) {
      await this.faceModule.toggleCamera();
    }
    if (this.voiceModule && !this.voiceModule.isLiveRecording) {
      await this.voiceModule.startLiveSession();
    }
  }

  async pauseAll() {
    if (this.faceModule && this.faceModule.isCameraRunning) {
      await this.faceModule.toggleCamera();
    }
    if (this.voiceModule && this.voiceModule.isLiveRecording) {
      this.voiceModule.stopLiveSession();
    }
  }

  resetAll() {
    this.sessionSeconds = 0;
    if (this.voiceModule) {
      this.voiceModule.segmentsHistory = [];
      if (this.voiceModule.historyTableBody) this.voiceModule.historyTableBody.innerHTML = '';
      if (this.voiceModule.timelineChart) {
        this.voiceModule.timelineChart.data.labels = [];
        this.voiceModule.timelineChart.data.datasets[0].data = [];
        this.voiceModule.timelineChart.update();
      }
    }
  }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.trusightApp = new TrusightDashboard();
});
