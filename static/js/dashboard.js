/**
 * TRUSIGHT Master Dashboard Coordinator
 * Coordinates Face and Voice modules, updates the Combined Analysis Overview,
 * handles health monitoring, and manages global session controls.
 */

class TrusightDashboard {
  constructor() {
    this.faceModule = null;
    this.voiceModule = null;
    this.geminiModule = null;

    // Header elements
    this.systemStatusBadge = document.getElementById('systemStatusBadge');
    this.systemStatusDot = document.getElementById('systemStatusDot');
    this.systemStatusText = document.getElementById('systemStatusText');
    this.sessionTimerVal = document.getElementById('sessionTimerVal');
    this.facePill = document.getElementById('faceEnginePill');
    this.voicePill = document.getElementById('voiceEnginePill');
    this.geminiPill = document.getElementById('geminiEnginePill');

    // Overview elements
    this.overviewFaceVerdict = document.getElementById('overviewFaceVerdict');
    this.overviewFaceScore = document.getElementById('overviewFaceScore');
    this.overviewFaceDesc = document.getElementById('overviewFaceDesc');

    this.overviewVoiceVerdict = document.getElementById('overviewVoiceVerdict');
    this.overviewVoiceScore = document.getElementById('overviewVoiceScore');
    this.overviewVoiceDesc = document.getElementById('overviewVoiceDesc');

    this.overviewGeminiVerdict = document.getElementById('overviewGeminiVerdict');
    this.overviewGeminiScore = document.getElementById('overviewGeminiScore');
    this.overviewGeminiDesc = document.getElementById('overviewGeminiDesc');

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
    if (window.GeminiAnalysisModule) {
      this.geminiModule = new GeminiAnalysisModule();
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

      // Gemini Engine health
      const geminiOk = data.gemini_engine && (data.gemini_engine.status === 'running' || data.gemini_engine.service);
      if (this.geminiPill) {
        const dot = this.geminiPill.querySelector('.pill-dot');
        const text = this.geminiPill.querySelector('.pill-text');
        if (dot) dot.className = 'pill-dot ' + (geminiOk ? 'online' : 'offline');
        if (text) text.textContent = geminiOk ? 'GEMINI: 8003 ONLINE' : 'GEMINI: 8003 OFFLINE';
      }

      // Global status
      if (this.systemStatusDot && this.systemStatusText) {
        const activeCount = [faceOk, voiceOk, geminiOk].filter(Boolean).length;
        if (activeCount === 3) {
          this.systemStatusDot.className = 'status-dot';
          this.systemStatusText.textContent = 'SYSTEM ACTIVE — TRIPLE MODALITIES SYNCHRONIZED';
        } else if (activeCount > 0) {
          this.systemStatusDot.className = 'status-dot';
          this.systemStatusText.textContent = `PARTIAL MODE — ${activeCount}/3 ENGINES ACTIVE`;
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
    const geminiState = this.geminiModule ? this.geminiModule.getState() : {};

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

    // 3. Sync Gemini Evidence Overview Card
    if (this.overviewGeminiVerdict) {
      const gv = geminiState.verdict || 'STANDBY';
      this.overviewGeminiVerdict.textContent = gv.toUpperCase();
      this.overviewGeminiVerdict.className = 'overview-verdict ' + (
        gv.toLowerCase().includes('truth') || gv.toLowerCase().includes('consistent') ? 'truth' :
        gv.toLowerCase().includes('lie') || gv.toLowerCase().includes('decept') || gv.toLowerCase().includes('contradict') ? 'lie' : 'uncertain'
      );
    }

    if (this.overviewGeminiScore) {
      if (geminiState.rawScore !== null && geminiState.rawScore !== undefined) {
        this.overviewGeminiScore.textContent = `Risk: ${Math.round(geminiState.rawScore)}/100`;
      } else {
        this.overviewGeminiScore.textContent = geminiState.caseLoaded ? 'Awaiting Speech' : 'No Case Loaded';
      }
    }

    if (this.overviewGeminiDesc) {
      if (geminiState.transcript) {
        const lastChunk = geminiState.transcript.trim().split('\n').pop();
        this.overviewGeminiDesc.textContent = `Latest: "${lastChunk.substring(0, 70)}${lastChunk.length > 70 ? '...' : ''}"`;
      } else if (geminiState.caseLoaded) {
        this.overviewGeminiDesc.textContent = 'Evidence indexed. Speak into Gemini channel to verify facts.';
      } else {
        this.overviewGeminiDesc.textContent = 'Upload case evidence PDF and stream live speech to assess contradiction probability.';
      }
    }

    // 4. Cross-Modal Forensic Assessment & Uncertainty Resolution
    const badge = document.getElementById('crossModalVerdictBadge');
    const scoreVal = document.getElementById('crossModalScoreVal');

    if (this.overviewCorrelationText) {
      const faceVerdict = (faceState.verdict || 'UNCERTAIN').toUpperCase();
      const voiceVerdict = (voiceState.verdict || 'Inconclusive');
      const voiceScore = voiceState.deceptionScore !== null && voiceState.deceptionScore !== undefined ? voiceState.deceptionScore : null;
      const facePLie = faceState.pLie !== null && faceState.pLie !== undefined ? faceState.pLie : (faceState.truthScore !== null ? (1.0 - faceState.truthScore) : null);
      const geminiScore = geminiState.deceptionScore !== null && geminiState.deceptionScore !== undefined ? geminiState.deceptionScore : null;

      // Uncertainty calculations:
      // Uncertainty is highest around 0.5 (1 - 2*|score - 0.5|)
      let faceUncertainty = 1.0;
      if (facePLie !== null) {
        faceUncertainty = Math.max(0, 1.0 - 2 * Math.abs(facePLie - 0.5));
      }

      let voiceUncertainty = 1.0;
      if (voiceScore !== null) {
        voiceUncertainty = Math.max(0, 1.0 - 2 * Math.abs(voiceScore - 0.5));
      }

      let geminiUncertainty = 1.0;
      if (geminiScore !== null) {
        geminiUncertainty = Math.max(0, 1.0 - 2 * Math.abs(geminiScore - 0.5));
      }

      // Collect active modalities
      const activeScores = [];
      const activeUncertainties = [];
      const modalNames = [];

      if (faceState.ready && facePLie !== null) {
        activeScores.push({ name: 'Face', score: facePLie, uncertainty: faceUncertainty, weight: 0.35 });
        activeUncertainties.push(faceUncertainty);
        modalNames.push('Face');
      }

      if (voiceState.ready && voiceScore !== null) {
        activeScores.push({ name: 'Voice', score: voiceScore, uncertainty: voiceUncertainty, weight: 0.35 });
        activeUncertainties.push(voiceUncertainty);
        modalNames.push('Voice');
      }

      if (geminiState.ready && geminiScore !== null) {
        activeScores.push({ name: 'Evidence (Gemini)', score: geminiScore, uncertainty: geminiUncertainty, weight: 0.30 });
        activeUncertainties.push(geminiUncertainty);
        modalNames.push('Evidence');
      }

      let crossVerdict = 'UNCERTAIN';
      let crossDeceptionScore = null;
      let narrative = '';

      // Check user condition: if active modules say uncertain > 0.5 it is uncertain
      const allActiveUncertain = activeUncertainties.length > 0 && activeUncertainties.every(u => u > 0.5);

      if (activeScores.length >= 2) {
        if (allActiveUncertain) {
          crossVerdict = 'UNCERTAIN';
          const sum = activeScores.reduce((acc, m) => acc + m.score, 0);
          crossDeceptionScore = sum / activeScores.length;
          narrative = `Cross-Modal Uncertainty Confirmed: All active modalities (${modalNames.join(', ')}) report ambiguous indicators (>50% uncertainty floor). Neither truth nor deception can be certified definitively.`;
        } else {
          // Normalize weights across active modalities
          const totalWeight = activeScores.reduce((acc, m) => acc + m.weight, 0);
          crossDeceptionScore = activeScores.reduce((acc, m) => acc + (m.score * (m.weight / totalWeight)), 0);

          if (crossDeceptionScore >= 0.58) {
            crossVerdict = 'DECEPTIVE';
            narrative = `Multimodal Elevated Deception Risk (${modalNames.join(' + ')}): Concurrently high probability of inconsistency or deceptive cues (Deception Score: ${(crossDeceptionScore * 100).toFixed(0)}%).`;
          } else if (crossDeceptionScore <= 0.42) {
            crossVerdict = 'TRUTHFUL';
            narrative = `Consensus Truthful Baseline (${modalNames.join(' + ')}): Verified agreement across sensory cues and evidence statements (Truth Probability: ${((1.0 - crossDeceptionScore) * 100).toFixed(0)}%).`;
          } else {
            crossVerdict = 'UNCERTAIN';
            narrative = `Modality Divergence: Divergent indicators between ${modalNames.join(', ')}. Cross-modal composite score is in the intermediate zone (${(crossDeceptionScore * 100).toFixed(0)}%).`;
          }
        }
      } else if (activeScores.length === 1) {
        const single = activeScores[0];
        crossDeceptionScore = single.score;
        if (single.uncertainty > 0.5) {
          crossVerdict = 'UNCERTAIN';
          narrative = `${single.name} telemetry active with high uncertainty (${(single.uncertainty * 100).toFixed(0)}%). Awaiting additional modalities for cross-verification.`;
        } else if (crossDeceptionScore >= 0.58) {
          crossVerdict = 'DECEPTIVE';
          narrative = `${single.name} analysis indicates elevated deceptive cues (${(crossDeceptionScore * 100).toFixed(0)}%). Provide additional modalities to confirm consensus.`;
        } else if (crossDeceptionScore <= 0.42) {
          crossVerdict = 'TRUTHFUL';
          narrative = `${single.name} analysis indicates baseline credibility (${((1 - crossDeceptionScore) * 100).toFixed(0)}% truth).`;
        } else {
          crossVerdict = 'UNCERTAIN';
          narrative = `${single.name} active in intermediate range.`;
        }
      } else {
        crossVerdict = 'STANDBY';
        narrative = 'System awaiting live inputs. Position face in front of webcam, speak into microphone, and load case evidence to begin concurrent multimodal analysis.';
      }

      this.overviewCorrelationText.textContent = narrative;

      if (badge) {
        badge.textContent = crossVerdict;
        badge.className = 'overview-verdict ' + (crossVerdict === 'TRUTHFUL' ? 'truth' : (crossVerdict === 'DECEPTIVE' ? 'lie' : 'uncertain'));
      }

      if (scoreVal) {
        if (crossDeceptionScore !== null && crossVerdict !== 'STANDBY') {
          scoreVal.textContent = `Score: ${(crossDeceptionScore * 100).toFixed(0)}%`;
        } else {
          scoreVal.textContent = 'Score: —';
        }
      }
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
    if (this.geminiModule && this.geminiModule.recording) {
      this.geminiModule.stopInterview();
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
    if (this.geminiModule) {
      this.geminiModule.reset();
    }
  }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.trusightApp = new TrusightDashboard();
});
