/**
 * TRUSIGHT Gemini Statement & Evidence Verification Module
 * Connects to port 8003 via the Gateway proxy (/api/gemini & /ws/gemini).
 * Handles PDF case upload, FAISS RAG indexing, live PCM audio streaming,
 * interim/final transcripts, and live statement contradiction verification.
 */

class GeminiAnalysisModule {
  constructor() {
    this.caseId = null;
    this.caseReport = null;
    this.recording = false;
    this.socket = null;
    this.stream = null;
    this.audioContext = null;
    this.audioSource = null;
    this.audioProcessor = null;
    this.silentGain = null;

    // State
    this.state = {
      ready: false,
      caseLoaded: false,
      verdict: 'STANDBY',
      confidence: null,
      deceptionScore: null, // 0 to 1 scale for cross-modal
      rawScore: null,        // 0 to 100
      scoreType: '',
      transcript: '',
      interimTranscript: '',
      explanation: null,
      evidence: [],
      error: null,
    };

    // DOM Elements
    this.pdfInput = document.getElementById('geminiPdfInput');
    this.btnUploadCase = document.getElementById('btnGeminiUploadCase');
    this.caseStatus = document.getElementById('geminiCaseStatus');
    this.caseInfo = document.getElementById('geminiCaseInfo');

    this.btnStart = document.getElementById('btnGeminiStart');
    this.btnStop = document.getElementById('btnGeminiStop');
    this.micStatus = document.getElementById('geminiMicStatus');

    this.verdictBadge = document.getElementById('geminiVerdictBadge');
    this.scoreVal = document.getElementById('geminiScoreVal');
    this.scoreFill = document.getElementById('geminiScoreFill');
    this.confidenceVal = document.getElementById('geminiConfidenceVal');

    this.transcriptBox = document.getElementById('geminiTranscriptBox');
    this.evidenceBox = document.getElementById('geminiEvidenceBox');
    this.explanationBox = document.getElementById('geminiExplanationBox');

    this.init();
  }

  init() {
    if (this.btnUploadCase) {
      this.btnUploadCase.addEventListener('click', () => this.uploadCase());
    }
    if (this.pdfInput) {
      this.pdfInput.addEventListener('change', () => {
        if (this.pdfInput.files && this.pdfInput.files[0]) {
          this.uploadCase();
        }
      });
    }

    if (this.btnStart) {
      this.btnStart.addEventListener('click', () => this.startInterview());
    }
    if (this.btnStop) {
      this.btnStop.addEventListener('click', () => this.stopInterview());
    }
  }

  async uploadCase() {
    if (!this.pdfInput || !this.pdfInput.files || !this.pdfInput.files[0]) {
      alert('Please select a case evidence PDF first.');
      return;
    }

    const file = this.pdfInput.files[0];
    if (this.caseStatus) {
      this.caseStatus.textContent = `⏳ Indexing "${file.name}" with Gemini FAISS RAG...`;
      this.caseStatus.className = 'gemini-status-text processing';
    }
    if (this.micStatus) {
      this.micStatus.textContent = 'Indexing PDF... please wait';
      this.micStatus.className = 'gemini-status-text processing';
    }
    if (this.btnUploadCase) this.btnUploadCase.disabled = true;

    const formData = new FormData();
    formData.append('pdf', file, file.name);

    try {
      const res = await fetch('/api/gemini/case', {
        method: 'POST',
        body: formData,
      });

      let data;
      try {
        data = await res.json();
      } catch (parseErr) {
        data = { detail: `HTTP ${res.status}: Server returned an invalid response.` };
      }

      if (!res.ok) {
        throw new Error(data.detail || `Server returned error HTTP ${res.status}.`);
      }

      if (!data.case_id) {
        throw new Error('Server response missing valid case ID.');
      }

      this.caseId = data.case_id;
      this.caseReport = data;
      this.state.caseLoaded = true;

      if (this.caseStatus) {
        this.caseStatus.textContent = `✅ Indexed: ${file.name}`;
        this.caseStatus.className = 'gemini-status-text success';
      }
      if (this.caseInfo) {
        this.caseInfo.textContent = `Pages: ${data.pages || '—'} | Chunks: ${data.chunks || '—'}`;
      }
      if (this.micStatus) {
        this.micStatus.textContent = 'Ready! Click "Start Interview" to speak.';
        this.micStatus.className = 'gemini-status-text success';
      }
      if (this.btnStart) {
        this.btnStart.disabled = false;
        this.btnStart.classList.remove('active-stop');
        this.btnStart.classList.add('active-start');
      }
    } catch (err) {
      console.error('Case upload error:', err);
      const isNetworkErr = err.name === 'TypeError' || err.message === 'Failed to fetch';
      const displayMsg = isNetworkErr
        ? 'Could not reach TRUSIGHT server. Verify the Gateway is online.'
        : err.message;
      if (this.caseStatus) {
        this.caseStatus.textContent = `❌ ${displayMsg}`;
        this.caseStatus.className = 'gemini-status-text error';
      }
      if (this.micStatus) {
        this.micStatus.textContent = 'Case indexing failed. Check server console.';
        this.micStatus.className = 'gemini-status-text error';
      }
      if (this.btnStart) {
        this.btnStart.disabled = true;
      }
    } finally {
      if (this.btnUploadCase) this.btnUploadCase.disabled = false;
    }
  }

  async startInterview() {
    if (!this.caseId) {
      alert('Please upload and index a case report PDF first.');
      return;
    }

    try {
      if (this.micStatus) {
        this.micStatus.textContent = 'Requesting audio input stream...';
        this.micStatus.className = 'gemini-status-text processing';
      }

      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/gemini/audio/${this.caseId}`;

      this.socket = new WebSocket(wsUrl);
      this.socket.binaryType = 'arraybuffer';

      this.socket.onopen = () => {
        this.recording = true;
        this.state.ready = true;
        if (this.micStatus) {
          this.micStatus.textContent = '🎙️ Live Gemini audio channel active. Speak naturally...';
          this.micStatus.className = 'gemini-status-text success';
        }
        if (this.btnStart) this.btnStart.disabled = true;
        if (this.btnStop) this.btnStop.disabled = false;
        this.startPCMStream();
      };

      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleServerMessage(data);
        } catch (e) {
          console.error('Invalid JSON from Gemini WS:', e);
        }
      };

      this.socket.onerror = (err) => {
        console.error('Gemini WS error:', err);
        if (this.micStatus) {
          this.micStatus.textContent = 'WebSocket connection error.';
          this.micStatus.className = 'gemini-status-text error';
        }
      };

      this.socket.onclose = () => {
        this.cleanupAudio();
      };
    } catch (err) {
      console.error('Mic error:', err);
      if (this.micStatus) {
        this.micStatus.textContent = `Microphone error: ${err.message}`;
        this.micStatus.className = 'gemini-status-text error';
      }
    }
  }

  startPCMStream() {
    if (!this.recording || !this.stream || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return;
    }

    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioCtx();
      this.audioSource = this.audioContext.createMediaStreamSource(this.stream);
      this.audioProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
      this.silentGain = this.audioContext.createGain();
      this.silentGain.gain.value = 0;

      this.audioProcessor.onaudioprocess = (event) => {
        if (!this.recording || !this.socket || this.socket.readyState !== WebSocket.OPEN) return;
        const input = event.inputBuffer.getChannelData(0);
        const ratio = this.audioContext.sampleRate / 16000;
        const outputLength = Math.floor(input.length / ratio);
        const pcm = new Int16Array(outputLength);

        for (let i = 0; i < outputLength; i++) {
          const sourceIdx = Math.min(input.length - 1, Math.floor(i * ratio));
          const sample = Math.max(-1, Math.min(1, input[sourceIdx]));
          pcm[i] = sample < 0 ? sample * 32768 : sample * 32767;
        }

        if (pcm.length) {
          this.socket.send(pcm.buffer);
        }
      };

      this.audioSource.connect(this.audioProcessor);
      this.audioProcessor.connect(this.silentGain);
      this.silentGain.connect(this.audioContext.destination);
    } catch (err) {
      console.error('PCM stream setup failed:', err);
    }
  }

  handleServerMessage(data) {
    if (data.status === 'error') {
      if (this.micStatus) {
        this.micStatus.textContent = `Error: ${data.message || 'Verification issue'}`;
        this.micStatus.className = 'gemini-status-text error';
      }
      return;
    }

    // 1. Interim Transcript
    if (data.interim_transcript) {
      this.state.interimTranscript = data.interim_transcript;
      if (this.transcriptBox) {
        const full = this.state.transcript ? `${this.state.transcript}\n\n` : '';
        this.transcriptBox.textContent = `${full}… ${data.interim_transcript}`;
        this.transcriptBox.scrollTop = this.transcriptBox.scrollHeight;
      }
    }

    // 2. Finalized Transcript Segment
    if (data.transcript) {
      this.state.transcript = (this.state.transcript ? `${this.state.transcript}\n\n` : '') + data.transcript;
      if (this.transcriptBox) {
        this.transcriptBox.textContent = this.state.transcript;
        this.transcriptBox.scrollTop = this.transcriptBox.scrollHeight;
      }
    }

    // 3. Verdict, Score & Verification
    if (data.status === 'success') {
      const v = data.verdict || 'Unknown';
      this.state.verdict = v;
      this.state.rawScore = Number(data.deception_score) || 0;
      if (data.inconsistency_index !== undefined && data.inconsistency_index !== null) {
        this.state.deceptionScore = Math.max(0, Math.min(1, Number(data.inconsistency_index)));
      } else {
        this.state.deceptionScore = Math.max(0, Math.min(1, this.state.rawScore / 100.0));
      }
      this.state.confidence = data.confidence !== undefined ? data.confidence : null;
      this.state.scoreType = data.score_type || '';
      this.state.timestamp = (data.timestamp ? parseFloat(data.timestamp) : Date.now() / 1000.0);
      this.state.explanation = data.explanation || null;
      this.state.evidence = data.evidence || [];
      this.state.evidenceResults = data.evidence_results || data.evidence || [];

      // Update UI badges
      if (this.verdictBadge) {
        const lowerV = v.toLowerCase();
        let displayBadge = v.toUpperCase();
        let badgeClass = 'uncertain';

        if (lowerV.includes('consistent') || lowerV.includes('entailment') || lowerV.includes('supported')) {
          displayBadge = 'SUPPORTED';
          badgeClass = 'truth';
        } else if (lowerV.includes('inconsistency') || lowerV.includes('contradict')) {
          displayBadge = 'CONTRADICTED';
          badgeClass = 'lie';
        } else if (lowerV.includes('mixed')) {
          displayBadge = 'MIXED';
          badgeClass = 'uncertain';
        } else if (lowerV.includes('no statement')) {
          displayBadge = 'NO STATEMENT';
          badgeClass = 'uncertain';
        } else if (lowerV.includes('no evidence') || lowerV.includes('insufficient')) {
          displayBadge = 'INSUFFICIENT EVIDENCE';
          badgeClass = 'uncertain';
        }

        this.verdictBadge.textContent = displayBadge;
        this.verdictBadge.className = 'metric-badge ' + badgeClass;
      }

      if (this.scoreVal) {
        this.scoreVal.textContent = `Index: ${this.state.deceptionScore.toFixed(2)}`;
      }
      if (this.scoreFill) {
        this.scoreFill.style.width = `${Math.round(this.state.deceptionScore * 100)}%`;
        this.scoreFill.className = 'meter-fill ' + (
          this.state.deceptionScore >= 0.5 ? 'lie' : this.state.deceptionScore > 0 ? 'truth' : 'uncertain'
        );
      }
      if (this.confidenceVal && this.state.confidence !== null) {
        this.confidenceVal.textContent = `Confidence: ${Math.round(this.state.confidence)}%`;
      }

      // Update Evidence box with structured source/page references
      if (this.evidenceBox && data.evidence) {
        if (Array.isArray(data.evidence) && data.evidence.length > 0) {
          this.evidenceBox.innerHTML = data.evidence.map((item, idx) => {
            const page = item.page !== undefined && item.page !== null ? `Page ${item.page}` : 'Doc Source';
            const txt = typeof item === 'string' ? item : (item.text || item.chunk || JSON.stringify(item));
            return `<div class="gemini-evidence-item"><span class="evidence-tag">REF #${idx + 1} • ${escapeHtml(page)}</span> ${escapeHtml(txt)}</div>`;
          }).join('');
        } else {
          this.evidenceBox.textContent = 'No contradicting evidence retrieved.';
        }
      }

      // Update Explanation box with structured forensic parsing
      if (this.explanationBox && data.explanation) {
        this.renderExplanation(data.explanation, data.verdict);
      }
    }
  }

  renderExplanation(explanationData, verdict) {
    if (!this.explanationBox) return;

    let parsed = explanationData;
    if (typeof parsed === 'string') {
      try {
        parsed = JSON.parse(parsed);
      } catch (e) {
        // Plain text fallback
        this.explanationBox.innerHTML = `
          <div class="contradiction-summary-card">
            <div class="contradiction-summary-title"><span>📋</span> Forensic Assessment Summary</div>
            <div>${parsed}</div>
          </div>
        `;
        return;
      }
    }

    if (!parsed || typeof parsed !== 'object') {
      this.explanationBox.textContent = 'No contradiction analysis generated.';
      return;
    }

    const summary = parsed.summary || 'Evidence comparison completed.';
    const conflicting = Array.isArray(parsed.conflicting_elements) ? parsed.conflicting_elements : [];
    const interviewStmt = parsed.interview_statement || '';
    const evidenceStmt = parsed.evidence_statement || '';

    let html = `<div class="contradiction-container">`;

    // 1. Summary Card
    html += `
      <div class="contradiction-summary-card">
        <div class="contradiction-summary-title">
          <span>⚖️</span> Forensic Finding Summary
        </div>
        <div>${summary}</div>
      </div>
    `;

    // 2. Conflicting Elements (if explicit conflicts found)
    if (conflicting.length > 0) {
      html += `
        <div class="conflict-list-header">
          <span>Detected Fact Contradictions (${conflicting.length})</span>
          <span style="color:var(--lie);font-size:10px;">EXPLICIT DIVERGENCE</span>
        </div>
      `;

      conflicting.forEach((item, idx) => {
        const type = (item.type || 'Fact').toUpperCase();
        const page = item.page !== undefined && item.page !== null ? `Page ${item.page}` : 'Case Doc';
        const interviewVal = item.interview || '—';
        const evidenceVal = item.evidence || '—';

        html += `
          <div class="conflict-item-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span class="conflict-type-badge">⚠️ ${type} INCONSISTENCY</span>
              <span style="font-size:10px;font-family:var(--font-mono);color:var(--text-muted);">${page}</span>
            </div>
            <div class="conflict-comparison-grid">
              <div class="conflict-cell">
                <div class="conflict-cell-label">
                  <span>Spoken Statement</span>
                  <span style="color:#38BDF8;">Audio</span>
                </div>
                <div class="conflict-cell-content">"${interviewVal}"</div>
              </div>
              <div class="conflict-cell">
                <div class="conflict-cell-label">
                  <span>Documented Record</span>
                  <span style="color:#A78BFA;">Evidence</span>
                </div>
                <div class="conflict-cell-content">"${evidenceVal}"</div>
              </div>
            </div>
          </div>
        `;
      });
    } else {
      // No explicit conflict elements
      const isConsistent = verdict && (verdict.toLowerCase().includes('truth') || verdict.toLowerCase().includes('entail'));
      html += `
        <div class="conflict-item-card ${isConsistent ? 'supported' : ''}">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span class="conflict-type-badge ${isConsistent ? 'supported' : ''}">
              ${isConsistent ? '✓ EVIDENCE CONSISTENT' : 'ℹ️ NO EXPLICIT CONFLICT DETECTED'}
            </span>
          </div>
          <div style="font-size:11px;color:var(--text-secondary);margin-top:4px;line-height:1.4;">
            ${isConsistent
          ? 'The spoken statement aligns with the factual baseline documented in the indexed case report.'
          : 'The excerpt neither directly confirms nor explicitly refutes the statement (insufficient counter-evidence).'}
          </div>
        </div>
      `;
    }

    html += `</div>`;
    this.explanationBox.innerHTML = html;
  }

  stopInterview() {
    this.recording = false;
    this.cleanupAudio();
    if (this.socket) {
      try { this.socket.close(); } catch (e) { }
      this.socket = null;
    }
    if (this.micStatus) {
      this.micStatus.textContent = 'Interview stopped.';
      this.micStatus.className = 'gemini-status-text';
    }
    if (this.btnStart) this.btnStart.disabled = false;
    if (this.btnStop) this.btnStop.disabled = true;
  }

  cleanupAudio() {
    if (this.audioProcessor) {
      try { this.audioProcessor.disconnect(); } catch (e) { }
      this.audioProcessor = null;
    }
    if (this.audioSource) {
      try { this.audioSource.disconnect(); } catch (e) { }
      this.audioSource = null;
    }
    if (this.silentGain) {
      try { this.silentGain.disconnect(); } catch (e) { }
      this.silentGain = null;
    }
    if (this.audioContext) {
      try { this.audioContext.close(); } catch (e) { }
      this.audioContext = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
  }

  getState() {
    return {
      ready: this.state.ready,
      caseLoaded: this.state.caseLoaded,
      verdict: this.state.verdict,
      deceptionScore: this.state.deceptionScore,
      inconsistencyIndex: this.state.deceptionScore,
      rawScore: this.state.rawScore,
      confidence: this.state.confidence,
      transcript: this.state.transcript,
      evidence: this.state.evidence,
      evidenceResults: this.state.evidenceResults || this.state.evidence || [],
      explanation: this.state.explanation,
    };
  }

  reset() {
    this.stopInterview();
    this.state.transcript = '';
    this.state.verdict = 'STANDBY';
    this.state.deceptionScore = null;
    this.state.rawScore = null;
    this.state.confidence = null;
    this.state.evidence = [];
    this.state.evidenceResults = [];
    this.state.explanation = null;

    if (this.transcriptBox) this.transcriptBox.textContent = 'Waiting for speech...';
    if (this.evidenceBox) this.evidenceBox.textContent = 'Awaiting statement analysis to retrieve evidence...';
    if (this.explanationBox) this.explanationBox.textContent = 'No explanation generated yet.';
    if (this.verdictBadge) {
      this.verdictBadge.textContent = 'STANDBY';
      this.verdictBadge.className = 'metric-badge uncertain';
    }
    if (this.scoreVal) this.scoreVal.textContent = '—';
    if (this.scoreFill) this.scoreFill.style.width = '0%';
    if (this.confidenceVal) this.confidenceVal.textContent = 'Confidence: —';
  }
}

window.GeminiAnalysisModule = GeminiAnalysisModule;
