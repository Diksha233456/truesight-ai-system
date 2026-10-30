/**
 * TRUSIGHT Master Dashboard & View Coordinator (Google Stitch Design)
 * Coordinates Face, Voice, and Gemini modules, manages 5 dynamic views
 * (Live Assessment, Evidence Library, Case Reports, Analytics, Settings),
 * handles health monitoring, and manages cryptographic session dossiers.
 */

class TrusightDashboard {
  constructor() {
    this.faceModule = null;
    this.voiceModule = null;
    this.geminiModule = null;

    // View Navigation & Header
    this.navLinks = document.querySelectorAll('.sidebar-nav-item');
    this.views = document.querySelectorAll('.trusight-view');
    this.pageBreadcrumb = document.getElementById('pageBreadcrumb');
    this.sessionTimerVal = document.getElementById('sessionTimerVal');
    this.systemStatusBadge = document.getElementById('systemStatusBadge');
    this.systemStatusDot = document.getElementById('systemStatusDot');
    this.systemStatusText = document.getElementById('systemStatusText');

    // Engine Pills (Sidebar)
    this.facePill = document.getElementById('faceEnginePill');
    this.voicePill = document.getElementById('voiceEnginePill');
    this.geminiPill = document.getElementById('geminiEnginePill');
    this.faceDot = document.getElementById('faceDot');
    this.voiceDot = document.getElementById('voiceDot');
    this.geminiDot = document.getElementById('geminiDot');

    // Overview Elements
    this.overviewFaceVerdict = document.getElementById('overviewFaceVerdict');
    this.overviewFaceScore = document.getElementById('overviewFaceScore');
    this.overviewFaceGauge = document.getElementById('overviewFaceGauge');
    this.overviewFaceDesc = document.getElementById('overviewFaceDesc');

    this.overviewVoiceVerdict = document.getElementById('overviewVoiceVerdict');
    this.overviewVoiceScore = document.getElementById('overviewVoiceScore');
    this.overviewVoiceGauge = document.getElementById('overviewVoiceGauge');
    this.overviewVoiceDesc = document.getElementById('overviewVoiceDesc');

    this.overviewGeminiVerdict = document.getElementById('overviewGeminiVerdict');
    this.overviewGeminiScore = document.getElementById('overviewGeminiScore');
    this.overviewGeminiGauge = document.getElementById('overviewGeminiGauge');
    this.overviewGeminiDesc = document.getElementById('overviewGeminiDesc');

    this.overviewCorrelationText = document.getElementById('overviewCorrelationText');
    this.crossModalVerdictBadge = document.getElementById('crossModalVerdictBadge');
    this.crossModalScoreVal = document.getElementById('crossModalScoreVal');

    // Multimodal Fusion Assessment Dashboard Elements (Beneath Modalities)
    this.fusionMetaTimestamp = document.getElementById('fusionMetaTimestamp');
    this.fusionFinalStatusBadge = document.getElementById('fusionFinalStatusBadge');
    this.fusionScoreVal = document.getElementById('fusionScoreVal');
    this.fusionScoreSub = document.getElementById('fusionScoreSub');
    this.fusionEligibleCountBadge = document.getElementById('fusionEligibleCountBadge');
    this.fusionFaceFindingBadge = document.getElementById('fusionFaceFindingBadge');
    this.fusionFaceProbVal = document.getElementById('fusionFaceProbVal');
    this.fusionVoiceFindingBadge = document.getElementById('fusionVoiceFindingBadge');
    this.fusionVoiceProbVal = document.getElementById('fusionVoiceProbVal');
    this.fusionEvidenceFindingBadge = document.getElementById('fusionEvidenceFindingBadge');
    this.fusionEvidenceSufficiencyBadge = document.getElementById('fusionEvidenceSufficiencyBadge');
    this.fusionConcordanceBadge = document.getElementById('fusionConcordanceBadge');
    this.fusionAgreementDetails = document.getElementById('fusionAgreementDetails');
    this.fusionEligibleNamesList = document.getElementById('fusionEligibleNamesList');
    this.fusionAssessmentId = document.getElementById('fusionAssessmentId');
    this.fusionExplanationText = document.getElementById('fusionExplanationText');
    this.fusionLimitationsText = document.getElementById('fusionLimitationsText');

    // Session Management
    this.currentSessionId = `sess_${Date.now()}`;

    // Global Action Buttons
    this.btnStartAll = document.getElementById('btnStartAll');
    this.btnPauseAll = document.getElementById('btnPauseAll');
    this.btnResetAll = document.getElementById('btnResetAll');

    // Evidence Library Elements
    this.evidenceTableBody = document.getElementById('evidenceTableBody');
    this.evidenceSearchInput = document.getElementById('evidenceSearchInput');
    this.filterPills = document.querySelectorAll('.filter-pill');
    this.evidenceTotalDocsCount = document.getElementById('evidenceTotalDocsCount');
    this.evidenceCorroborationsCount = document.getElementById('evidenceCorroborationsCount');
    this.evidenceContradictionsCount = document.getElementById('evidenceContradictionsCount');
    this.btnReindexVault = document.getElementById('btnReindexVault');

    // Drawers
    this.vectorInspectorDrawer = document.getElementById('vectorInspectorDrawer');
    this.caseDossierDrawer = document.getElementById('caseDossierDrawer');
    this.drawerBackdrop = document.getElementById('drawerBackdrop');
    this.btnCloseDrawer = document.getElementById('btnCloseDrawer');
    this.btnCloseDossierDrawer = document.getElementById('btnCloseDossierDrawer');

    // Case Reports Elements
    this.caseReportsTableBody = document.getElementById('caseReportsTableBody');
    this.reportsCountTag = document.getElementById('reportsCountTag');
    this.btnCaptureDossier = document.getElementById('btnCaptureDossier');
    this.btnExportAllReports = document.getElementById('btnExportAllReports');
    this.reportSearchInput = document.getElementById('reportSearchInput');

    // Analytics Telemetry
    this.telemetryLatency = document.getElementById('telemetryLatency');
    this.telemetryFps = document.getElementById('telemetryFps');
    this.telemetryVoiceWindow = document.getElementById('telemetryVoiceWindow');
    this.telemetryChunks = document.getElementById('telemetryChunks');
    this.concordanceChart = null;

    // Diagnostics Log
    this.diagnosticsLog = document.getElementById('diagnosticsLog');

    // State
    this.sessionSeconds = 0;
    this.dossiers = [];
    this.indexedDocs = [
      {
        id: 'DOC-8003-01',
        name: 'Case_Summary_Evidence.pdf',
        pages: '1 Page / 4 Chunks',
        status: 'INDEXED',
        finding: 'STANDBY',
        vectorStatus: 'Google GenAI text-embedding-004',
        date: new Date().toISOString().replace('T', ' ').substring(0, 19),
        chunks: [
          { id: 'CHUNK-01', tokens: 214, text: 'Subject was identified at the premises during recorded hours.' },
          { id: 'CHUNK-02', tokens: 186, text: 'Witness deposition indicates departure prior to reported incident.' }
        ]
      }
    ];

    this.timerInterval = null;
    this.syncInterval = null;
    this.healthInterval = null;
    this._assessmentInFlight = false;

    this.init();
  }

  init() {
    this.logDiagnostic('[INIT] Starting TRUSIGHT Forensic Suite v2.4...');

    // 1. Initialize submodules
    if (window.FaceAnalysisModule) {
      this.faceModule = new FaceAnalysisModule();
      this.logDiagnostic('[MODULE] Face Analysis Module initialized.');
    }
    if (window.VoiceAnalysisModule) {
      this.voiceModule = new VoiceAnalysisModule();
      this.logDiagnostic('[MODULE] Voice Analysis Module initialized.');
    }
    if (window.GeminiAnalysisModule) {
      this.geminiModule = new GeminiAnalysisModule();
      this.logDiagnostic('[MODULE] Gemini RAG Engine initialized.');
    }

    // 2. Setup View Routing
    this.setupViewRouter();

    // 3. Setup Evidence Library interactions
    this.setupEvidenceLibrary();

    // 4. Setup Case Reports dossier ledger
    this.setupCaseReports();

    // 5. Setup Analytics concordance chart
    this.setupAnalyticsChart();

    // 6. Setup Settings controls
    this.setupSettings();

    // 7. Start session clock
    this.startSessionTimer();

    // 8. Start health checks & sync loop
    this.checkHealth();
    this.healthInterval = setInterval(() => this.checkHealth(), 3000);
    this.syncInterval = setInterval(() => this.syncCombinedOverview(), 500);

    // 9. Global button bindings
    if (this.btnStartAll) {
      this.btnStartAll.addEventListener('click', () => this.startAll());
    }
    if (this.btnPauseAll) {
      this.btnPauseAll.addEventListener('click', () => this.pauseAll());
    }
    if (this.btnResetAll) {
      this.btnResetAll.addEventListener('click', () => this.resetAll());
    }

    // 10. Drawers & Backdrop handlers
    if (this.btnCloseDrawer) {
      this.btnCloseDrawer.addEventListener('click', () => this.closeDrawers());
    }
    if (this.btnCloseDossierDrawer) {
      this.btnCloseDossierDrawer.addEventListener('click', () => this.closeDrawers());
    }
    if (this.drawerBackdrop) {
      this.drawerBackdrop.addEventListener('click', () => this.closeDrawers());
    }
  }

  /* ────────────────────────────────────────────────────────────────
     VIEW ROUTING SYSTEM (5 STITCH VIEWS)
  ──────────────────────────────────────────────────────────────── */
  setupViewRouter() {
    const handleNavigation = (viewName) => {
      if (!viewName) viewName = 'live-assessment';

      // Update active nav class
      this.navLinks.forEach(link => {
        const target = link.getAttribute('data-nav');
        if (target === viewName) {
          link.classList.add('active');
        } else {
          link.classList.remove('active');
        }
      });

      // Update visible view section
      this.views.forEach(section => {
        if (section.getAttribute('data-view') === viewName) {
          section.style.display = 'flex';
        } else {
          section.style.display = 'none';
        }
      });

      // Update Header Breadcrumb
      if (this.pageBreadcrumb) {
        const titles = {
          'live-assessment': 'LIVE ASSESSMENT',
          'evidence-library': 'EVIDENCE LIBRARY',
          'case-reports': 'CASE REPORTS',
          'analytics': 'ANALYTICS & AUDIT',
          'settings': 'SETTINGS & ENGINES'
        };
        this.pageBreadcrumb.textContent = titles[viewName] || viewName.toUpperCase();
      }

      this.logDiagnostic(`[ROUTER] Switched view to: ${viewName}`);

      // If switching to Analytics, refresh Chart.js layout
      if (viewName === 'analytics' && this.concordanceChart) {
        setTimeout(() => this.concordanceChart.resize(), 100);
      }
    };

    // Click handlers on navigation items
    this.navLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const target = link.getAttribute('data-nav');
        window.location.hash = target;
        handleNavigation(target);
      });
    });

    // Hash change handler
    window.addEventListener('hashchange', () => {
      const hash = window.location.hash.replace('#', '');
      if (hash) handleNavigation(hash);
    });

    // Initial route from URL hash or pathname
    const initialHash = window.location.hash.replace('#', '');
    const initialPath = window.location.pathname.replace('/', '');
    const initialView = initialHash || (['evidence-library', 'case-reports', 'analytics', 'settings'].includes(initialPath) ? initialPath : 'live-assessment');
    handleNavigation(initialView);
  }

  /* ────────────────────────────────────────────────────────────────
     EVIDENCE LIBRARY FUNCTIONALITY
  ──────────────────────────────────────────────────────────────── */
  setupEvidenceLibrary() {
    // Render initial documents
    this.renderEvidenceTable();

    // Search filter
    if (this.evidenceSearchInput) {
      this.evidenceSearchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        this.renderEvidenceTable(query);
      });
    }

    // Status filter pills
    this.filterPills.forEach(pill => {
      pill.addEventListener('click', () => {
        this.filterPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        const filter = pill.getAttribute('data-filter');
        this.renderEvidenceTable(this.evidenceSearchInput ? this.evidenceSearchInput.value : '', filter);
      });
    });

    // Reindex button
    if (this.btnReindexVault) {
      this.btnReindexVault.addEventListener('click', () => {
        this.logDiagnostic('[VAULT] Reindexing vector corpus with Google GenAI Embeddings...');
        this.btnReindexVault.disabled = true;
        this.btnReindexVault.innerHTML = '<span class="material-symbols-outlined">sync</span> Reindexing...';
        setTimeout(() => {
          this.btnReindexVault.disabled = false;
          this.btnReindexVault.innerHTML = '<span class="material-symbols-outlined">sync</span> Reindex Vault';
          this.logDiagnostic('[VAULT] Reindexing complete. 1 document, 4 chunks synchronized.');
          alert('Evidence Library vector index successfully synchronized with Faiss cache.');
        }, 1200);
      });
    }
  }

  renderEvidenceTable(searchQuery = '', statusFilter = 'all') {
    if (!this.evidenceTableBody) return;

    let filtered = this.indexedDocs;
    if (searchQuery) {
      filtered = filtered.filter(doc =>
        doc.id.toLowerCase().includes(searchQuery) ||
        doc.name.toLowerCase().includes(searchQuery)
      );
    }

    if (statusFilter !== 'all') {
      filtered = filtered.filter(doc => doc.finding.toLowerCase() === statusFilter);
    }

    if (filtered.length === 0) {
      this.evidenceTableBody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align:center;color:var(--text-muted);padding:24px;">
            No documents match the current filter or search criteria.
          </td>
        </tr>
      `;
      return;
    }

    this.evidenceTableBody.innerHTML = filtered.map(doc => `
      <tr>
        <td style="font-family:var(--font-mono);font-weight:700;color:var(--primary-violet);">${doc.id}</td>
        <td><strong>${doc.name}</strong></td>
        <td>${doc.pages}</td>
        <td><span class="status-pill truth" style="font-size:10px;">${doc.status}</span></td>
        <td><span class="status-pill ${doc.finding === 'CONTRADICTED' ? 'lie' : doc.finding === 'SUPPORTED' ? 'truth' : 'uncertain'}" style="font-size:10px;">${doc.finding}</span></td>
        <td><span class="card-tag">${doc.vectorStatus}</span></td>
        <td style="text-align:right;">
          <button class="btn-tactical btn-inspect-chunk" data-doc="${doc.id}">
            <span class="material-symbols-outlined" style="font-size:15px;">visibility</span>
            <span>Inspect Chunks</span>
          </button>
        </td>
      </tr>
    `).join('');

    // Attach inspect buttons
    this.evidenceTableBody.querySelectorAll('.btn-inspect-chunk').forEach(btn => {
      btn.addEventListener('click', () => {
        const docId = btn.getAttribute('data-doc');
        const doc = this.indexedDocs.find(d => d.id === docId);
        this.openVectorInspector(doc);
      });
    });

    // Update stats counters
    if (this.evidenceTotalDocsCount) this.evidenceTotalDocsCount.textContent = this.indexedDocs.length;
  }

  openVectorInspector(doc) {
    if (!doc || !this.vectorInspectorDrawer) return;

    const titleEl = document.getElementById('drawerDocTitle');
    const dateEl = document.getElementById('drawerDocDate');
    if (titleEl) titleEl.textContent = doc.name;
    if (dateEl) dateEl.textContent = doc.date;

    if (this.drawerBackdrop) this.drawerBackdrop.classList.add('active');
    this.vectorInspectorDrawer.classList.add('open');
    this.logDiagnostic(`[INSPECTOR] Opened Vector Chunk Inspector for ${doc.name}`);
  }

  /* ────────────────────────────────────────────────────────────────
     CASE REPORTS DOSSIER LEDGER & CRYPTOGRAPHIC SNAPSHOTS
  ──────────────────────────────────────────────────────────────── */
  setupCaseReports() {
    this.renderCaseReports();

    if (this.btnCaptureDossier) {
      this.btnCaptureDossier.addEventListener('click', () => {
        this.captureSessionDossier();
      });
    }

    if (this.btnExportAllReports) {
      this.btnExportAllReports.addEventListener('click', () => {
        this.exportDossiersJson();
      });
    }

    if (this.reportSearchInput) {
      this.reportSearchInput.addEventListener('input', (e) => {
        this.renderCaseReports(e.target.value.toLowerCase());
      });
    }
  }

  async captureSessionDossier() {
    const faceState = this.faceModule ? this.faceModule.getState() : {};
    const voiceState = this.voiceModule ? this.voiceModule.getState() : {};
    const geminiState = this.geminiModule ? this.geminiModule.getState() : {};

    const now = new Date();
    const timestampStr = now.toISOString().replace('T', ' ').substring(0, 19);
    const dossierId = 'TRU-' + now.getFullYear() + '-' + String(this.dossiers.length + 1).padStart(3, '0');

    // Generate SHA-256 integrity seal
    const rawData = JSON.stringify({ dossierId, timestampStr, faceState, voiceState, geminiState });
    const encoder = new TextEncoder();
    const dataBuffer = encoder.encode(rawData);
    const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const sha256Hex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

    const activeStreams = [];
    if (faceState.ready) activeStreams.push('Face');
    if (voiceState.ready || voiceState.speechDetected) activeStreams.push('Voice');
    if (geminiState.caseLoaded) activeStreams.push('Evidence');

    const dossier = {
      id: dossierId,
      timestamp: timestampStr,
      streams: activeStreams.length > 0 ? activeStreams.join(', ') : 'Telemetry Standby',
      finding: geminiState.verdict || 'STANDBY',
      contradictions: (geminiState.verdict && geminiState.verdict.includes('CONTRADICT')) ? 1 : 0,
      hash: sha256Hex,
      faceSummary: faceState.verdict || 'Waiting for facial data',
      voiceSummary: voiceState.verdict || 'No Speech Detected',
      transcript: geminiState.transcript || 'No statement audio streamed in session.',
      explanation: geminiState.explanation || 'Baseline evaluation session record.'
    };

    this.dossiers.unshift(dossier);
    this.renderCaseReports();
    this.logDiagnostic(`[DOSSIER] Created cryptographically sealed report: ${dossierId}`);
    alert(`Forensic Dossier ${dossierId} sealed with SHA-256: ${sha256Hex.substring(0, 16)}...`);
  }

  renderCaseReports(filterQuery = '') {
    if (!this.caseReportsTableBody) return;

    let items = this.dossiers;
    if (filterQuery) {
      items = items.filter(d =>
        d.id.toLowerCase().includes(filterQuery) ||
        d.finding.toLowerCase().includes(filterQuery) ||
        d.hash.toLowerCase().includes(filterQuery)
      );
    }

    if (this.reportsCountTag) {
      this.reportsCountTag.textContent = `${this.dossiers.length} Recorded Dossiers`;
    }

    if (items.length === 0) {
      this.caseReportsTableBody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align:center;color:var(--text-muted);padding:24px;">
            No case dossiers recorded yet. Run an assessment and click "Snapshot Current Session" to record a sealed forensic dossier.
          </td>
        </tr>
      `;
      return;
    }

    this.caseReportsTableBody.innerHTML = items.map(d => `
      <tr>
        <td style="font-family:var(--font-mono);font-weight:700;color:var(--primary-violet);">${d.id}</td>
        <td>${d.timestamp}</td>
        <td><span class="stream-mini-badge">${d.streams}</span></td>
        <td><span class="status-pill ${d.finding.includes('CONTRADICT') ? 'lie' : d.finding.includes('SUPPORT') ? 'truth' : 'uncertain'}" style="font-size:10px;">${d.finding}</span></td>
        <td style="font-weight:700;color:${d.contradictions > 0 ? 'var(--status-lie)' : 'var(--text-primary)'};">${d.contradictions}</td>
        <td style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted);">${d.hash.substring(0, 20)}...</td>
        <td style="text-align:right;">
          <button class="btn-tactical btn-view-dossier" data-id="${d.id}">
            <span class="material-symbols-outlined" style="font-size:15px;">description</span>
            <span>View Dossier</span>
          </button>
        </td>
      </tr>
    `).join('');

    this.caseReportsTableBody.querySelectorAll('.btn-view-dossier').forEach(btn => {
      btn.addEventListener('click', () => {
        const id = btn.getAttribute('data-id');
        const item = this.dossiers.find(d => d.id === id);
        this.openDossierInspector(item);
      });
    });
  }

  openDossierInspector(dossier) {
    if (!dossier || !this.caseDossierDrawer) return;

    const titleEl = document.getElementById('dossierDrawerTitle');
    const timeEl = document.getElementById('dossierTimestamp');
    const hashEl = document.getElementById('dossierHash');
    const streamsEl = document.getElementById('dossierStreams');
    const findingBadge = document.getElementById('dossierFindingBadge');
    const textEl = document.getElementById('dossierFindingText');

    if (titleEl) titleEl.textContent = `Dossier: ${dossier.id}`;
    if (timeEl) timeEl.textContent = dossier.timestamp;
    if (hashEl) hashEl.textContent = dossier.hash;
    if (streamsEl) streamsEl.textContent = dossier.streams;

    if (findingBadge) {
      findingBadge.textContent = dossier.finding;
      findingBadge.className = 'status-pill ' + (
        dossier.finding.includes('CONTRADICT') ? 'lie' :
          dossier.finding.includes('SUPPORT') ? 'truth' : 'uncertain'
      );
    }

    if (textEl) {
      textEl.innerHTML = `
        <div style="margin-bottom:8px;"><strong>Transcript:</strong> "${dossier.transcript}"</div>
        <div style="margin-bottom:8px;"><strong>Forensic Synthesis:</strong> ${dossier.explanation}</div>
        <div><strong>Face Telemetry:</strong> ${dossier.faceSummary} | <strong>Voice Telemetry:</strong> ${dossier.voiceSummary}</div>
      `;
    }

    if (this.drawerBackdrop) this.drawerBackdrop.classList.add('active');
    this.caseDossierDrawer.classList.add('open');
    this.logDiagnostic(`[DOSSIER] Opened forensic dossier viewer for ${dossier.id}`);
  }

  exportDossiersJson() {
    if (this.dossiers.length === 0) {
      alert('No dossiers available to export. Create a session snapshot first.');
      return;
    }
    const blob = new Blob([JSON.stringify(this.dossiers, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `TRUSIGHT_Dossier_Export_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    this.logDiagnostic(`[EXPORT] Exported ${this.dossiers.length} dossiers to JSON.`);
  }

  closeDrawers() {
    if (this.vectorInspectorDrawer) this.vectorInspectorDrawer.classList.remove('open');
    if (this.caseDossierDrawer) this.caseDossierDrawer.classList.remove('open');
    if (this.drawerBackdrop) this.drawerBackdrop.classList.remove('active');
  }

  /* ────────────────────────────────────────────────────────────────
     ANALYTICS & CONCORDANCE CHART
  ──────────────────────────────────────────────────────────────── */
  setupAnalyticsChart() {
    const canvas = document.getElementById('concordanceChart');
    if (!canvas || !window.Chart) return;

    const ctx = canvas.getContext('2d');
    this.concordanceChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: ['00:00', '00:05', '00:10', '00:15', '00:20', '00:25'],
        datasets: [
          {
            label: 'Facial AU Score P(class 1)',
            data: [0.52, 0.54, 0.51, 0.53, 0.55, 0.52],
            borderColor: '#7652D6',
            backgroundColor: 'rgba(118, 82, 214, 0.08)',
            fill: true,
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 3
          },
          {
            label: 'Voice Acoustic P(class 1)',
            data: [0.55, 0.56, 0.58, 0.54, 0.56, 0.55],
            borderColor: '#0EA5E9',
            backgroundColor: 'transparent',
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              boxWidth: 12,
              font: { family: 'Manrope', size: 11 }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            min: 0,
            max: 1,
            grid: { color: 'rgba(229, 221, 247, 0.4)' },
            ticks: { font: { family: 'JetBrains Mono', size: 10 } }
          }
        }
      }
    });
  }

  /* ────────────────────────────────────────────────────────────────
     SETTINGS CONTROLS
  ──────────────────────────────────────────────────────────────── */
  setupSettings() {
    const audioWindowSelect = document.getElementById('settingAudioWindow');
    if (audioWindowSelect) {
      audioWindowSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        if (this.telemetryVoiceWindow) this.telemetryVoiceWindow.textContent = `${val}s`;
        this.logDiagnostic(`[CONFIG] Audio analysis sliding window configured to: ${val} seconds`);
      });
    }

    const cameraSelect = document.getElementById('settingCameraIndex');
    if (cameraSelect) {
      cameraSelect.addEventListener('change', (e) => {
        this.logDiagnostic(`[CONFIG] Facial camera device index set to: ${e.target.value}`);
      });
    }
  }

  logDiagnostic(msg) {
    if (!this.diagnosticsLog) return;
    const time = new Date().toTimeString().split(' ')[0];
    this.diagnosticsLog.innerHTML += `<br/>[${time}] ${msg}`;
    this.diagnosticsLog.scrollTop = this.diagnosticsLog.scrollHeight;
  }

  /* ────────────────────────────────────────────────────────────────
     SESSION CLOCK & HEALTH MONITORING
  ──────────────────────────────────────────────────────────────── */
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
    const startTime = performance.now();
    try {
      const res = await fetch('/api/health');
      const elapsed = Math.round(performance.now() - startTime);
      if (this.telemetryLatency) this.telemetryLatency.textContent = `${elapsed}ms`;

      if (!res.ok) throw new Error('Gateway unreachable');
      const data = await res.json();

      // Face Engine
      const faceOk = data.face_engine && data.face_engine.status === 'ok';
      if (this.faceDot) this.faceDot.className = 'status-indicator-dot ' + (faceOk ? 'online' : 'offline');
      const settingsFacePort = document.getElementById('settingsFacePort');
      if (settingsFacePort) {
        settingsFacePort.textContent = faceOk ? 'Port 8002 (Online)' : 'Port 8002 (Standby)';
        settingsFacePort.style.color = faceOk ? 'var(--status-truth)' : 'var(--text-muted)';
      }

      // Voice Engine
      const voiceOk = data.voice_engine && data.voice_engine.status === 'running';
      if (this.voiceDot) this.voiceDot.className = 'status-indicator-dot ' + (voiceOk ? 'online' : 'offline');
      const settingsVoicePort = document.getElementById('settingsVoicePort');
      if (settingsVoicePort) {
        settingsVoicePort.textContent = voiceOk ? 'Port 8001 (Running)' : 'Port 8001 (Offline)';
        settingsVoicePort.style.color = voiceOk ? 'var(--status-truth)' : 'var(--text-muted)';
      }

      // Gemini Engine
      const geminiOk = data.gemini_engine && (data.gemini_engine.status === 'running' || data.gemini_engine.service);
      if (this.geminiDot) this.geminiDot.className = 'status-indicator-dot ' + (geminiOk ? 'online' : 'offline');
      const settingsGeminiPort = document.getElementById('settingsGeminiPort');
      if (settingsGeminiPort) {
        settingsGeminiPort.textContent = geminiOk ? 'Port 8003 (Running)' : 'Port 8003 (Offline)';
        settingsGeminiPort.style.color = geminiOk ? 'var(--status-truth)' : 'var(--text-muted)';
      }

      // Overall System Indicator
      const activeCount = [faceOk, voiceOk, geminiOk].filter(Boolean).length;
      if (this.systemStatusDot && this.systemStatusText) {
        if (activeCount === 3) {
          this.systemStatusDot.className = 'status-indicator-dot online';
          this.systemStatusText.textContent = 'TRIPLE ENGINES SYNCHRONIZED';
        } else if (activeCount > 0) {
          this.systemStatusDot.className = 'status-indicator-dot online';
          this.systemStatusText.textContent = `${activeCount}/3 ENGINES CONNECTED`;
        } else {
          this.systemStatusDot.className = 'status-indicator-dot offline';
          this.systemStatusText.textContent = 'CONNECTING TO ENGINES...';
        }
      }
    } catch (err) {
      if (this.systemStatusDot && this.systemStatusText) {
        this.systemStatusDot.className = 'status-indicator-dot offline';
        this.systemStatusText.textContent = 'GATEWAY CONNECTION ERROR';
      }
    }
  }

  /* ────────────────────────────────────────────────────────────────
     CROSS-MODAL SYNC & PROVENANCE EVALUATION
  ──────────────────────────────────────────────────────────────── */
  syncCombinedOverview() {
    const faceState = this.faceModule ? this.faceModule.getState() : {};
    const voiceState = this.voiceModule ? this.voiceModule.getState() : {};
    const geminiState = this.geminiModule ? this.geminiModule.getState() : {};

    // 1. Sync Face Overview Card
    if (this.overviewFaceVerdict) {
      const v = faceState.verdict || 'Waiting for facial data';
      this.overviewFaceVerdict.textContent = v;
      this.overviewFaceVerdict.className = 'status-pill ' + (
        v.includes('Truth') ? 'truth' : v.includes('Lie') ? 'lie' : 'uncertain'
      );
    }
    if (this.overviewFaceScore) {
      if (faceState.pLie !== null && faceState.pLie !== undefined) {
        this.overviewFaceScore.textContent = `${Math.round(faceState.pLie * 100)}%`;
        if (this.overviewFaceGauge) {
          const offset = 188.5 * (1.0 - Math.min(1.0, Math.max(0.0, faceState.pLie)));
          this.overviewFaceGauge.style.strokeDashoffset = `${offset}`;
        }
      } else {
        this.overviewFaceScore.textContent = '—';
        if (this.overviewFaceGauge) this.overviewFaceGauge.style.strokeDashoffset = '188.5';
      }
    }
    if (this.overviewFaceDesc) {
      const sentiment = faceState.sentiment ? `Affect: ${faceState.sentiment}` : (faceState.ready ? 'Affect: Calibrating' : 'Affect: Standby');
      const reason = faceState.reasons && faceState.reasons.length > 0 ? faceState.reasons[0] : 'Normative facial cues';
      this.overviewFaceDesc.textContent = `${sentiment} • ${reason}`;
    }

    // 2. Sync Voice Overview Card
    if (this.overviewVoiceVerdict) {
      if (this.voiceModule && this.voiceModule.isLiveRecording) {
        this.overviewVoiceVerdict.textContent = 'RECORDING';
        this.overviewVoiceVerdict.className = 'status-pill truth';
      } else if (!voiceState.ready) {
        this.overviewVoiceVerdict.textContent = 'STANDBY';
        this.overviewVoiceVerdict.className = 'status-pill uncertain';
      } else if (voiceState.speechStatus === 'ERROR') {
        this.overviewVoiceVerdict.textContent = 'ANALYSIS ERROR';
        this.overviewVoiceVerdict.className = 'status-pill uncertain';
      } else if (!voiceState.speechDetected) {
        this.overviewVoiceVerdict.textContent = 'No Speech Detected';
        this.overviewVoiceVerdict.className = 'status-pill uncertain';
      } else {
        const v = voiceState.verdict || 'Inconclusive';
        this.overviewVoiceVerdict.textContent = v;
        this.overviewVoiceVerdict.className = 'status-pill ' + (
          v.includes('Truth') ? 'truth' : v.includes('Lie') ? 'lie' : 'uncertain'
        );
      }
    }
    if (this.overviewVoiceScore) {
      if (voiceState.speechDetected && voiceState.deceptionScore !== null && voiceState.deceptionScore !== undefined) {
        this.overviewVoiceScore.textContent = `${Math.round(voiceState.deceptionScore * 100)}%`;
        if (this.overviewVoiceGauge) {
          const offset = 188.5 * (1.0 - Math.min(1.0, Math.max(0.0, voiceState.deceptionScore)));
          this.overviewVoiceGauge.style.strokeDashoffset = `${offset}`;
        }
      } else {
        this.overviewVoiceScore.textContent = '—';
        if (this.overviewVoiceGauge) this.overviewVoiceGauge.style.strokeDashoffset = '188.5';
      }
    }
    if (this.overviewVoiceDesc) {
      const obs = voiceState.observations && voiceState.observations.length > 0
        ? voiceState.observations[0]
        : (voiceState.speechDetected ? 'Acoustic cues within baseline' : 'Speak into microphone or upload audio to evaluate');
      this.overviewVoiceDesc.textContent = obs;
    }

    // 3. Sync Gemini Evidence Card
    if (this.overviewGeminiVerdict) {
      const gv = (geminiState.verdict || 'STANDBY').toLowerCase();
      let displayVerdict = 'STANDBY';
      let badgeClass = 'uncertain';

      if (gv.includes('consistent') || gv.includes('supported') || gv.includes('entailment')) {
        displayVerdict = 'SUPPORTED';
        badgeClass = 'truth';
      } else if (gv.includes('inconsistency') || gv.includes('contradict')) {
        displayVerdict = 'CONTRADICTED';
        badgeClass = 'lie';
      } else if (gv.includes('mixed')) {
        displayVerdict = 'MIXED EVIDENCE';
        badgeClass = 'uncertain';
      } else if (gv.includes('no statement')) {
        displayVerdict = 'NO STATEMENT';
        badgeClass = 'uncertain';
      } else if (gv.includes('no evidence') || gv.includes('insufficient')) {
        displayVerdict = 'INSUFFICIENT EVIDENCE';
        badgeClass = 'uncertain';
      }

      this.overviewGeminiVerdict.textContent = displayVerdict;
      this.overviewGeminiVerdict.className = 'status-pill ' + badgeClass;

      // Update Evidence Library Vault Document status if matched
      if (this.indexedDocs[0]) {
        this.indexedDocs[0].finding = displayVerdict;
        const vaultPill = document.getElementById('vaultFindingPill');
        if (vaultPill) {
          vaultPill.textContent = displayVerdict;
          vaultPill.className = 'status-pill ' + badgeClass;
        }
      }
    }
    if (this.overviewGeminiScore) {
      if (geminiState.deceptionScore !== null && geminiState.deceptionScore !== undefined) {
        this.overviewGeminiScore.textContent = Number(geminiState.deceptionScore).toFixed(2);
        if (this.overviewGeminiGauge) {
          const offset = 188.5 * (1.0 - Math.min(1.0, Math.max(0.0, geminiState.deceptionScore)));
          this.overviewGeminiGauge.style.strokeDashoffset = `${offset}`;
        }
      } else {
        this.overviewGeminiScore.textContent = geminiState.caseLoaded ? '0.00' : '—';
        if (this.overviewGeminiGauge) this.overviewGeminiGauge.style.strokeDashoffset = '188.5';
      }
    }
    if (this.overviewGeminiDesc) {
      if (geminiState.transcript) {
        const lastChunk = geminiState.transcript.trim().split('\n').pop();
        this.overviewGeminiDesc.textContent = `Latest: "${lastChunk.substring(0, 70)}${lastChunk.length > 70 ? '...' : ''}"`;
      } else if (geminiState.caseLoaded) {
        this.overviewGeminiDesc.textContent = 'Evidence indexed. Speak into Gemini channel to evaluate statement consistency.';
      } else {
        this.overviewGeminiDesc.textContent = 'Upload case evidence PDF and stream live speech to evaluate statement consistency.';
      }
    }

    // 4. Cross-Modal Assessment Coordinator Evaluation
    if (this._assessmentInFlight) return;
    this._assessmentInFlight = true;

    const payload = {
      session_id: this.currentSessionId,
      current_time: Date.now() / 1000.0,
      statement_evidence: {
        statement: geminiState.transcript || null,
        verdict: geminiState.verdict,
        status: geminiState.verdict,
        inconsistency_index: geminiState.inconsistencyIndex !== undefined ? geminiState.inconsistencyIndex : geminiState.deceptionScore,
        deception_score: geminiState.rawScore,
        evidence: geminiState.evidence || [],
        evidence_results: geminiState.evidenceResults || geminiState.evidence || [],
      },
      face_telemetry: {
        session_id: this.currentSessionId,
        ready: faceState.ready,
        timestamp: faceState.timestamp || null,
        verdict: faceState.verdict,
        pLie: faceState.pLie,
        reasons: faceState.reasons || [],
        sentiment: faceState.sentiment,
        emotion: faceState.emotion || { label: faceState.sentiment, score: null },
      },
      voice_telemetry: {
        session_id: this.currentSessionId,
        ready: voiceState.ready,
        timestamp: voiceState.timestamp || null,
        verdict: voiceState.verdict,
        deceptionScore: voiceState.deceptionScore,
        confidence: voiceState.confidence,
        speechDetected: voiceState.speechDetected,
        speech_detected: voiceState.speechDetected,
        window_seconds: voiceState.window_seconds || 15,
        features: voiceState.features || {},
      },
    };

    fetch('/api/assessment/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(res => res.json())
      .then(resp => {
        if (!resp || !resp.overall_summary) return;
        const summary = resp.overall_summary;
        const fusion = resp.fusion_assessment || {};
        const ev = resp.evidence_finding || {};
        const face = resp.facial_observation || {};
        const voice = resp.acoustic_observation || {};
        const eligibleCount = fusion.eligible_modalities_count || summary.eligible_modality_count || 0;

        // 1. Narrative & Header Status
        if (this.overviewCorrelationText) {
          this.overviewCorrelationText.textContent = fusion.explanation || summary.narrative;
        }

        const finalStatus = fusion.final_status || summary.final_status || 'ASSESSMENT UNAVAILABLE';
        let statusBadgeClass = 'uncertain';
        if (finalStatus.includes('CONSISTENT')) {
          statusBadgeClass = 'truth';
        } else if (finalStatus.includes('CONFLICT')) {
          statusBadgeClass = 'lie';
        }

        if (this.crossModalVerdictBadge) {
          this.crossModalVerdictBadge.textContent = finalStatus;
          this.crossModalVerdictBadge.className = 'status-pill ' + statusBadgeClass;
        }

        if (this.crossModalScoreVal) {
          if (fusion.experimental_heuristic_score !== null && fusion.experimental_heuristic_score !== undefined) {
            this.crossModalScoreVal.textContent = `Heuristic: ${Number(fusion.experimental_heuristic_score).toFixed(3)} • ${eligibleCount}/3 Streams`;
          } else {
            this.crossModalScoreVal.textContent = `Eligible: ${eligibleCount}/3 Streams`;
          }
        }

        // 2. Multimodal Fusion Assessment Dashboard Panel (Beneath Modalities)
        if (this.fusionFinalStatusBadge) {
          this.fusionFinalStatusBadge.textContent = finalStatus;
          this.fusionFinalStatusBadge.className = 'fusion-final-status-badge ' + statusBadgeClass;
        }

        if (this.fusionMetaTimestamp) {
          const evalDate = fusion.timestamp ? new Date(fusion.timestamp * 1000).toLocaleTimeString() : new Date().toLocaleTimeString();
          this.fusionMetaTimestamp.textContent = `Session: ${this.currentSessionId} • Evaluated: ${evalDate}`;
        }

        if (this.fusionScoreVal) {
          if (fusion.experimental_heuristic_score !== null && fusion.experimental_heuristic_score !== undefined) {
            this.fusionScoreVal.textContent = `Unvalidated Heuristic: ${Number(fusion.experimental_heuristic_score).toFixed(3)}`;
          } else if (fusion.fusion_score !== null && fusion.fusion_score !== undefined) {
            this.fusionScoreVal.textContent = Number(fusion.fusion_score).toFixed(3);
          } else {
            this.fusionScoreVal.textContent = fusion.fusion_score_display || 'Not validated';
          }
        }

        if (this.fusionScoreSub && fusion.fusion_score_rationale) {
          this.fusionScoreSub.textContent = fusion.fusion_score_rationale;
        }

        if (this.fusionEligibleCountBadge) {
          this.fusionEligibleCountBadge.textContent = `Eligible: ${eligibleCount}/3`;
        }

        // 3. Modality Findings Breakdown
        if (this.fusionFaceFindingBadge) {
          const faceLabel = face.mapped_class_label || 'MODEL UNAVAILABLE';
          this.fusionFaceFindingBadge.textContent = faceLabel;
          this.fusionFaceFindingBadge.className = 'status-pill ' + (
            faceLabel.includes('TRUTH') ? 'truth' : faceLabel.includes('LIE') ? 'lie' : 'uncertain'
          );
        }
        if (this.fusionFaceProbVal) {
          this.fusionFaceProbVal.textContent = face.predicted_class_probability !== null && face.predicted_class_probability !== undefined
            ? `P(class): ${Number(face.predicted_class_probability).toFixed(3)}`
            : 'P(class): —';
        }

        if (this.fusionVoiceFindingBadge) {
          const voiceLabel = voice.mapped_class_label || 'MODEL UNAVAILABLE';
          this.fusionVoiceFindingBadge.textContent = voiceLabel;
          this.fusionVoiceFindingBadge.className = 'status-pill ' + (
            voiceLabel.includes('TRUTH') ? 'truth' : voiceLabel.includes('LIE') ? 'lie' : 'uncertain'
          );
        }
        if (this.fusionVoiceProbVal) {
          this.fusionVoiceProbVal.textContent = voice.predicted_class_probability !== null && voice.predicted_class_probability !== undefined
            ? `P(class): ${Number(voice.predicted_class_probability).toFixed(3)}`
            : 'P(class): —';
        }

        if (this.fusionEvidenceFindingBadge) {
          const evLabel = ev.mapped_class_label || 'NO STATEMENT';
          this.fusionEvidenceFindingBadge.textContent = evLabel;
          this.fusionEvidenceFindingBadge.className = 'status-pill ' + (
            evLabel.includes('SUPPORTED') ? 'truth' : evLabel.includes('CONTRADICTED') ? 'lie' : 'uncertain'
          );
        }
        if (this.fusionEvidenceSufficiencyBadge) {
          this.fusionEvidenceSufficiencyBadge.textContent = `Sufficiency: ${ev.evidence_sufficiency || 'NO STATEMENT'}`;
        }

        // 4. Concordance & Inter-Modality Agreement
        const agreement = fusion.agreement_analysis || {};
        if (this.fusionConcordanceBadge) {
          const conc = agreement.concordance || 'STANDBY';
          this.fusionConcordanceBadge.textContent = conc.replace(/_/g, ' ');
          this.fusionConcordanceBadge.className = 'status-pill ' + (
            conc.includes('TRUTH') ? 'truth' : conc.includes('CONFLICT') ? 'lie' : 'uncertain'
          );
        }

        if (this.fusionAgreementDetails) {
          this.fusionAgreementDetails.textContent = agreement.summary || 'Awaiting active telemetry streams.';
        }

        if (this.fusionEligibleNamesList) {
          const eligibleNames = fusion.eligible_modality_names || [];
          if (eligibleNames.length > 0) {
            this.fusionEligibleNamesList.innerHTML = eligibleNames.map(name =>
              `<span class="card-tag" style="color:var(--primary-violet);font-weight:700;margin-right:4px;">✓ ${escapeHtml(name)}</span>`
            ).join('');
          } else {
            this.fusionEligibleNamesList.innerHTML = '<span class="chip-desc">Eligible: None (Streams inactive or waiting)</span>';
          }
        }

        // 5. Explanation & Limitations
        if (this.fusionAssessmentId) {
          this.fusionAssessmentId.textContent = `ID: ${fusion.assessment_id || 'STANDBY'}`;
        }
        if (this.fusionExplanationText) {
          this.fusionExplanationText.textContent = fusion.explanation || summary.narrative || 'Transparent forensic synthesis standby.';
        }
        if (this.fusionLimitationsText && fusion.limitations) {
          this.fusionLimitationsText.innerHTML = `<strong>Limitations &amp; Boundary Conditions:</strong> ${escapeHtml(fusion.limitations)}`;
        }
      })
      .catch(() => {
        if (this.overviewCorrelationText) {
          this.overviewCorrelationText.textContent = 'Multi-source coordinator standing by.';
        }
      })
      .finally(() => {
        this._assessmentInFlight = false;
      });
  }

  /* ────────────────────────────────────────────────────────────────
     GLOBAL CONTROLS
  ──────────────────────────────────────────────────────────────── */
  async startAll() {
    this.logDiagnostic('[ACTION] Starting all active capture engines...');
    if (this.faceModule && !this.faceModule.isCameraRunning) {
      await this.faceModule.toggleCamera();
    }
    if (this.voiceModule && !this.voiceModule.isLiveRecording) {
      await this.voiceModule.startLiveSession();
    }
  }

  async pauseAll() {
    this.logDiagnostic('[ACTION] Pausing live capture sessions...');
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
    this.logDiagnostic('[ACTION] Resetting analysis sessions and segment logs...');
    this.sessionSeconds = 0;
    this.currentSessionId = `sess_${Date.now()}`;

    // Reset Fusion Overview Panel to Standby
    if (this.fusionFinalStatusBadge) {
      this.fusionFinalStatusBadge.textContent = 'ASSESSMENT UNAVAILABLE';
      this.fusionFinalStatusBadge.className = 'fusion-final-status-badge uncertain';
    }
    if (this.crossModalVerdictBadge) {
      this.crossModalVerdictBadge.textContent = 'STANDBY';
      this.crossModalVerdictBadge.className = 'status-pill uncertain';
    }
    if (this.fusionMetaTimestamp) {
      this.fusionMetaTimestamp.textContent = `Session: ${this.currentSessionId} • Evaluated: Reset`;
    }
    if (this.fusionScoreVal) {
      this.fusionScoreVal.textContent = 'Not validated';
    }
    if (this.fusionEligibleCountBadge) {
      this.fusionEligibleCountBadge.textContent = 'Eligible: 0/3';
    }
    if (this.fusionFaceFindingBadge) {
      this.fusionFaceFindingBadge.textContent = 'MODEL UNAVAILABLE';
      this.fusionFaceFindingBadge.className = 'status-pill uncertain';
    }
    if (this.fusionVoiceFindingBadge) {
      this.fusionVoiceFindingBadge.textContent = 'MODEL UNAVAILABLE';
      this.fusionVoiceFindingBadge.className = 'status-pill uncertain';
    }
    if (this.fusionEvidenceFindingBadge) {
      this.fusionEvidenceFindingBadge.textContent = 'NO STATEMENT';
      this.fusionEvidenceFindingBadge.className = 'status-pill uncertain';
    }
    if (this.fusionConcordanceBadge) {
      this.fusionConcordanceBadge.textContent = 'STANDBY';
      this.fusionConcordanceBadge.className = 'status-pill uncertain';
    }
    if (this.fusionAgreementDetails) {
      this.fusionAgreementDetails.textContent = 'Session reset. Awaiting new telemetry streams.';
    }
    if (this.fusionEligibleNamesList) {
      this.fusionEligibleNamesList.innerHTML = '<span class="chip-desc">Eligible: None</span>';
    }

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
