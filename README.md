TRUSIGHT --- Multimodal Forensic Intelligence Platform
An integrated AI platform for facial, voice, and text-based evidence
analysis.

TRUSIGHT is a multimodal investigative intelligence platform that brings
together facial cue analysis, vocal acoustic analysis, and text-based
evidence analysis through a unified dashboard. It helps users examine
different forms of evidence in one interface, combining machine-learning
outputs, extracted observations, and structured explanations to support
investigative review.
Important: TRUSIGHT is an investigative support tool, not a
definitive lie detector. Facial expressions, vocal characteristics,
and textual patterns can be influenced by many factors and cannot
independently establish deception. Results should be reviewed
alongside source evidence and appropriate validation.

Table of Contents
- Overview
- Core Modules
  - Facial Analysis
  - Voice Analysis
  - Text-Based Evidence Analysis
- Unified Dashboard
- Technology Stack
- Project Structure
- Local Execution
- How TRUSIGHT Works
- Responsible Use and Limitations
Overview
TRUSIGHT combines three complementary analysis modules:
- Face Analysis: Processes webcam video to extract facial
  landmarks, facial action cues, emotion classifications, head pose,
  gaze information, and model-generated truth/lie predictions.
- Voice Analysis: Examines audio recordings and live microphone
  input using acoustic features such as pitch, jitter, shimmer,
  pauses, speech rate, and intensity. A Random Forest model generates
  a deception-related score.
- Text-Based Evidence Analysis: Supports case-level review of
  textual evidence. Depending on the configured pipeline, it can use
  language models and retrieval-augmented generation (RAG) to examine
  supplied material and produce structured explanations.
The modules are accessible through a unified dashboard. Their outputs
can be examined independently and compared as part of a broader evidence
review.
Core Modules
A. Facial Analysis
- Live webcam video processing and facial-cue visualization.
- Facial landmark detection using MediaPipe.
- Emotion classification using the configured facial emotion model.
- Facial action-unit visualization and head-pose estimation.
- Model-generated truth/lie classification and associated scores.
- Camera start and stop controls within the dashboard.
B. Voice Analysis
- Live microphone recording and audio-file upload.
- Acoustic feature extraction using Parselmouth and Librosa.
- Pitch and fundamental-frequency statistics.
- Jitter and shimmer measurements related to vocal stability.
- Speech tempo, pauses, hesitation-related observations, and RMS
  intensity.
- Random Forest-based deception scoring.
- Structured explanations of acoustic observations and signal-quality
  warnings.
C. Text-Based Evidence Analysis
- Analysis of text supplied as investigative evidence.
- Document-based evidence processing and case indexing, where
  supported by the configured pipeline.
- Retrieval-augmented generation (RAG) to retrieve relevant
  information from indexed material.
- Language-model-assisted analysis and structured explanations.
- Case-oriented evidence organization to support review alongside
  facial and voice outputs.
The text module complements the other analysis components. Its output
should be checked against the original source material and should not be
treated as automatic proof that a statement is true or false.
Unified Dashboard
The TRUSIGHT dashboard provides a central interface for accessing the
available analysis modules:
- Face: Webcam feed, facial observations, and model outputs.
- Voice: Live and uploaded audio analysis, acoustic measurements,
  and model outputs.
- Text: Textual evidence analysis and supported document-based
  workflows.
- Cross-Modal Review: Side-by-side presentation of available
  results to help users compare observations across modalities.
TRUSIGHT presents outputs from separate analysis pipelines. It does
not imply a scientifically validated combined deception verdict
unless a fusion method has been implemented and appropriately evaluated.
Technology Stack
  Component                           Technologies
  Backend API                         Python, FastAPI

  Frontend                            HTML, CSS, JavaScript

  Facial analysis                     OpenCV, MediaPipe, configured
                                      deep-learning models

  Voice analysis                      Parselmouth, Librosa, scikit-learn

  Text analysis                       Configured language-model clients,
                                      retrieval-augmented generation
                                      (RAG)

  Data processing                     Acoustic feature extraction,
                                      facial-cue extraction, document and
                                      text processing

  Communication                       REST APIs, WebSockets, live video
                                      streaming

  Local deployment                    Python services and a unified
                                  localhost dashboard
Exact model providers, endpoints, and dependencies depend on each
module's configuration.
Project Structure
The following is a high-level view of the project. Exact files and
optional modules may vary by configuration.
TRUSIGHT/
├── trusight_launcher.py                  # Coordinates local application services
├── trusight_server.py                    # Unified FastAPI gateway
├── requirements.txt                      # Root-level Python dependencies
├── static/
│   ├── index.html                        # Dashboard interface
│   ├── css/
│   │   └── dashboard.css                 # Dashboard styling
│   └── js/
│       ├── dashboard.js                  # Dashboard coordination
│       ├── face_module.js                # Face module frontend
│       ├── voice_module.js               # Voice module frontend
│       └── gemini_module.js              # Text/LLM module frontend, if configured
├── RealTime-Facial-Expression-Recognition/
│   └── AU/
│       └── facial_cues/                  # Facial analysis service and pipeline
├── truesight-voice-analysis/
│   └── voice_service/
│       └── app/                          # Voice analysis API and processing
└── TruSight_Gemini_API/
    └── TruSight_Gemini_API/               # Text analysis, LLM, and RAG components
Local Execution
The unified dashboard is designed to run locally at:

http://localhost:8000
Option 1: Run with Python
From the project root, execute:
python trusight_launcher.py
Option 2: Windows startup script
If the repository includes and configures the startup script, run:
.\run_trusight.bat
The launcher is intended to coordinate the configured dashboard,
facial-analysis, and voice-analysis services. Text-analysis services and
external model clients must also be configured according to their own
requirements.
Configuration Notes
- Install the dependencies required by each service.
- Configure any required API credentials through environment
  variables.
- Keep real .env files and API keys out of version control.
- Follow the setup instructions in the relevant module's README, if
  available.
- Confirm that the required local ports are available before starting
  the application.
The exact installation commands and required environment variables
depend on the configuration of the checked-out project. Consult the
module documentation for service-specific setup.
How TRUSIGHT Works
1. Provide evidence: Start the webcam or provide audio, text, or
   supported documents.
2. Process each modality: The configured face, voice, and text
   pipelines extract their respective signals or content.
3. Generate module outputs: Each pipeline returns its own
   predictions, observations, scores, or structured explanations.
4. Review results: Examine the outputs in the dashboard and compare
   them with the original evidence.
5. Validate conclusions: Use source verification and human review
   before drawing investigative conclusions.
Responsible Use and Limitations
- A model score is an output of the configured model, not a guarantee
  of truth or deception.
- Facial expressions and vocal characteristics are affected by
  context, individual differences, recording conditions, and other
  factors.
- Language-model outputs may be incomplete or incorrect and should be
  verified against source documents.
- Do not treat a single modality---or an unvalidated combination of
  modalities---as conclusive evidence of deception.
- Use appropriate privacy, consent, data-protection, and
  access-control practices when handling audio, video, and case
  documents.
Project Objective
TRUSIGHT aims to provide a unified environment for reviewing facial,
vocal, and textual evidence with AI-assisted tools. By bringing these
complementary modalities into one interface, the platform seeks to make
evidence review more organized, interpretable, and accessible while
keeping human judgment and source verification central.
