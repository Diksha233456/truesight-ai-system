# TRUESIGHT — Voice Analysis Service

TRUESIGHT is an experimental voice analysis service that uses acoustic features and machine learning to examine speech patterns and generate explanations.

## Features

- **Audio analysis:** Analyze speech recordings using acoustic features.
- **Voice feature extraction:** Examine characteristics such as pitch, pauses, and other speech patterns.
- **Machine learning:** Use a trained model to produce experimental predictions.
- **Explanation engine:** Provide explanations alongside analysis results.
- **API backend:** Built with FastAPI.
- **Web interface:** A browser-based frontend for interacting with the service.

## Tech Stack

- Python
- FastAPI
- Librosa and Parselmouth
- Scikit-learn
- HTML, CSS, and JavaScript

## Project Structure

```text
truesight-voice-analysis/
├── .gitignore
└── voice_service/
    ├── app/
    │   ├── models/
    │   ├── static/
    │   ├── main.py
    │   ├── predictor.py
    │   ├── feature_extractor.py
    │   └── explanation.py
    ├── docs/
    ├── tests/
    ├── requirements.txt
    ├── Dockerfile
    ├── README.md
    └── setup.bat
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Diksha233456/truesight-voice-analysis.git
cd truesight-voice-analysis/voice_service
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the service

Check `app/main.py` for the FastAPI application entry point, then run:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` in your browser. If the application uses a different route for its interface, follow the instructions in `voice_service/README.md`.

## Documentation

Additional project documentation is available in `voice_service/docs/`, including the dataset card, model card, evaluation protocol, and model audit.

## Important Limitations

TRUESIGHT is an experimental research project. Acoustic patterns cannot establish whether a person is lying, and predictions should not be treated as proof of deception or used as the sole basis for consequential decisions. Model performance and generalization require further validation.

## Collaboration

Contributions, code reviews, testing, and improvements are welcome through branches and pull requests.

## License

No open-source license has been specified yet. All rights remain with the applicable copyright holders unless a license is added.
