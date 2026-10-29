import asyncio

from dotenv import load_dotenv
load_dotenv()

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    WebSocket,
    WebSocketDisconnect
)

from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import shutil
import os
import uuid
import tempfile

from rag import (
    extract_pdf,
    chunk_text,
    build_index,
    retrieve
)

from predict import predict
from explanation import generate_explanation
from gemini_client import get_client, LIVE_TRANSCRIPTION_MODEL
from google.genai import types


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Truth Verification System",
    description="Live interview evidence verification system.",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STORE LOADED CASES
# ============================================================

cases = {}


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "service": "AI Truth Verification System",
        "version": "2.0.0"
    }


# ============================================================
# LIVE INTERVIEW PAGE
# ============================================================

@app.get("/live", response_class=HTMLResponse)
def live_page():

    html = r"""
<!DOCTYPE html>

<html>

<head>

    <title>AI Truth Verification System</title>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #222;
        }

        .container {
            max-width: 1100px;
            margin: 30px auto;
            padding: 20px;
        }

        h1 {
            margin-bottom: 5px;
        }

        .subtitle {
            color: #666;
            margin-bottom: 25px;
        }

        .card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }

        input[type="file"] {
            margin: 15px 0;
        }

        button {
            padding: 12px 20px;
            border: none;
            border-radius: 7px;
            cursor: pointer;
            font-size: 15px;
            margin-right: 10px;
        }

        #loadCase {
            background: #333;
            color: white;
        }

        #start {
            background: #2e7d32;
            color: white;
        }

        #stop {
            background: #c62828;
            color: white;
        }

        button:disabled {
            background: #aaa !important;
            cursor: not-allowed;
        }

        .status {
            margin-top: 15px;
            padding: 15px;
            border-radius: 6px;
            background: #eeeeee;
            white-space: pre-wrap;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            min-height: 150px;
            overflow-wrap: anywhere;
        }

        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: Consolas, monospace;
        }

        .verdict {
            font-size: 28px;
            font-weight: bold;
            margin: 10px 0;
        }

        .confidence {
            font-size: 18px;
            margin-top: 8px;
        }

        .score-label {
            font-size: 14px;
            color: #666;
            margin-top: 14px;
        }

        .score-value {
            font-size: 38px;
            font-weight: bold;
            margin: 4px 0 8px;
        }

        .score-track {
            width: 100%;
            height: 12px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
            margin: 8px 0;
        }

        .score-fill {
            height: 100%;
            width: 0%;
            background: #64748b;
            transition: width 250ms ease;
        }

        .score-note {
            font-size: 12px;
            color: #666;
            line-height: 1.4;
            margin-top: 8px;
        }

        .success {
            background: #dff5df;
        }

        .error {
            background: #ffe0e0;
        }

        @media(max-width: 700px) {

            .grid {
                grid-template-columns: 1fr;
            }

        }

    </style>

</head>


<body>

<div class="container">


    <h1>
        AI Truth Verification System
    </h1>

    <div class="subtitle">
        Live Interview Evidence Verification
    </div>


    <!-- ================================================= -->
    <!-- CASE REPORT -->
    <!-- ================================================= -->

    <div class="card">

        <h2>
            1. Load Case Report
        </h2>

        <input
            type="file"
            id="pdfFile"
            accept=".pdf,application/pdf"
        >

        <br>

        <button
            type="button"
            id="loadCase"
        >
            Load Case
        </button>

        <div
            id="caseStatus"
            class="status"
        >
            No case loaded.
        </div>

    </div>


    <!-- ================================================= -->
    <!-- LIVE INTERVIEW -->
    <!-- ================================================= -->

    <div class="card">

        <h2>
            2. Live Interview
        </h2>

        <button
            type="button"
            id="start"
            disabled
        >
            Start Interview
        </button>

        <button
            type="button"
            id="stop"
            disabled
        >
            Stop Interview
        </button>

        <div
            id="micStatus"
            class="status"
        >
            Microphone not started.
        </div>

    </div>


    <!-- ================================================= -->
    <!-- RESULTS -->
    <!-- ================================================= -->

    <div class="grid">


        <div class="card">

            <h2>
                Live Transcript
            </h2>

            <div class="box">

                <pre id="transcript">Waiting for speech...</pre>

            </div>

        </div>


        <div class="card">

            <h2>
                Deception Score
            </h2>

            <div class="box">

                <div
                    id="verdict"
                    class="verdict"
                >
                    Waiting...
                </div>

                <div class="score-label">Estimated Inconsistency Score</div>
                <div id="deceptionScore" class="score-value">--/100</div>
                <div class="score-track" aria-label="Estimated inconsistency score">
                    <div id="scoreFill" class="score-fill"></div>
                </div>
                <div id="scoreNote" class="score-note">
                    Waiting for enough transcript and evidence to analyze.
                </div>

                <div
                    id="confidence"
                    class="confidence"
                >
                    Confidence: -
                </div>

            </div>

        </div>


        <div class="card">

            <h2>
                Retrieved Evidence
            </h2>

            <div class="box">

                <pre id="evidence">No evidence retrieved.</pre>

            </div>

        </div>


        <div class="card">

            <h2>
                Explanation
            </h2>

            <div class="box">

                <pre id="explanation">No explanation generated.</pre>

            </div>

        </div>


    </div>

</div>


<script>


// ============================================================
// VARIABLES
// ============================================================

let caseId = null;

let socket = null;

let stream = null;

let recorder = null;

let recording = false;

let audioChunks = [];

let segmentTimer = null;
let audioContext = null;
let finalizedTranscript = "";
let audioSource = null;
let audioProcessor = null;
let silentGain = null;


// ============================================================
// ELEMENTS
// ============================================================

const pdfFile =
    document.getElementById("pdfFile");

const loadCase =
    document.getElementById("loadCase");

const start =
    document.getElementById("start");

const stop =
    document.getElementById("stop");

const caseStatus =
    document.getElementById("caseStatus");

const micStatus =
    document.getElementById("micStatus");

const transcript =
    document.getElementById("transcript");

const verdict =
    document.getElementById("verdict");

const confidence =
    document.getElementById("confidence");

const deceptionScore =
    document.getElementById("deceptionScore");

const scoreFill =
    document.getElementById("scoreFill");

const scoreNote =
    document.getElementById("scoreNote");

const evidence =
    document.getElementById("evidence");

const explanation =
    document.getElementById("explanation");


console.log(
    "Live page loaded."
);


// ============================================================
// LOAD CASE
// ============================================================

loadCase.addEventListener(
    "click",
    async function () {

        console.log(
            "LOAD CASE CLICKED"
        );


        const file =
            pdfFile.files[0];


        if (!file) {

            alert(
                "Please select a PDF first."
            );

            return;
        }


        console.log(
            "Selected file:",
            file.name
        );


        caseStatus.className =
            "status";


        caseStatus.textContent =
            "Loading case report...";


        loadCase.disabled = true;


        const formData =
            new FormData();


        formData.append(
            "pdf",
            file,
            file.name
        );


        try {

            const response =
                await fetch(
                    "/case",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            console.log(
                "Response status:",
                response.status
            );


            const data =
                await response.json();


            console.log(
                "Response:",
                data
            );


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Failed to load case."
                );

            }


            caseId =
                data.case_id;


            if (!caseId) {

                throw new Error(
                    "No case_id received from server."
                );

            }


            caseStatus.className =
                "status success";


            caseStatus.textContent =
                "Case loaded successfully.\n\n" +

                "Pages: " +
                data.pages +

                "\nChunks: " +
                data.chunks +

                "\nCase ID: " +
                caseId;


            start.disabled = false;


        }

        catch (error) {

            console.error(
                "CASE ERROR:",
                error
            );


            caseStatus.className =
                "status error";


            caseStatus.textContent =
                "Error loading case:\n" +
                error.message;

        }

        finally {

            loadCase.disabled = false;

        }

    }
);


// ============================================================
// START INTERVIEW
// ============================================================

start.addEventListener(
    "click",
    async function () {


        if (!caseId) {

            alert(
                "Load the case report first."
            );

            return;
        }


        try {


            // ------------------------------------------------
            // MICROPHONE
            // ------------------------------------------------

            micStatus.className =
                "status";


            micStatus.textContent =
                "Requesting microphone permission...";


            stream =
                await navigator
                    .mediaDevices
                    .getUserMedia({
                        audio: {
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        }
                    });


            console.log(
                "Microphone access granted."
            );


            // ------------------------------------------------
            // WEBSOCKET
            // ------------------------------------------------

            const protocol =
                window.location.protocol === "https:"
                    ? "wss:"
                    : "ws:";


            const wsUrl =
                protocol +
                "//" +
                window.location.host +
                "/ws/audio/" +
                caseId;


            console.log(
                "WebSocket:",
                wsUrl
            );


            socket =
                new WebSocket(
                    wsUrl
                );


            socket.binaryType =
                "arraybuffer";


            // ------------------------------------------------
            // CONNECTED
            // ------------------------------------------------

            socket.onopen =
            function () {

                console.log(
                    "WebSocket connected."
                );

                micStatus.className =
                    "status success";


                micStatus.textContent =
                    "Connected. Listening...";


                recording = true;


                start.disabled = true;

                stop.disabled = false;


                startLiveAudio();

            };


            // ------------------------------------------------
            // MESSAGE
            // ------------------------------------------------

            socket.onmessage =
            function (event) {


                console.log(
                    "SERVER:",
                    event.data
                );


                let data;


                try {

                    data =
                        JSON.parse(
                            event.data
                        );

                }

                catch (error) {

                    console.error(
                        "Invalid server response:",
                        error
                    );

                    return;

                }


                // --------------------------------------------
                // ERROR
                // --------------------------------------------

                if (
                    data.status === "error"
                ) {

                    micStatus.className =
                        "status error";


                    micStatus.textContent =
                        "Error: " +
                        data.message;


                    return;

                }


                // --------------------------------------------
                // LIVE INTERIM TRANSCRIPT
                // --------------------------------------------

                if (data.interim_transcript) {
                    micStatus.textContent = "Listening...";
                    transcript.textContent =
                        finalizedTranscript +
                        (finalizedTranscript ? "\n\n" : "") +
                        "… " + data.interim_transcript;
                }

                // --------------------------------------------
                // FINAL TRANSCRIPT
                // --------------------------------------------

                if (
                    data.transcript
                ) {

                    if (transcript.textContent.trim() === "Waiting for speech...") {
                        finalizedTranscript = "";
                    }

                    finalizedTranscript += data.transcript + "\n\n";
                    transcript.textContent = finalizedTranscript;

                }


                // --------------------------------------------
                // VERDICT
                // --------------------------------------------

                if (
                    data.verdict
                ) {

                    verdict.textContent =
                        data.verdict;

                }

                // --------------------------------------------
                // ESTIMATED INCONSISTENCY SCORE
                // --------------------------------------------
                if (data.deception_score !== undefined && data.deception_score !== null) {
                    const score = Math.max(0, Math.min(100, Number(data.deception_score) || 0));
                    deceptionScore.textContent = Math.round(score) + "/100";
                    scoreFill.style.width = score + "%";
                    scoreNote.textContent = data.score_type ||
                        "Demo score based on explicit contradictions in retrieved evidence.";
                }

                // --------------------------------------------
                // CONFIDENCE
                // --------------------------------------------

                if (
                    data.confidence !== undefined
                ) {

                    confidence.textContent =
                        "Confidence: " +
                        data.confidence +
                        "%";

                }


                // --------------------------------------------
                // EVIDENCE
                // --------------------------------------------

                if (
                    data.evidence
                ) {

                    evidence.textContent =
                        JSON.stringify(
                            data.evidence,
                            null,
                            2
                        );

                }


                // --------------------------------------------
                // EXPLANATION
                // --------------------------------------------

                if (
                    data.explanation
                ) {

                    explanation.textContent =
                        JSON.stringify(
                            data.explanation,
                            null,
                            2
                        );

                }

            };


            // ------------------------------------------------
            // ERROR
            // ------------------------------------------------

            socket.onerror =
            function (error) {

                console.error(
                    "WebSocket error:",
                    error
                );


                micStatus.className =
                    "status error";


                micStatus.textContent =
                    "WebSocket connection error.";

            };


            // ------------------------------------------------
            // CLOSE
            // ------------------------------------------------

            socket.onclose =
            function () {

                console.log(
                    "WebSocket closed."
                );

            };


        }

        catch (error) {

            console.error(
                "MICROPHONE ERROR:",
                error
            );


            micStatus.className =
                "status error";


            micStatus.textContent =
                "Microphone error: " +
                error.message;

        }

    }
);


// ============================================================
// CONTINUOUS LIVE PCM AUDIO STREAM (16 kHz, mono, signed 16-bit)
// ============================================================

function startLiveAudio() {
    if (!recording || !stream || !socket || socket.readyState !== WebSocket.OPEN) return;

    try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        audioContext = new AudioContextClass();
        audioSource = audioContext.createMediaStreamSource(stream);
        // ScriptProcessor is used here for compatibility with the existing single-file UI.
        // It streams small PCM frames continuously, unlike the old 5-second MediaRecorder clips.
        audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        silentGain = audioContext.createGain();
        silentGain.gain.value = 0;

        audioProcessor.onaudioprocess = function (event) {
            if (!recording || !socket || socket.readyState !== WebSocket.OPEN) return;
            const input = event.inputBuffer.getChannelData(0);
            const ratio = audioContext.sampleRate / 16000;
            const outputLength = Math.floor(input.length / ratio);
            const pcm = new Int16Array(outputLength);

            for (let i = 0; i < outputLength; i++) {
                const sourceIndex = Math.min(input.length - 1, Math.floor(i * ratio));
                const sample = Math.max(-1, Math.min(1, input[sourceIndex]));
                pcm[i] = sample < 0 ? sample * 32768 : sample * 32767;
            }

            if (pcm.length) socket.send(pcm.buffer);
        };

        audioSource.connect(audioProcessor);
        audioProcessor.connect(silentGain);
        silentGain.connect(audioContext.destination);
        micStatus.className = "status success";
        micStatus.textContent = "Live transcription connected. Speak naturally; partial text appears as you speak.";
    } catch (error) {
        console.error("Live audio setup failed:", error);
        micStatus.className = "status error";
        micStatus.textContent = "Live audio setup failed: " + error.message;
    }
}

// ============================================================
// STOP INTERVIEW
// ============================================================

stop.addEventListener(
    "click",
    function () {


        console.log(
            "Stopping interview..."
        );


        recording = false;


        if (segmentTimer) {

            clearTimeout(
                segmentTimer
            );

            segmentTimer = null;

        }


        if (
            recorder &&
            recorder.state ===
            "recording"
        ) {

            recorder.stop();

        }


        if (audioProcessor) {
            audioProcessor.onaudioprocess = null;
            try { audioProcessor.disconnect(); } catch (_) {}
            audioProcessor = null;
        }
        if (audioSource) {
            try { audioSource.disconnect(); } catch (_) {}
            audioSource = null;
        }
        if (silentGain) {
            try { silentGain.disconnect(); } catch (_) {}
            silentGain = null;
        }
        if (audioContext) {
            audioContext.close().catch(() => {});
            audioContext = null;
        }

        if (stream) {

            stream
                .getTracks()
                .forEach(
                    function(track) {
                        track.stop();
                    }
                );

            stream = null;

        }


        if (
            socket &&
            socket.readyState ===
            WebSocket.OPEN
        ) {

            socket.close();

        }


        micStatus.className =
            "status";


        micStatus.textContent =
            "Interview stopped.";


        start.disabled = false;

        stop.disabled = true;

    }
);


</script>

</body>

</html>
"""

    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control":
                "no-store, no-cache, must-revalidate",
            "Pragma":
                "no-cache",
            "Expires":
                "0"
        }
    )


# ============================================================
# LOAD CASE PDF
# ============================================================

@app.post("/case")
async def create_case(
    pdf: UploadFile = File(...)
):

    print("\n========================================")
    print("POST /case")
    print("Filename:", pdf.filename)
    print("========================================")


    if not pdf.filename:

        raise HTTPException(
            status_code=400,
            detail="PDF file is required."
        )


    if not pdf.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )


    case_id = str(uuid.uuid4())


    pdf_path = os.path.join(
        tempfile.gettempdir(),
        f"{case_id}.pdf"
    )


    try:


        # ----------------------------------------------------
        # SAVE PDF
        # ----------------------------------------------------

        print(
            "Saving PDF..."
        )


        with open(
            pdf_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                pdf.file,
                buffer
            )


        print(
            "PDF saved:",
            pdf_path
        )


        # ----------------------------------------------------
        # EXTRACT PDF
        # ----------------------------------------------------

        print(
            "Extracting PDF text..."
        )


        pages = extract_pdf(
            pdf_path
        )


        if not pages:

            raise ValueError(
                "No text could be extracted from the PDF."
            )


        print(
            "Pages extracted:",
            len(pages)
        )


        # ----------------------------------------------------
        # CHUNK
        # ----------------------------------------------------

        print(
            "Creating evidence chunks..."
        )


        chunks = chunk_text(
            pages
        )


        if not chunks:

            raise ValueError(
                "No evidence chunks were created."
            )


        print(
            "Chunks created:",
            len(chunks)
        )


        # ----------------------------------------------------
        # BUILD FAISS
        # ----------------------------------------------------

        print(
            "Building FAISS index..."
        )


        index, _ = await asyncio.to_thread(
            build_index,
            chunks
        )


        print(
            "Gemini embeddings index created."
        )


        # ----------------------------------------------------
        # STORE CASE
        # ----------------------------------------------------

        cases[case_id] = {
            "chunks": chunks,
            "index": index
        }


        print(
            "Case stored:",
            case_id
        )


        print(
            "========================================"
        )


        return {

            "status":
                "success",

            "case_id":
                case_id,

            "message":
                "Case report loaded successfully.",

            "pages":
                len(pages),

            "chunks":
                len(chunks)

        }


    except Exception as e:

        print(
            "CASE ERROR:",
            repr(e)
        )

        message = str(e)
        if "GEMINI_API_KEY" in message:
            message = "Gemini API key is missing. Copy .env.example to .env and set GEMINI_API_KEY."
        elif not message:
            message = "Case processing failed. Check the server logs and Gemini API configuration."

        raise HTTPException(
            status_code=500,
            detail=message
        )


    finally:

        if os.path.exists(
            pdf_path
        ):

            os.remove(
                pdf_path
            )


# ============================================================
# LIVE AUDIO WEBSOCKET
# ============================================================

@app.websocket("/ws/audio/{case_id}")
async def audio_stream(websocket: WebSocket, case_id: str):
    """Proxy continuous browser PCM audio to Gemini Live transcription."""
    await websocket.accept()
    if case_id not in cases:
        await websocket.send_json({"status": "error", "message": "Invalid case_id."})
        await websocket.close()
        return

    chunks = cases[case_id]["chunks"]
    index = cases[case_id]["index"]
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=[],  # automatic language detection
            mode="VERBATIM",   # preserve the speaker's wording for evidence review
        ),
    )

    await websocket.send_json({"status": "connected", "case_id": case_id,
                               "message": "Connected to Gemini live transcription."})

    try:
        client = get_client()
        async with client.aio.live.connect(
            model=LIVE_TRANSCRIPTION_MODEL,
            config=config,
        ) as session:

            async def receive_transcripts_and_analyze():
                async for response in session.receive():
                    content = response.server_content
                    if not content:
                        continue

                    interim = getattr(content, "interim_input_transcription", None)
                    if interim and getattr(interim, "text", ""):
                        await websocket.send_json({
                            "status": "interim",
                            "interim_transcript": interim.text,
                        })

                    final = getattr(content, "input_transcription", None)
                    interview = (getattr(final, "text", "") or "").strip() if final else ""
                    if not interview:
                        continue

                    await websocket.send_json({
                        "status": "transcript",
                        "transcript": interview,
                    })

                    try:
                        evidence_items = await asyncio.to_thread(
                            retrieve, interview, chunks, index, 3
                        )
                        result = await asyncio.to_thread(
                            predict, evidence_items, interview
                        )
                        explanation = await asyncio.to_thread(
                            generate_explanation, evidence_items, interview, result
                        )
                        await websocket.send_json({
                            "status": "success",
                            "evidence": evidence_items,
                            "evidence_results": result.get("evidence_results", []),
                            "verdict": result.get("verdict", "Unknown"),
                            "confidence": result.get("confidence", 0),
                            "deception_score": result.get("deception_score", 0),
                            "inconsistency_index": result.get("inconsistency_index", round(result.get("deception_score", 0) / 100.0, 4)),
                            "score_type": result.get(
                                "score_type",
                                "Estimated inconsistency score based on explicit contradictions; not a probability of deception.",
                            ),
                            "explanation": explanation,
                        })
                    except Exception as analysis_error:
                        # Print the complete traceback so the actual Gemini/RAG
                        # failure is visible in the VS Code terminal.
                        import traceback
                        print("\n===== EVIDENCE ANALYSIS FAILED =====", flush=True)
                        traceback.print_exc()

                        try:
                            await websocket.send_json({
                                "status": "error",
                                "stage": "evidence_analysis",
                                "message": str(analysis_error),
                            })
                        except Exception:
                            # The browser may have disconnected while analysis failed.
                            pass

            async def forward_audio():
                while True:
                    audio_bytes = await websocket.receive_bytes()
                    if not audio_bytes:
                        continue
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=audio_bytes,
                            mime_type="audio/pcm;rate=16000",
                        )
                    )

            transcript_task = asyncio.create_task(receive_transcripts_and_analyze())
            audio_task = asyncio.create_task(forward_audio())
            done, pending = await asyncio.wait(
                {transcript_task, audio_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exception = task.exception()
                if exception:
                    raise exception

    except WebSocketDisconnect:
        print(f"Client disconnected from case {case_id}")
    except asyncio.CancelledError:
        raise
    except Exception as error:
        print("Gemini live WebSocket error:", repr(error))
        try:
            await websocket.send_json({
                "status": "error",
                "stage": "gemini_live_transcription",
                "message": str(error),
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
