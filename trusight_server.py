"""
TRUSIGHT Unified Gateway Server
Acts as the single localhost entrypoint on port 8000.
Coordinates the Face Analysis Engine (port 8002) and Voice Analysis Engine (port 8001).
Serves the unified forensic dashboard and reverse-proxies API requests, video feeds, and WebSockets.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

VOICE_BACKEND = os.getenv("VOICE_BACKEND", "http://127.0.0.1:8001").rstrip("/")
FACE_BACKEND = os.getenv("FACE_BACKEND", "http://127.0.0.1:8002").rstrip("/")
VOICE_WS = VOICE_BACKEND.replace("http://", "ws://").replace("https://", "wss://")

app = FastAPI(
    title="TRUSIGHT Multimodal Intelligence Gateway",
    description="Unified localhost gateway for Face and Voice forensic analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(content={"service": "TRUSIGHT Unified Gateway", "status": "online"})


@app.get("/api/health")
async def health_check():
    """Aggregated health check querying both backend engines."""
    voice_health = {"status": "unreachable", "model_loaded": False}
    face_health = {"status": "unreachable", "camera_running": False}

    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            res_voice = await client.get(f"{VOICE_BACKEND}/health")
            if res_voice.status_code == 200:
                voice_health = res_voice.json()
        except Exception:
            pass

        try:
            res_face = await client.get(f"{FACE_BACKEND}/health")
            if res_face.status_code == 200:
                face_health = res_face.json()
        except Exception:
            pass

    return {
        "gateway": "online",
        "voice_engine": voice_health,
        "face_engine": face_health,
        "unified": voice_health.get("status") == "running" and face_health.get("status") == "ok",
    }


# ── Face Engine Endpoints ───────────────────────────────────────────────────

@app.get("/api/face/video_feed")
async def face_video_feed():
    """Proxy MJPEG stream from the Face Analysis Engine directly to the browser."""
    client = httpx.AsyncClient(timeout=None)
    try:
        req = client.build_request("GET", f"{FACE_BACKEND}/video_feed")
        res = await client.send(req, stream=True)
        return StreamingResponse(
            res.aiter_bytes(),
            status_code=res.status_code,
            media_type=res.headers.get("content-type", "multipart/x-mixed-replace; boundary=frame"),
            background=BackgroundTask(res.aclose),
        )
    except Exception as exc:
        await client.aclose()
        # Fallback 1x1 transparent JPEG or error notice if camera is paused
        return JSONResponse(status_code=503, content={"detail": f"Face camera stream unavailable: {exc}"})


@app.get("/api/face/{path:path}")
async def face_proxy_get(path: str, request: Request):
    """Proxy GET requests to the Face Engine (e.g. /live, /health, /camera/status)."""
    url = f"{FACE_BACKEND}/{path}"
    query = request.url.query
    if query:
        url += f"?{query}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            res = await client.get(url)
            return Response(content=res.content, status_code=res.status_code, media_type=res.headers.get("content-type"))
        except httpx.RequestError as exc:
            return JSONResponse(status_code=503, content={"detail": f"Face Engine unavailable: {exc}"})


@app.post("/api/face/{path:path}")
async def face_proxy_post(path: str, request: Request):
    """Proxy POST requests to the Face Engine (e.g. /camera/start, /camera/stop)."""
    url = f"{FACE_BACKEND}/{path}"
    query = request.url.query
    if query:
        url += f"?{query}"
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            res = await client.post(url, content=body, headers=headers)
            return Response(content=res.content, status_code=res.status_code, media_type=res.headers.get("content-type"))
        except httpx.RequestError as exc:
            return JSONResponse(status_code=503, content={"detail": f"Face Engine unavailable: {exc}"})


# ── Voice Engine Endpoints ──────────────────────────────────────────────────

@app.get("/api/voice/{path:path}")
async def voice_proxy_get(path: str, request: Request):
    """Proxy GET requests to the Voice Engine (e.g. /health, /live/status, /live/results, /live/summary)."""
    url = f"{VOICE_BACKEND}/{path}"
    query = request.url.query
    if query:
        url += f"?{query}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            res = await client.get(url)
            return Response(content=res.content, status_code=res.status_code, media_type=res.headers.get("content-type"))
        except httpx.RequestError as exc:
            return JSONResponse(status_code=503, content={"detail": f"Voice Engine unavailable: {exc}"})


@app.post("/api/voice/{path:path}")
async def voice_proxy_post(path: str, request: Request):
    """Proxy POST requests to Voice Engine (e.g. /analyze, /live/analyze, /live/start, /live/stop)."""
    url = f"{VOICE_BACKEND}/{path}"
    query = request.url.query
    if query:
        url += f"?{query}"
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            res = await client.post(url, content=body, headers=headers)
            return Response(content=res.content, status_code=res.status_code, media_type=res.headers.get("content-type"))
        except httpx.RequestError as exc:
            return JSONResponse(status_code=503, content={"detail": f"Voice Engine unavailable: {exc}"})


@app.websocket("/ws/voice/live")
async def voice_websocket_proxy(websocket: WebSocket):
    """Proxy WebSocket connection to the Voice Engine /ws/live endpoint."""
    import websockets
    await websocket.accept()
    target_url = f"{VOICE_WS}/ws/live"
    try:
        async with websockets.connect(target_url) as backend_ws:
            async def forward_to_client():
                try:
                    while True:
                        msg = await backend_ws.recv()
                        await websocket.send_text(msg)
                except Exception:
                    pass

            async def forward_to_server():
                try:
                    while True:
                        msg = await websocket.receive_text()
                        await backend_ws.send(msg)
                except Exception:
                    pass

            await asyncio.gather(forward_to_client(), forward_to_server(), return_exceptions=True)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.close(code=1011, reason=f"Voice WebSocket proxy error: {exc}")
        except Exception:
            pass


def main():
    import uvicorn
    port = int(os.getenv("PORT", os.getenv("TRUSIGHT_PORT", "8000")))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"\n=======================================================")
    print(f"  TRUSIGHT UNIFIED GATEWAY STARTING")
    print(f"  Dashboard: http://localhost:{port}")
    print(f"  Voice Engine Target: {VOICE_BACKEND}")
    print(f"  Face Engine Target : {FACE_BACKEND}")
    print(f"=======================================================\n")
    uvicorn.run("trusight_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
