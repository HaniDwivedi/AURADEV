import atexit
import json
import os
import secrets
import shutil
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from config import UPLOAD_CHUNK_BYTES

UPLOAD_EXPIRY_SECONDS = 24 * 60 * 60
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}


@dataclass
class UploadRecord:
    upload_id: str
    filename: str
    size: int
    path: Path
    received: int = 0
    status: str = "uploading"
    last_activity: float = field(default_factory=time.monotonic)
    receiving_chunk: bool = False
    cancel_requested: bool = False


_lock = threading.RLock()
_sessions = {}
_uploads = {}
_server_thread = None
_server = None
_server_port = None
_upload_directory = None
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def _get_upload_directory():
    configured_directory = os.environ.get("AURADEV_UPLOAD_DIR")
    parent_directory = (
        Path(configured_directory).expanduser()
        if configured_directory
        else Path(tempfile.gettempdir())
    )
    parent_directory.mkdir(parents=True, exist_ok=True)
    if not parent_directory.is_dir():
        raise OSError(f"Upload location is not a directory: {parent_directory}")
    return Path(tempfile.mkdtemp(prefix="auradev-uploads-", dir=parent_directory))


def _remove_record(token, upload_id):
    with _lock:
        record = _uploads.get(token, {}).pop(upload_id, None)
    if record is not None:
        record.path.unlink(missing_ok=True)


def _clean_expired_records():
    now = time.monotonic()
    expired = []
    with _lock:
        for token, uploads in list(_uploads.items()):
            for upload_id, record in list(uploads.items()):
                if (
                    not record.receiving_chunk
                    and record.status != "processing"
                    and now - record.last_activity > UPLOAD_EXPIRY_SECONDS
                ):
                    expired.append((token, upload_id))
            if token not in _sessions and not uploads:
                _uploads.pop(token, None)
        for token, last_seen in list(_sessions.items()):
            has_active_upload = any(
                record.status == "processing" or record.receiving_chunk
                for record in _uploads.get(token, {}).values()
            )
            if (
                now - last_seen > UPLOAD_EXPIRY_SECONDS
                and not has_active_upload
            ):
                _sessions.pop(token, None)
                for upload_id in list(_uploads.get(token, {})):
                    expired.append((token, upload_id))

    for token, upload_id in expired:
        _remove_record(token, upload_id)


def register_upload_session(token):
    _clean_expired_records()
    with _lock:
        _sessions[token] = time.monotonic()
        _uploads.setdefault(token, {})


def list_uploads(token):
    _clean_expired_records()
    with _lock:
        if token not in _sessions:
            return []
        _sessions[token] = time.monotonic()
        records = list(_uploads.get(token, {}).values())
        return [
            {
                "upload_id": record.upload_id,
                "filename": record.filename,
                "size": record.size,
                "received": record.received,
                "status": record.status,
            }
            for record in records
        ]


def claim_completed_upload(token, upload_id):
    with _lock:
        record = _uploads.get(token, {}).get(upload_id)
        if record is None or record.status != "complete":
            raise RuntimeError("That upload is no longer available for analysis.")
        record.status = "processing"
        record.last_activity = time.monotonic()
        _sessions[token] = time.monotonic()
        return record.path, record.filename


def finish_upload(token, upload_id, *, remove):
    if remove:
        _remove_record(token, upload_id)
        return
    with _lock:
        record = _uploads.get(token, {}).get(upload_id)
        if record is not None and record.status == "processing":
            record.status = "complete"
            record.last_activity = time.monotonic()


def discard_upload(token, upload_id):
    with _lock:
        record = _uploads.get(token, {}).get(upload_id)
        if record is None:
            return
        if record.receiving_chunk:
            record.cancel_requested = True
            return
    _remove_record(token, upload_id)


def _validate_local_request(request, token):
    with _lock:
        if token not in _sessions:
            raise HTTPException(status_code=404, detail="Upload session expired.")
        _sessions[token] = time.monotonic()

    if request.url.hostname not in {"127.0.0.1", "localhost"}:
        raise HTTPException(status_code=403, detail="Local uploads only.")

    origin = request.headers.get("origin")
    if origin:
        parsed_origin = urlsplit(origin)
        if (
            parsed_origin.scheme != "http"
            or parsed_origin.hostname not in {"127.0.0.1", "localhost"}
            or parsed_origin.port != _server_port
        ):
            raise HTTPException(status_code=403, detail="Cross-origin request denied.")


def _parse_upload_id(upload_id):
    try:
        return str(uuid.UUID(upload_id))
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid upload identifier.") from error


@app.get("/upload/{token}", response_class=HTMLResponse)
async def upload_page(token: str, request: Request):
    _validate_local_request(request, token)
    nonce = secrets.token_urlsafe(18)
    safe_token = json.dumps(token)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AuraDev local video upload</title>
  <style>
    body {{ font: 16px system-ui, sans-serif; max-width: 760px; margin: 3rem auto; padding: 0 1rem; color: #172033; }}
    main {{ border: 1px solid #d6deea; border-radius: 12px; padding: 1.5rem; }}
    input, button {{ font: inherit; margin: .5rem 0; }}
    progress {{ display: block; width: 100%; height: 1.25rem; }}
    #status {{ min-height: 1.5rem; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <main>
    <h1>Upload a video to AuraDev</h1>
    <p>The browser sends fixed-size pieces directly to this computer. The whole video is never sent as one request.</p>
    <p>Uploads are saved to local disk; keep enough free disk space for the video and temporary audio extraction.</p>
    <label for="video">Choose an MP4, MOV, or AVI video</label><br>
    <input id="video" type="file" accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo">
    <button id="start" type="button">Start upload</button>
    <button id="cancel" type="button" disabled>Cancel upload</button>
    <progress id="progress" value="0" max="100"></progress>
    <p id="status" role="status" aria-live="polite"></p>
  </main>
  <script nonce="{nonce}">
    const token = {safe_token};
    const chunkBytes = {UPLOAD_CHUNK_BYTES};
    const input = document.getElementById("video");
    const startButton = document.getElementById("start");
    const cancelButton = document.getElementById("cancel");
    const progress = document.getElementById("progress");
    const status = document.getElementById("status");
    let activeUpload = null;

    function formatBytes(bytes) {{
      if (bytes < 1024 ** 3) return (bytes / (1024 ** 2)).toFixed(1) + " MiB";
      return (bytes / (1024 ** 3)).toFixed(2) + " GiB";
    }}

    async function checkedFetch(url, options) {{
      const response = await fetch(url, options);
      if (!response.ok) {{
        const body = await response.json().catch(() => ({{}}));
        throw new Error(body.detail || "Upload request failed (" + response.status + ").");
      }}
      return response;
    }}

    startButton.addEventListener("click", async () => {{
      const file = input.files[0];
      if (!file) {{ status.textContent = "Choose a video first."; return; }}
      const extension = file.name.split(".").pop().toLowerCase();
      if (!["mp4", "mov", "avi"].includes(extension)) {{
        status.textContent = "Choose an MP4, MOV, or AVI file.";
        return;
      }}

      const uploadId = crypto.randomUUID();
      const controller = new AbortController();
      activeUpload = {{uploadId, controller}};
      startButton.disabled = true;
      cancelButton.disabled = false;
      progress.value = 0;
      status.textContent = "Preparing " + file.name + " (" + formatBytes(file.size) + ")…";
      const base = "/api/uploads/" + encodeURIComponent(token) + "/" + uploadId;

      try {{
        await checkedFetch(base + "/start", {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{filename: file.name, size: file.size}}),
          signal: controller.signal
        }});

        let sent = 0;
        let chunkIndex = 0;
        while (sent < file.size) {{
          const end = Math.min(sent + chunkBytes, file.size);
          await checkedFetch(base + "/chunks/" + chunkIndex, {{
            method: "POST",
            headers: {{"Content-Type": "application/octet-stream"}},
            body: file.slice(sent, end),
            signal: controller.signal
          }});
          sent = end;
          chunkIndex += 1;
          progress.value = (sent / file.size) * 100;
          status.textContent = "Uploaded " + formatBytes(sent) + " of " + formatBytes(file.size);
        }}
        status.textContent = "Upload complete. Return to AuraDev to start the analysis.";
      }} catch (error) {{
        if (error.name === "AbortError") {{
          status.textContent = "Upload cancelled.";
        }} else {{
          status.textContent = "Upload stopped: " + error.message;
        }}
        await fetch(base, {{method: "DELETE"}}).catch(() => {{}});
      }} finally {{
        activeUpload = null;
        startButton.disabled = false;
        cancelButton.disabled = true;
      }}
    }});

    cancelButton.addEventListener("click", () => {{
      if (activeUpload) activeUpload.controller.abort();
    }});
  </script>
</body>
</html>"""
    response = HTMLResponse(page)
    response.headers["Content-Security-Policy"] = (
        f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.post("/api/uploads/{token}/{upload_id}/start")
async def start_upload(token: str, upload_id: str, request: Request):
    _validate_local_request(request, token)
    upload_id = _parse_upload_id(upload_id)

    content_length = request.headers.get("content-length", "")
    if not content_length.isdigit():
        raise HTTPException(status_code=411, detail="Content-Length is required.")
    if int(content_length) > 4096:
        raise HTTPException(status_code=413, detail="Invalid upload metadata.")
    try:
        metadata = await request.json()
    except (ValueError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=400, detail="Invalid upload metadata.") from error
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="Invalid upload metadata.")

    filename = metadata.get("filename")
    size = metadata.get("size")
    if not isinstance(filename, str) or not isinstance(size, int) or isinstance(size, bool):
        raise HTTPException(status_code=400, detail="Invalid file name or size.")
    safe_filename = Path(filename.replace("\\", "/")).name
    if not safe_filename or Path(safe_filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Choose an MP4, MOV, or AVI file.")
    if size <= 0:
        raise HTTPException(status_code=400, detail="The selected file is empty.")

    try:
        free_bytes = shutil.disk_usage(_upload_directory).free
    except OSError as error:
        raise HTTPException(
            status_code=507,
            detail=f"Cannot check upload disk space: {error}",
        ) from error
    if size > free_bytes:
        raise HTTPException(
            status_code=507,
            detail=(
                f"Not enough free disk space. Video size: {size / (1024 ** 3):.2f} GiB; "
                f"available: {free_bytes / (1024 ** 3):.2f} GiB. "
                "Choose a disk with more free space using AURADEV_UPLOAD_DIR."
            ),
        )

    with _lock:
        if token not in _sessions:
            raise HTTPException(status_code=404, detail="Upload session expired.")
        if upload_id in _uploads.setdefault(token, {}):
            raise HTTPException(status_code=409, detail="Upload identifier already exists.")
        suffix = Path(safe_filename).suffix.lower()
        path = _upload_directory / f"{upload_id}{suffix}"
        try:
            with path.open("xb"):
                pass
        except OSError as error:
            raise HTTPException(
                status_code=507,
                detail=f"Cannot create upload file: {error}",
            ) from error
        _uploads[token][upload_id] = UploadRecord(
            upload_id=upload_id,
            filename=safe_filename,
            size=size,
            path=path,
        )

    return {"chunk_bytes": UPLOAD_CHUNK_BYTES}


@app.post("/api/uploads/{token}/{upload_id}/chunks/{chunk_index}")
async def receive_chunk(token: str, upload_id: str, chunk_index: int, request: Request):
    _validate_local_request(request, token)
    upload_id = _parse_upload_id(upload_id)

    with _lock:
        record = _uploads.get(token, {}).get(upload_id)
        if record is None or record.status != "uploading":
            raise HTTPException(status_code=404, detail="Upload is not active.")
        if record.receiving_chunk:
            raise HTTPException(status_code=409, detail="A chunk is already being written.")
        expected_index = record.received // UPLOAD_CHUNK_BYTES
        if chunk_index != expected_index:
            raise HTTPException(status_code=409, detail="Unexpected chunk order.")
        expected_bytes = min(UPLOAD_CHUNK_BYTES, record.size - record.received)
        if expected_bytes <= 0:
            raise HTTPException(status_code=409, detail="Upload is already complete.")
        record.receiving_chunk = True
        offset = record.received

    received_bytes = 0
    try:
        if shutil.disk_usage(_upload_directory).free < expected_bytes:
            raise HTTPException(
                status_code=507,
                detail="The upload disk is full. Free space and retry this upload.",
            )
        with record.path.open("r+b") as output:
            output.seek(offset)
            async for data in request.stream():
                received_bytes += len(data)
                if received_bytes > expected_bytes:
                    raise HTTPException(status_code=413, detail="Chunk exceeds allowed size.")
                output.write(data)
            if received_bytes != expected_bytes:
                raise HTTPException(status_code=400, detail="Chunk was incomplete.")
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        with record.path.open("r+b") as output:
            output.truncate(offset)
        with _lock:
            record.receiving_chunk = False
            record.last_activity = time.monotonic()
            if record.cancel_requested:
                _uploads.get(token, {}).pop(upload_id, None)
                record.path.unlink(missing_ok=True)
        raise

    with _lock:
        record.receiving_chunk = False
        record.last_activity = time.monotonic()
        if record.cancel_requested:
            _uploads.get(token, {}).pop(upload_id, None)
            record.path.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail="Upload was cancelled.")
        record.received += received_bytes
        if record.received == record.size:
            record.status = "complete"
        _sessions[token] = time.monotonic()
        complete = record.status == "complete"

    return {"received": record.received, "complete": complete}


@app.delete("/api/uploads/{token}/{upload_id}")
async def cancel_upload(token: str, upload_id: str, request: Request):
    _validate_local_request(request, token)
    upload_id = _parse_upload_id(upload_id)
    with _lock:
        record = _uploads.get(token, {}).get(upload_id)
        if record is None:
            return JSONResponse({"cancelled": True})
        if record.receiving_chunk:
            record.cancel_requested = True
            return JSONResponse({"cancelled": True, "pending_chunk_cleanup": True})
    _remove_record(token, upload_id)
    return JSONResponse({"cancelled": True})


def start_upload_server():
    global _server_thread, _server, _server_port, _upload_directory

    with _lock:
        if _server_thread is not None and _server_thread.is_alive():
            return _server_port

        _upload_directory = _get_upload_directory()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            sock.listen(128)
        except OSError:
            sock.close()
            raise

        _server_port = sock.getsockname()[1]
        _server = uvicorn.Server(
            uvicorn.Config(
                app,
                log_level="warning",
                access_log=False,
                lifespan="off",
            )
        )
        _server_thread = threading.Thread(
            target=_server.run,
            kwargs={"sockets": [sock]},
            name="auradev-chunk-upload",
            daemon=True,
        )
        _server_thread.start()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if _server.started:
            return _server_port
        if not _server_thread.is_alive():
            raise RuntimeError("The local chunk-upload server stopped during startup.")
        time.sleep(0.05)
    _server.should_exit = True
    raise RuntimeError("The local chunk-upload server did not start within 10 seconds.")


def get_upload_url(token):
    if _server_port is None:
        raise RuntimeError("The local chunk-upload server has not started.")
    return f"http://127.0.0.1:{_server_port}/upload/{token}"


def stop_upload_server():
    if _server is not None:
        _server.should_exit = True
    if _server_thread is not None and _server_thread is not threading.current_thread():
        _server_thread.join(timeout=5)


def _cleanup_upload_directory():
    if _upload_directory is not None:
        shutil.rmtree(_upload_directory, ignore_errors=True)


import atexit

atexit.register(_cleanup_upload_directory)
atexit.register(stop_upload_server)
