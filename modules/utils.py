# modules/utils.py

import psutil
import os
import tempfile
from pathlib import Path

from config import MIN_FREE_RAM_PERCENT, UPLOAD_COPY_CHUNK_BYTES


def get_ram_status():
    """Return available and total system memory in bytes and available percent."""
    mem = psutil.virtual_memory()
    available_percent = (mem.available / mem.total) * 100.0
    return mem.available, mem.total, round(available_percent, 2)


def check_ram_availability(min_percent=MIN_FREE_RAM_PERCENT):
    """Return whether the configured free-memory reserve is available."""
    _, _, available_percent = get_ram_status()
    return available_percent >= min_percent, available_percent


def save_uploaded_file_in_chunks(uploaded_file):
    """Copy an uploaded file to local disk without creating another full-size copy."""
    suffix = Path(uploaded_file.name).suffix
    descriptor, temp_path = tempfile.mkstemp(suffix=suffix)

    try:
        with os.fdopen(descriptor, "wb") as destination:
            uploaded_file.seek(0)
            while True:
                chunk = uploaded_file.read(UPLOAD_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                destination.write(chunk)
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise

    return Path(temp_path)