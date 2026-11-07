import cv2
import tempfile
import os
import time
from pathlib import Path
from config import MIN_FREE_RAM_PERCENT, SAMPLE_RATE
from modules.utils import check_ram_availability

# Load OpenCV's built-in Haar Cascade classifiers for face and upper body/eye tracking
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_posture_slump(frame):
    """
    Uses face/eye alignment as a rough head-tilt cue, not a full posture diagnosis.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) == 0:
        return False, frame

    is_slumping = False
    for (x, y, w, h) in faces:
        # Draw face boundary box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Detect eyes within the face ROI
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Eye level tilt check
        if len(eyes) >= 2:
            # Sort eyes horizontally (left to right)
            eyes = sorted(eyes, key=lambda e: e[0])
            eye_1_y = eyes[0][1]
            eye_2_y = eyes[1][1]
            
            # Eye height difference threshold (head tilt / leaning)
            if abs(eye_1_y - eye_2_y) > 12:
                is_slumping = True

    return is_slumping, frame

def analyze_video_posture(video_source, sample_rate=SAMPLE_RATE, progress_callback=None):
    """Analyze a video path frame-by-frame, or accept legacy in-memory bytes."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")

    owns_temp_file = isinstance(video_source, bytes)
    temp_video_path = None
    if owns_temp_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_source)
            temp_video_path = temp_file.name
    else:
        temp_video_path = str(Path(video_source))

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        cap.release()
        if owns_temp_file:
            os.remove(temp_video_path)
        raise RuntimeError("The uploaded file could not be opened as a video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_total = total_frames if total_frames > 0 else None
    progress_stage = "Scanning video frames"
    last_progress_time = time.monotonic()
    if progress_callback:
        progress_callback(progress_stage, 0, progress_total)

    slump_frames = 0
    processed_frames = 0
    frame_index = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            if frame_index == 1 or frame_index % sample_rate == 0:
                has_memory, available_percent = check_ram_availability()
                if not has_memory:
                    raise MemoryError(
                        f"Video analysis paused to preserve memory. "
                        f"{available_percent}% RAM is available; "
                        f"{MIN_FREE_RAM_PERCENT}% is required."
                    )

                processed_frames += 1
                is_slumped, _ = detect_posture_slump(frame)
                if is_slumped:
                    slump_frames += 1
                del frame

            now = time.monotonic()
            if progress_callback and now - last_progress_time >= 1:
                progress_callback(progress_stage, frame_index, progress_total)
                last_progress_time = now
    finally:
        cap.release()
        if owns_temp_file and os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    if progress_callback:
        progress_callback(
            progress_stage,
            frame_index,
            progress_total if progress_total is not None else frame_index,
        )

    if processed_frames == 0:
        raise RuntimeError("The uploaded file contains no readable video frames.")

    slump_pct = slump_frames / processed_frames * 100
    return round(slump_pct, 2)


def open_live_webcam():
    """Open the webcam for an interactive live session."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return None, "Webcam could not be opened. Check camera permissions."
    return cap, None


def process_live_webcam_frame(cap):
    """Read and analyze one webcam frame."""
    has_memory, available_percent = check_ram_availability()
    if not has_memory:
        return (
            None,
            None,
            f"Live posture analysis paused to preserve memory. "
            f"{available_percent}% RAM is available; "
            f"{MIN_FREE_RAM_PERCENT}% is required.",
        )

    ret, frame = cap.read()
    if not ret:
        return None, None, "Could not read a frame from the webcam."

    is_slumped, annotated_frame = detect_posture_slump(frame)
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    return rgb_frame, is_slumped, None


def calculate_slump_percentage(slump_frames, total_frames):
    """Calculate the percentage of sampled frames flagged for head/eye tilt."""
    slump_pct = (slump_frames / total_frames * 100) if total_frames else 0.0
    return round(slump_pct, 2)