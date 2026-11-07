import re
import shutil
import subprocess
import tempfile
import urllib.error
import wave
import os
from collections import Counter
from pathlib import Path

import imageio_ffmpeg
import whisper

from config import (
    MIN_FREE_RAM_PERCENT,
    TRANSCRIPTION_CHUNK_SECONDS,
    WHISPER_MODEL_SIZE,
)
from modules.utils import check_ram_availability

# Ensure ffmpeg from imageio_ffmpeg is exposed in environment PATH for Whisper subprocesses
try:
    ffmpeg_exe_path = imageio_ffmpeg.get_ffmpeg_exe()
    if ffmpeg_exe_path:
        ffmpeg_dir = str(Path(ffmpeg_exe_path).parent)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except (ImportError, OSError, RuntimeError):
    pass


def _count_hesitations(text):
    words = re.findall(r"[\w']+", text.lower())
    fillers = {"um", "uh", "ah", "erm", "hmm", "mmm"}
    count = sum(word in fillers for word in words)
    count += sum(
        words[index:index + 2] == ["you", "know"]
        for index in range(len(words) - 1)
    )
    count += sum(
        words[index] == words[index + 1] and len(words[index]) > 1
        for index in range(len(words) - 1)
    )
    count += sum(
        words[index:index + 2] == words[index + 2:index + 4]
        for index in range(len(words) - 3)
    )
    return count


def get_transcript_insights(text):
    """Return objective transcript-only indicators without scoring pronunciation."""
    words = re.findall(r"[\w']+", text.lower())
    sentences = [
        re.findall(r"[\w']+", sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]
    fillers = Counter(
        word for word in words if word in {"um", "uh", "ah", "erm", "hmm", "mmm"}
    )
    repeated_words = Counter(
        words[index]
        for index in range(len(words) - 1)
        if words[index] == words[index + 1] and len(words[index]) > 1
    )
    repeated_phrases = Counter(
        " ".join(words[index:index + 2])
        for index in range(len(words) - 3)
        if words[index:index + 2] == words[index + 2:index + 4]
    )
    you_know_count = sum(
        words[index:index + 2] == ["you", "know"]
        for index in range(len(words) - 1)
    )
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "average_sentence_words": (
            round(sum(map(len, sentences)) / len(sentences), 1) if sentences else 0
        ),
        "filler_counts": dict(fillers),
        "repeated_words": dict(repeated_words),
        "repeated_phrases": dict(repeated_phrases),
        "you_know_count": you_know_count,
    }


def _resolve_ffmpeg_executable():
    """Return a usable FFmpeg executable or raise an actionable error."""
    candidates = []
    try:
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, OSError, RuntimeError):
        pass

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        candidates.append(system_ffmpeg)

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)

    raise FileNotFoundError(
        "FFmpeg was not found. Install it or run: "
        "py -3.13 -m pip install imageio-ffmpeg. On Windows, ensure ffmpeg.exe "
        "is available on PATH."
    )


def analyze_speech_audio(video_source, progress_callback=None):
    """Transcribe local audio in fixed-size chunks to bound working memory."""
    with tempfile.TemporaryDirectory(prefix="auradev-audio-") as work_dir:
        work_path = Path(work_dir)
        if isinstance(video_source, bytes):
            source_path = work_path / "source.mp4"
            source_path.write_bytes(video_source)
        else:
            source_path = Path(video_source)
            if not source_path.is_file():
                raise FileNotFoundError(f"Media file does not exist: {source_path}")

        audio_path = work_path / "audio.wav"
        try:
            ffmpeg_exe = _resolve_ffmpeg_executable()
        except FileNotFoundError as error:
            raise RuntimeError(str(error)) from error

        command = [
            ffmpeg_exe,
            "-nostdin",
            "-y",
            "-threads",
            "1",
            "-i",
            str(source_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]
        if progress_callback:
            progress_callback("Extracting audio", 0, 1)
        try:
            extraction = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                f"Could not run FFmpeg to extract audio. {error}. "
                "Install FFmpeg or ensure it is on PATH."
            ) from error
        if extraction.returncode != 0 or not audio_path.is_file():
            details = extraction.stderr[-1000:].strip()
            raise RuntimeError(f"Could not extract an audio track. {details}")
        if progress_callback:
            progress_callback("Extracting audio", 1, 1)

        has_memory, available_percent = check_ram_availability()
        if not has_memory:
            raise MemoryError(
                f"Speech analysis paused to preserve memory. "
                f"{available_percent}% RAM is available; "
                f"{MIN_FREE_RAM_PERCENT}% is required."
            )

        if progress_callback:
            progress_callback("Loading speech model", 0, 1)
        try:
            model = whisper.load_model(WHISPER_MODEL_SIZE)
        except (
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
            urllib.error.URLError,
        ) as error:
            raise RuntimeError(
                f"Could not load the local Whisper '{WHISPER_MODEL_SIZE}' model. "
                f"Check that the model is available and enough memory is free. {error}"
            ) from error
        if progress_callback:
            progress_callback("Loading speech model", 1, 1)
        transcript_parts = []
        total_duration = 0.0

        try:
            audio_file = wave.open(str(audio_path), "rb")
        except (wave.Error, EOFError) as error:
            raise RuntimeError(
                "The uploaded video does not contain a readable audio track."
            ) from error

        with audio_file:
            sample_rate = audio_file.getframerate()
            frames_per_chunk = sample_rate * TRANSCRIPTION_CHUNK_SECONDS

            if sample_rate <= 0 or frames_per_chunk <= 0:
                raise RuntimeError("The extracted audio has an invalid sample rate.")
            if audio_file.getnframes() == 0:
                raise RuntimeError(
                    "No audio track was found in the video. Upload a video with "
                    "recorded sound, or use Live Real-Time Webcam mode to analyze "
                    "speech separately."
                )

            total_chunks = (
                audio_file.getnframes() + frames_per_chunk - 1
            ) // frames_per_chunk
            chunk_index = 0
            if progress_callback:
                progress_callback("Transcribing audio chunks", 0, total_chunks)

            while True:
                frames = audio_file.readframes(frames_per_chunk)
                if not frames:
                    break

                has_memory, available_percent = check_ram_availability()
                if not has_memory:
                    raise MemoryError(
                        f"Speech analysis paused between audio chunks to preserve "
                        f"memory. {available_percent}% RAM is available; "
                        f"{MIN_FREE_RAM_PERCENT}% is required."
                    )

                chunk_path = work_path / f"chunk-{chunk_index:06d}.wav"
                with wave.open(str(chunk_path), "wb") as chunk_file:
                    chunk_file.setnchannels(audio_file.getnchannels())
                    chunk_file.setsampwidth(audio_file.getsampwidth())
                    chunk_file.setframerate(sample_rate)
                    chunk_file.writeframes(frames)

                absolute_chunk_path = str(chunk_path.resolve())

                try:
                    result = model.transcribe(
                        absolute_chunk_path,
                        fp16=False,
                        condition_on_previous_text=False,
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    raise RuntimeError(
                        f"Speech transcription failed on audio chunk "
                        f"{chunk_index + 1} of {total_chunks}. {error}"
                    ) from error
                transcript_part = result.get("text", "").strip()
                if transcript_part:
                    transcript_parts.append(transcript_part)

                total_duration += len(frames) / (
                    audio_file.getsampwidth()
                    * audio_file.getnchannels()
                    * sample_rate
                )
                
                try:
                    chunk_path.unlink()
                except OSError:
                    pass

                chunk_index += 1
                if progress_callback:
                    progress_callback(
                        "Transcribing audio chunks", chunk_index, total_chunks
                    )

        transcript = " ".join(transcript_parts).strip()
        if not transcript:
            return "No clear spoken words were detected in the audio track.", 0.0, 0

        words_count = len(transcript.split())
        wpm = (words_count / total_duration) * 60.0 if total_duration > 0 else 0.0
        return transcript, round(wpm, 2), _count_hesitations(transcript)