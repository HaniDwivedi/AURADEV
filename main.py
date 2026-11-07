import hashlib
import logging
import secrets
import time
from pathlib import Path

import streamlit as st
from modules.pose_analyzer import (
    analyze_video_posture,
    calculate_slump_percentage,
    open_live_webcam,
    process_live_webcam_frame,
)
from modules.audio_analyzer import analyze_speech_audio
from modules.audio_analyzer import get_transcript_insights
from modules.llm_coach import generate_coaching_feedback
from modules.utils import (
    check_ram_availability,
    get_ram_status,
    save_uploaded_file_in_chunks,
)
from modules.db_manager import init_db, save_session, get_all_sessions
from config import MIN_FREE_RAM_PERCENT
from modules.chunk_upload import (
    claim_completed_upload,
    discard_upload,
    finish_upload,
    get_upload_url,
    list_uploads,
    register_upload_session,
    start_upload_server,
)

logger = logging.getLogger(__name__)


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _render_transcript_insights(insights):
    columns = st.columns(3)
    columns[0].metric("Transcript words", insights["word_count"])
    columns[1].metric("Recognized sentences", insights["sentence_count"])
    columns[2].metric(
        "Average words per sentence", insights["average_sentence_words"]
    )

    hesitation_parts = [
        f"{word}: {count}"
        for word, count in sorted(insights["filler_counts"].items())
    ]
    hesitation_parts.extend(
        f'repeated "{word}": {count}'
        for word, count in sorted(insights["repeated_words"].items())
    )
    hesitation_parts.extend(
        f'repeated phrase "{phrase}": {count}'
        for phrase, count in sorted(insights["repeated_phrases"].items())
    )
    if insights["you_know_count"]:
        hesitation_parts.append(f'"you know": {insights["you_know_count"]}')
    if hesitation_parts:
            st.caption("Transcript signals: " + "; ".join(hesitation_parts))
    else:
        st.caption("No tracked filler words or repeated-word signals were detected.")
    st.caption(
        "These are transcript-based counts only; pronunciation, pauses, and "
        "emotional tone are not scored."
    )


def _report_download_text(report, transcript, insights):
    filler_lines = [
        f"- **{word}:** {count}"
        for word, count in sorted(insights["filler_counts"].items())
    ]
    filler_lines.extend(
        f'- **Repeated "{word}":** {count}'
        for word, count in sorted(insights["repeated_words"].items())
    )
    filler_lines.extend(
        f'- **Repeated phrase "{phrase}":** {count}'
        for phrase, count in sorted(insights["repeated_phrases"].items())
    )
    if insights["you_know_count"]:
        filler_lines.append(
            f'- **"You know" phrase:** {insights["you_know_count"]}'
        )
    if not filler_lines:
        filler_lines.append("- No tracked filler or repeated-word signals detected.")

    return (
        f"{report}\n\n---\n\n"
        "## Transcript metrics\n\n"
        f"- Words: {insights['word_count']}\n"
        f"- Recognized sentences: {insights['sentence_count']}\n"
        f"- Average words per sentence: {insights['average_sentence_words']}\n\n"
        "### Transcript signals\n\n"
        + "\n".join(filler_lines)
        + "\n\n"
        "## Transcript\n\n"
        f"{transcript}\n\n"
        "_Automated feedback is practice support, not a medical assessment. "
        "Transcript-based counts do not measure pronunciation or pauses._\n"
    )


# Initialize Database
init_db()

st.set_page_config(page_title="AuraDev - AI Analytics Dashboard", layout="wide")

st.title("⚡ AuraDev: Offline Posture, English & Syllabus Analyzer")
available_ram, total_ram, available_ram_percent = get_ram_status()
st.caption(
    f"Available RAM: {available_ram / (1024 ** 3):.1f} / "
    f"{total_ram / (1024 ** 3):.1f} GB ({available_ram_percent}%). "
    f"Analysis pauses below the {MIN_FREE_RAM_PERCENT}% free-memory reserve."
)

# Create Navigation Tabs
tab_analysis, tab_dashboard = st.tabs(["🎥 Live / File Analysis", "📊 Past Sessions Analytics"])

# ----------------- TAB 1: ANALYSIS WORKFLOW -----------------
with tab_analysis:
    col_input, col_config = st.columns([2, 1])
    
    with col_config:
        target_topic = st.text_input("Target Topic / Syllabus Context", placeholder="e.g. Operating Systems - Process Scheduling")
        mode = st.radio("Select Analysis Mode:", ["Upload Video File", "Live Real-Time Webcam"])

    session_defaults = {
        "live_camera": None,
        "live_active": False,
        "live_slump_frames": 0,
        "live_total_frames": 0,
        "live_error": None,
        "live_result": None,
        "live_current_pct": 0.0,
        "live_audio_digest": None,
        "upload_analysis_error": None,
    }
    for key, default in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if mode != "Live Real-Time Webcam" and st.session_state.live_active:
        st.session_state.live_camera.release()
        st.session_state.live_camera = None
        st.session_state.live_active = False

    with col_input:
        if mode == "Upload Video File":
            if "upload_token" not in st.session_state:
                st.session_state.upload_token = secrets.token_urlsafe(32)

            upload_port = None
            try:
                upload_port = start_upload_server()
                register_upload_session(st.session_state.upload_token)
                st.link_button(
                    "Choose a large video to upload",
                    get_upload_url(st.session_state.upload_token),
                    icon=":material/upload_file:",
                    type="primary",
                )
                st.caption(
                    "The local uploader sends 8 MiB chunks directly to disk, avoiding "
                    "Streamlit's whole-file upload buffer and its default 200 MiB limit. "
                    "Upload size is limited by available disk space, not a 30 MiB cap. "
                    f"Local upload service: 127.0.0.1:{upload_port}"
                )
            except (OSError, RuntimeError) as error:
                st.error(f"Could not start the local large-file uploader: {error}")

            def render_upload_result():
                result = st.session_state.get("upload_analysis_result")
                if result is None:
                    return

                st.subheader("📊 Session Breakdown")
                m1, m2, m3 = st.columns(3)
                m1.metric("Speaking Speed", f"{result['wpm']} WPM")
                m2.metric("Estimated Head-Tilt Frames", f"{result['slump_pct']}%")
                m3.metric("Filler / Repeated Words", f"{result['stammer_count']}")
                with st.expander("View Full Speech Transcript"):
                    st.write(result["transcript"])
                st.markdown("---")
                st.subheader("💡 Speech, Structure & Delivery Coaching")
                _render_transcript_insights(result["insights"])
                st.markdown(result["report"])
                st.download_button(
                    "Download full report",
                    data=_report_download_text(
                        result["report"], result["transcript"], result["insights"]
                    ),
                    file_name=f"{Path(result['filename']).stem}-report.md",
                    mime="text/markdown",
                    key="download_upload_analysis_report",
                )

            if "upload_token" in st.session_state and upload_port:
                @st.fragment(run_every="2s")
                def large_upload_controls():
                    uploads = list_uploads(st.session_state.upload_token)
                    completed = [
                        upload for upload in uploads if upload["status"] == "complete"
                    ]
                    active = [
                        upload for upload in uploads if upload["status"] == "uploading"
                    ]

                    for upload in active:
                        progress = (
                            upload["received"] / upload["size"]
                            if upload["size"]
                            else 0
                        )
                        st.progress(
                            progress,
                            text=(
                                f"Uploading {upload['filename']}: "
                                f"{upload['received'] / (1024 ** 3):.2f} / "
                                f"{upload['size'] / (1024 ** 3):.2f} GiB"
                            ),
                        )

                    if completed:
                        uploads_by_id = {
                            upload["upload_id"]: upload for upload in completed
                        }
                        selected_id = st.selectbox(
                            "Uploaded video",
                            options=list(uploads_by_id),
                            format_func=lambda upload_id: (
                                f"{uploads_by_id[upload_id]['filename']} "
                                f"({uploads_by_id[upload_id]['size'] / (1024 ** 3):.2f} GiB)"
                            ),
                            key="completed_video_upload",
                        )
                        saved_error = st.session_state.upload_analysis_error
                        if (
                            saved_error is not None
                            and saved_error["upload_id"] == selected_id
                        ):
                            st.error(
                                f"Analysis stopped: {saved_error['message']}",
                                icon=":material/error:",
                            )
                        if st.button(
                            "Analyze uploaded video",
                            type="primary",
                            key="analyze_large_video",
                        ):
                            st.session_state.upload_analysis_error = None
                            has_memory, current_ram_percent = check_ram_availability()
                            if not has_memory:
                                message = (
                                    f"Analysis not started: only {current_ram_percent}% "
                                    f"RAM is available; keep at least "
                                    f"{MIN_FREE_RAM_PERCENT}% free."
                                )
                                st.session_state.upload_analysis_error = {
                                    "upload_id": selected_id,
                                    "message": message,
                                }
                                st.error(message)
                            else:
                                claimed = False
                                analysis_succeeded = False
                                analysis_status = None
                                try:
                                    upload_path, filename = claim_completed_upload(
                                        st.session_state.upload_token,
                                        selected_id,
                                    )
                                    claimed = True
                                    with st.status(
                                        "Analyzing uploaded video...", expanded=True
                                    ) as analysis_status:
                                        progress_bar = st.progress(
                                            0, text="Preparing analysis progress..."
                                        )
                                        phase_started = {}

                                        def update_analysis_progress(
                                            stage, completed, total
                                        ):
                                            now = time.monotonic()
                                            started = phase_started.setdefault(
                                                stage, now
                                            )
                                            if total:
                                                fraction = min(
                                                    max(completed / total, 0.0), 1.0
                                                )
                                                percent = round(fraction * 100)
                                                if completed > 0 and fraction < 1:
                                                    elapsed = now - started
                                                    if elapsed >= 2:
                                                        remaining = (
                                                            elapsed
                                                            * (total - completed)
                                                            / completed
                                                        )
                                                        time_left = (
                                                            f"about {_format_duration(remaining)} remaining"
                                                        )
                                                    else:
                                                        time_left = "estimating time left"
                                                else:
                                                    time_left = (
                                                        "step complete"
                                                        if fraction >= 1
                                                        else "estimating time left"
                                                    )
                                                if (
                                                    stage == "Loading speech model"
                                                    and completed == 0
                                                ):
                                                    progress_text = (
                                                        "Loading local Whisper model. "
                                                        "On first use this may take a "
                                                        "few minutes."
                                                    )
                                                else:
                                                    progress_text = (
                                                        f"{stage}: {percent}% complete "
                                                        f"({completed:,}/{total:,}); "
                                                        f"{time_left}"
                                                    )
                                            else:
                                                fraction = 0
                                                progress_text = (
                                                    f"{stage}: {completed:,} frames "
                                                    "processed; total and time remaining "
                                                    "are unavailable"
                                                )
                                            progress_bar.progress(
                                                fraction, text=progress_text
                                            )

                                        slump_pct = analyze_video_posture(
                                            upload_path,
                                            progress_callback=update_analysis_progress,
                                        )
                                        status_text = st.empty()
                                        status_text.write(
                                            "Posture analysis complete. Starting audio transcription."
                                        )
                                        transcript, wpm, stammer_count = (
                                            analyze_speech_audio(
                                                upload_path,
                                                progress_callback=update_analysis_progress,
                                            )
                                        )
                                        progress_bar.progress(
                                            1,
                                            text=(
                                                "Audio transcription complete. "
                                                "Preparing the coaching report..."
                                            ),
                                        )
                                        status_text.write(
                                            "Generating the coaching report. "
                                            "The local model does not provide a reliable "
                                            "completion percentage or time estimate."
                                        )
                                        insights = get_transcript_insights(transcript)
                                        report = generate_coaching_feedback(
                                            transcript,
                                            wpm,
                                            slump_pct,
                                            stammer_count,
                                            target_topic,
                                        )
                                        analysis_status.update(
                                            label="Analysis complete",
                                            state="complete",
                                            expanded=False,
                                        )
                                    save_session(
                                        wpm,
                                        slump_pct,
                                        transcript,
                                        stammer_count,
                                        85,
                                        report,
                                    )
                                    st.session_state.upload_analysis_result = {
                                        "filename": filename,
                                        "wpm": wpm,
                                        "slump_pct": slump_pct,
                                        "transcript": transcript,
                                        "insights": insights,
                                        "stammer_count": stammer_count,
                                        "report": report,
                                    }
                                    analysis_succeeded = True
                                    st.success(
                                        "Analysis complete. Session saved locally."
                                    )
                                except (
                                    ImportError,
                                    MemoryError,
                                    OSError,
                                    RuntimeError,
                                    ValueError,
                                ) as error:
                                    logger.exception(
                                        "Uploaded-video analysis failed for %s",
                                        selected_id,
                                    )
                                    if analysis_status is not None:
                                        analysis_status.update(
                                            label="Analysis stopped",
                                            state="error",
                                            expanded=True,
                                        )
                                    st.session_state.upload_analysis_error = {
                                        "upload_id": selected_id,
                                        "message": str(error),
                                    }
                                    st.error(
                                        f"Analysis stopped: {error}",
                                        icon=":material/error:",
                                    )
                                finally:
                                    if claimed:
                                        finish_upload(
                                            st.session_state.upload_token,
                                            selected_id,
                                            remove=analysis_succeeded,
                                        )

                        if st.button("Delete uploaded video", key="discard_large_video"):
                            discard_upload(st.session_state.upload_token, selected_id)

                    if not active and not completed:
                        st.info(
                            "Choose a video in the uploader window. When the upload "
                            "finishes, it will appear here for analysis."
                        )
                    render_upload_result()

                if "upload_token" in st.session_state:
                    large_upload_controls()

        elif mode == "Live Real-Time Webcam":
            st.info(
                "Start and stop the webcam whenever you choose. Posture feedback updates "
                "during the session. Use the speech recorder's start/stop controls below; "
                "speech and fluency coaching appears after recording stops."
            )

            @st.fragment(run_every="100ms")
            def live_webcam_controls():
                if st.session_state.live_active:
                    stop_clicked = st.button("⏹ Stop Live Session", type="primary")
                    frame_placeholder = st.empty()

                    if stop_clicked:
                        st.session_state.live_camera.release()
                        st.session_state.live_camera = None
                        st.session_state.live_active = False
                        st.session_state.live_result = calculate_slump_percentage(
                            st.session_state.live_slump_frames,
                            st.session_state.live_total_frames,
                        )
                        st.session_state.live_current_pct = st.session_state.live_result
                    else:
                        frame, is_slumped, error = process_live_webcam_frame(
                            st.session_state.live_camera
                        )
                        if error:
                            st.session_state.live_camera.release()
                            st.session_state.live_camera = None
                            st.session_state.live_active = False
                            st.session_state.live_error = error
                        else:
                            st.session_state.live_total_frames += 1
                            if is_slumped:
                                st.session_state.live_slump_frames += 1
                            st.session_state.live_current_pct = calculate_slump_percentage(
                                st.session_state.live_slump_frames,
                                st.session_state.live_total_frames,
                            )
                            frame_placeholder.image(
                                frame,
                                channels="RGB",
                                use_container_width=True,
                            )
                            st.caption(
                                f"Analyzing live — "
                                f"{st.session_state.live_total_frames} frames processed"
                            )
                            ram_available, ram_total, ram_percent = get_ram_status()
                            metric_columns = st.columns(2)
                            metric_columns[0].metric(
                                "Estimated head-tilt frames",
                                f"{st.session_state.live_current_pct}%",
                            )
                            metric_columns[1].metric(
                                "Available RAM",
                                f"{ram_available / (1024 ** 3):.1f} GB ({ram_percent}%)",
                            )
                            if is_slumped:
                                st.warning(
                                    "Head tilt detected in this frame. Try centering "
                                    "your head over your shoulders and relaxing them."
                                )
                            else:
                                st.caption(
                                    "Head alignment looks steady in this frame. "
                                    "Keep your shoulders relaxed and feet grounded."
                                )
                else:
                    if st.button("🔴 Start Live Webcam Session", type="primary"):
                        camera, error = open_live_webcam()
                        if error:
                            st.session_state.live_error = error
                        else:
                            st.session_state.live_camera = camera
                            st.session_state.live_active = True
                            st.session_state.live_slump_frames = 0
                            st.session_state.live_total_frames = 0
                            st.session_state.live_current_pct = 0.0
                            st.session_state.live_error = None
                            st.session_state.live_result = None

                if st.session_state.live_error:
                    st.error(st.session_state.live_error)
                    st.session_state.live_error = None

                if st.session_state.live_result is not None:
                    slump_pct = st.session_state.live_result
                    st.success(
                        f"Live posture check complete. Estimated head-tilt frames: "
                        f"{slump_pct}%"
                    )
                    report = (
                        f"**Posture practice:** "
                        f"{'Try aligning your head over your shoulders and relaxing them.' if slump_pct > 20 else 'Your sampled head alignment was mostly steady; keep your shoulders relaxed and feet grounded.'} "
                        "This webcam estimate is a simple head/eye-alignment cue, not a medical assessment."
                    )
                    st.markdown(report)
                    save_session(0, slump_pct, "Posture-only live session.", 0, 90, report)
                    st.session_state.live_result = None

            live_webcam_controls()

            recorded_audio = st.audio_input(
                "Record a speech practice (use the recorder's start/stop controls)",
                sample_rate=16000,
                key="live_speech_audio",
            )
            if recorded_audio is not None:
                audio_digest = hashlib.sha256(recorded_audio.getbuffer()).hexdigest()
                if audio_digest != st.session_state.live_audio_digest:
                    has_memory, current_ram_percent = check_ram_availability()
                    if not has_memory:
                        st.error(
                            f"Speech analysis not started: only {current_ram_percent}% "
                            f"RAM is available; keep at least "
                            f"{MIN_FREE_RAM_PERCENT}% free."
                        )
                    else:
                        temp_audio_path = None
                        try:
                            temp_audio_path = save_uploaded_file_in_chunks(recorded_audio)
                            with st.spinner(
                                "Transcribing your recording in bounded audio chunks..."
                            ):
                                transcript, wpm, stammer_count = analyze_speech_audio(
                                    temp_audio_path
                                )

                            slump_pct = st.session_state.live_current_pct
                            report = generate_coaching_feedback(
                                transcript,
                                wpm,
                                slump_pct,
                                stammer_count,
                                target_topic,
                            )
                            insights = get_transcript_insights(transcript)
                            st.subheader("🎙️ Speech Practice Results")
                            m1, m2 = st.columns(2)
                            m1.metric("Speaking Speed", f"{wpm} WPM")
                            m2.metric("Filler / Repeated Words", f"{stammer_count}")
                            _render_transcript_insights(insights)
                            with st.expander("View Speech Transcript"):
                                st.write(transcript)
                            st.markdown(report)
                            st.download_button(
                                "Download full report",
                                data=_report_download_text(
                                    report, transcript, insights
                                ),
                                file_name="auradev-speech-practice-report.md",
                                mime="text/markdown",
                                key="download_live_speech_report",
                            )
                            save_session(
                                wpm, slump_pct, transcript, stammer_count, 85, report
                            )
                            st.session_state.live_audio_digest = audio_digest
                            st.success(
                                "Speech practice analyzed and saved locally. "
                                "Fluency exercises are practice support, not treatment."
                            )
                        except (MemoryError, OSError, RuntimeError) as error:
                            st.error(f"Speech analysis stopped: {error}")
                        finally:
                            if temp_audio_path is not None:
                                temp_audio_path.unlink(missing_ok=True)

# ----------------- TAB 2: LOCAL DASHBOARD HISTORY -----------------
with tab_dashboard:
    st.subheader("📈 Performance History & Metrics Dashboard")
    history_data = get_all_sessions()

    if not history_data:
        st.info("No saved session records found. Complete an analysis session first.")
    else:
        # Display latest overall metrics
        timestamps = [row[0] for row in history_data]
        wpms = [row[1] for row in history_data]
        slumps = [row[2] for row in history_data]
        stammers = [row[3] for row in history_data]

        d1, d2, d3 = st.columns(3)
        d1.metric("Avg Speaking Speed", f"{round(sum(wpms)/len(wpms), 1)} WPM")
        d2.metric("Avg Head-Tilt Estimate", f"{round(sum(slumps)/len(slumps), 1)}%")
        d3.metric("Total Hesitations Tracked", f"{sum(stammers)}")

        st.markdown("---")
        st.write("### Past Session Records")

        for row in history_data:
            with st.expander(f"Session on {row[0]} | WPM: {row[1]} | Head tilt: {row[2]}% | Hesitations: {row[3]}"):
                st.markdown(f"**AI Report:**\n{row[5]}")