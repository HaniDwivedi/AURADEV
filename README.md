# 🚀 AuraDev

<div align="center">

**A modular, offline-first AI performance evaluator for tracking body posture, English speaking pace (WPM), hesitation patterns, and syllabus accuracy.**

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.41%2B-orange.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/Status-Active-success.svg)]()

</div>

---

## 📖 About AuraDev

AuraDev is an advanced, privacy-focused AI toolkit designed to evaluate real-time or uploaded video sessions. Utilizing local computer vision and speech models (OpenCV, MediaPipe, and OpenAI Whisper), AuraDev runs completely offline to analyze physical posture alignment, calculate speaking speed, and detect verbal fillers without compromising user data.

---

## ✨ Key Features

* **🤖 Local AI Speech-to-Text:** Transcribes audio chunks locally using OpenAI Whisper (`tiny`/`base`) with memory-safe execution blocks.
* **📏 Posture & Slump Tracking:** Analyzes body alignment via MediaPipe and OpenCV to flag slouching or posture fatigue in real-time.
* **🗣️ Speech & Hesitation Analytics:** Automatically calculates Words Per Minute (WPM) and tracks verbal fillers (e.g., *um, uh, you know*).
* **🔒 100% Offline Capability:** Operates securely on local machine hardware without sending sensitive video or audio streams to external cloud APIs.

---

## 🛠️ Tech Stack

* **Frontend & Dashboard:** Streamlit
* **Computer Vision:** OpenCV, MediaPipe
* **Speech Processing & Transcription:** OpenAI Whisper, `imageio-ffmpeg`, Python `wave`
* **Backend Utilities:** FastAPI, Uvicorn, SQLite

---

## ⚙️ Prerequisites & Installation

To run AuraDev locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/HaniDwivedi/AURADEV.git](https://github.com/HaniDwivedi/AURADEV.git)
cd AuraDev2. Set Up a Virtual Environment (Recommended)
Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
3. Install Dependencies
Install all required packages matching your environment:

Bash
pip install -r requirements.txt
4. Run the Application
Launch the dashboard via Streamlit:

Bash
streamlit run main.py
```
📋 Version Updates & Changelog
All notable updates to this project are documented below.

🌟 Version 1.0.0 (Current Release)
Core Architecture Initialized: Established modular structure for audio, video, and database managers (modules/).

Audio Pipeline Fixes: Resolved Windows subprocess file-locking issues ([WinError 2]) by integrating imageio_ffmpeg and absolute path resolution for audio chunking.

Memory Safeguards: Added real-time RAM availability checks (check_ram_availability) to gracefully pause transcription tasks when system thresholds are tight.

Git & Repository Maintenance: Cleaned up project tracking, excluded local SQLite database files, build caches, and __pycache__ directories via .gitignore.

