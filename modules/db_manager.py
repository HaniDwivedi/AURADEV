import sqlite3
from datetime import datetime

DB_NAME = "history.db"

def init_db():
    """Local SQLite Database table initialize karta hai."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            wpm REAL,
            slump_pct REAL,
            transcript TEXT,
            stammer_count INTEGER,
            syllabus_score INTEGER,
            feedback TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_session(wpm, slump_pct, transcript, stammer_count, syllabus_score, feedback):
    """Session ka data locally SQLite me store karta hai."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO session_history (timestamp, wpm, slump_pct, transcript, stammer_count, syllabus_score, feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (now, wpm, slump_pct, transcript, stammer_count, syllabus_score, feedback))
    conn.commit()
    conn.close()

def get_all_sessions():
    """Past saare sessions Fetch karta hai."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, wpm, slump_pct, stammer_count, syllabus_score, feedback FROM session_history ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows