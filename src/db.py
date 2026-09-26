"""SQLite persistence layer for students and attendance records."""

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime

from config import DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    course TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    confidence REAL,
    FOREIGN KEY (student_id) REFERENCES students (id),
    UNIQUE (student_id, date)
);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def add_student(roll_no: str, name: str, course: str = "") -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO students (roll_no, name, course, created_at) VALUES (?, ?, ?, ?)",
            (roll_no, name, course, datetime.now().isoformat(timespec="seconds")),
        )
        return cursor.lastrowid


def get_student_by_label(label_id: int):
    """label_id is the students.id used as the LBPH training label."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM students WHERE id = ?", (label_id,)).fetchone()
        return dict(row) if row else None


def get_student_by_roll(roll_no: str):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM students WHERE roll_no = ?", (roll_no,)).fetchone()
        return dict(row) if row else None


def list_students():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
        return [dict(r) for r in rows]


def mark_attendance(student_id: int, confidence: float, when: datetime = None) -> bool:
    """Insert one attendance row for today. Returns False if already marked today."""
    when = when or datetime.now()
    today = when.date().isoformat()
    with get_connection() as conn:
        try:
            conn.execute(
                "INSERT INTO attendance (student_id, date, time, confidence) VALUES (?, ?, ?, ?)",
                (student_id, today, when.strftime("%H:%M:%S"), confidence),
            )
            return True
        except sqlite3.IntegrityError:
            return False  # already marked for this student today


def get_attendance_for_date(day: date = None):
    day = day or date.today()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT a.date, a.time, a.confidence, s.roll_no, s.name, s.course
            FROM attendance a JOIN students s ON s.id = a.student_id
            WHERE a.date = ?
            ORDER BY a.time
            """,
            (day.isoformat(),),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_attendance():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT a.date, a.time, a.confidence, s.roll_no, s.name, s.course
            FROM attendance a JOIN students s ON s.id = a.student_id
            ORDER BY a.date, a.time
            """
        ).fetchall()
        return [dict(r) for r in rows]
