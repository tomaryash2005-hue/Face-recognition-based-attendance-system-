"""Populate the database with fake students and attendance history.

Useful for demoing or developing the dashboard without a webcam.
This does NOT create face encodings, so recognition won't work for
these students until you register them for real with capture_faces.py.

Usage:
    python src/generate_sample_data.py --days 30
"""

import argparse
import random
from datetime import datetime, timedelta

import db

SAMPLE_STUDENTS = [
    ("21CS001", "Aarav Sharma", "CS101"),
    ("21CS002", "Priya Patel", "CS101"),
    ("21CS003", "Rohan Gupta", "CS101"),
    ("21CS004", "Ananya Reddy", "CS101"),
    ("21CS005", "Vikram Singh", "CS102"),
    ("21CS006", "Sneha Iyer", "CS102"),
    ("21CS007", "Karan Mehta", "CS102"),
    ("21CS008", "Divya Nair", "CS102"),
]


def generate(days: int, attendance_probability: float = 0.85):
    db.init_db()

    student_ids = []
    for roll_no, name, course in SAMPLE_STUDENTS:
        existing = db.get_student_by_roll(roll_no)
        student_ids.append(existing["id"] if existing else db.add_student(roll_no, name, course))

    today = datetime.now().date()
    inserted = 0
    for day_offset in range(days, 0, -1):
        day = today - timedelta(days=day_offset)
        if day.weekday() >= 5:  # skip weekends
            continue
        for student_id in student_ids:
            if random.random() <= attendance_probability:
                when = datetime.combine(day, datetime.min.time()) + timedelta(
                    hours=9, minutes=random.randint(0, 30)
                )
                confidence = round(random.uniform(35, 65), 1)
                if db.mark_attendance(student_id, confidence, when):
                    inserted += 1

    print(f"Inserted {inserted} attendance records for {len(student_ids)} sample students.")


def main():
    parser = argparse.ArgumentParser(description="Generate sample attendance data for the dashboard.")
    parser.add_argument("--days", type=int, default=30, help="How many past days to backfill")
    parser.add_argument("--probability", type=float, default=0.85, help="Chance a student attends any given day")
    args = parser.parse_args()
    generate(args.days, args.probability)


if __name__ == "__main__":
    main()
