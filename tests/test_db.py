import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import db  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db, "DATABASE_PATH", tmp_path / "test.db")
    db.init_db()
    yield


def test_add_and_fetch_student():
    student_id = db.add_student("21CS001", "Test Student", "CS101")
    student = db.get_student_by_roll("21CS001")
    assert student["id"] == student_id
    assert student["name"] == "Test Student"


def test_duplicate_roll_number_rejected():
    db.add_student("21CS001", "Test Student", "CS101")
    with pytest.raises(Exception):
        db.add_student("21CS001", "Another Student", "CS101")


def test_mark_attendance_once_per_day():
    student_id = db.add_student("21CS001", "Test Student", "CS101")
    when = datetime(2024, 1, 1, 9, 0, 0)

    assert db.mark_attendance(student_id, 42.0, when) is True
    assert db.mark_attendance(student_id, 50.0, when) is False  # duplicate same day

    records = db.get_attendance_for_date(when.date())
    assert len(records) == 1
    assert records[0]["roll_no"] == "21CS001"


def test_mark_attendance_on_different_days():
    student_id = db.add_student("21CS001", "Test Student", "CS101")
    day1 = datetime(2024, 1, 1, 9, 0, 0)
    day2 = datetime(2024, 1, 2, 9, 0, 0)

    assert db.mark_attendance(student_id, 42.0, day1) is True
    assert db.mark_attendance(student_id, 42.0, day2) is True

    all_records = db.get_all_attendance()
    assert len(all_records) == 2


def test_get_student_by_label_returns_none_for_unknown():
    assert db.get_student_by_label(9999) is None
