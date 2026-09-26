"""Register a new student by capturing face samples from the webcam.

Usage:
    python src/capture_faces.py --roll 21CS001 --name "Jane Doe" --course "CS101"
"""

import argparse
import sys

import cv2

import db
from config import DATASET_DIR, FACE_SIZE, HAAR_CASCADE_PATH, SAMPLES_PER_STUDENT


def capture(roll_no: str, name: str, course: str, samples: int = SAMPLES_PER_STUDENT):
    db.init_db()
    if db.get_student_by_roll(roll_no):
        print(f"Roll number {roll_no} is already registered.")
        sys.exit(1)

    student_id = db.add_student(roll_no, name, course)
    student_dir = DATASET_DIR / str(student_id)
    student_dir.mkdir(parents=True, exist_ok=True)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + HAAR_CASCADE_PATH
    )
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open webcam. Check that a camera is connected and not in use.")
        sys.exit(1)

    print(f"Look at the camera. Capturing {samples} samples for {name} ({roll_no})...")
    count = 0
    try:
        while count < samples:
            ok, frame = cam.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)
                count += 1
                cv2.imwrite(str(student_dir / f"{count}.jpg"), face)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"Sample {count}/{samples}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                break  # one face per frame is enough

            cv2.imshow("Registering face - press q to abort", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    if count == 0:
        print("No face samples were captured; removing incomplete registration.")
        db_remove_incomplete_student(student_id, student_dir)
        sys.exit(1)

    print(f"Captured {count} samples. Run train_model.py before taking attendance.")


def db_remove_incomplete_student(student_id, student_dir):
    import shutil

    with db.get_connection() as conn:
        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
    shutil.rmtree(student_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Register a student's face for attendance.")
    parser.add_argument("--roll", required=True, help="Unique roll number / student ID")
    parser.add_argument("--name", required=True, help="Student's full name")
    parser.add_argument("--course", default="", help="Course or class section (optional)")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_STUDENT)
    args = parser.parse_args()
    capture(args.roll, args.name, args.course, args.samples)


if __name__ == "__main__":
    main()
