"""Run live face recognition and mark attendance for recognized students.

Usage:
    python src/attendance_system.py
Press 'q' to quit.
"""

import sys
from datetime import datetime

import cv2

import db
from config import (
    FACE_SIZE,
    HAAR_CASCADE_PATH,
    MODEL_PATH,
    RECOGNITION_CONFIDENCE_THRESHOLD,
)


def load_recognizer():
    if not MODEL_PATH.exists():
        print("No trained model found. Run capture_faces.py then train_model.py first.")
        sys.exit(1)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    return recognizer


def run():
    db.init_db()
    recognizer = load_recognizer()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + HAAR_CASCADE_PATH)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open webcam. Check that a camera is connected and not in use.")
        sys.exit(1)

    marked_this_session = set()
    print("Starting attendance session. Press 'q' to stop.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)
                label_id, confidence = recognizer.predict(face)

                student = db.get_student_by_label(label_id)
                if student and confidence <= RECOGNITION_CONFIDENCE_THRESHOLD:
                    display_name = f"{student['name']} ({student['roll_no']})"
                    color = (0, 255, 0)
                    if label_id not in marked_this_session:
                        marked = db.mark_attendance(label_id, confidence, datetime.now())
                        marked_this_session.add(label_id)
                        if marked:
                            print(f"Marked present: {display_name} [confidence {confidence:.1f}]")
                        else:
                            print(f"Already marked today: {display_name}")
                else:
                    display_name = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, display_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

            cv2.imshow("Attendance - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    print(f"Session ended. {len(marked_this_session)} student(s) recognized.")


if __name__ == "__main__":
    run()
