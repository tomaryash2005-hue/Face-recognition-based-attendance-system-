"""Train an LBPH face recognizer on the captured dataset.

Usage:
    python src/train_model.py
"""

import json
import sys

import cv2
import numpy as np

from config import DATASET_DIR, LABELS_PATH, MODEL_PATH
import db


def load_training_data():
    faces, labels = [], []
    for student_dir in sorted(DATASET_DIR.iterdir()):
        if not student_dir.is_dir():
            continue
        try:
            student_id = int(student_dir.name)
        except ValueError:
            continue

        for image_path in student_dir.glob("*.jpg"):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            faces.append(image)
            labels.append(student_id)

    return faces, labels


def train():
    faces, labels = load_training_data()
    if not faces:
        print("No training images found. Run capture_faces.py for at least one student first.")
        sys.exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write(str(MODEL_PATH))

    students = {str(s["id"]): s["name"] for s in db.list_students()}
    LABELS_PATH.write_text(json.dumps(students, indent=2))

    print(f"Trained on {len(faces)} images across {len(set(labels))} students.")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
