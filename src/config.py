"""Shared paths and tunables for the attendance system."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
MODEL_DIR = DATA_DIR / "models"

DATABASE_PATH = DATA_DIR / "attendance.db"
LABELS_PATH = MODEL_DIR / "labels.json"
MODEL_PATH = MODEL_DIR / "lbph_model.yml"

# Haar cascade shipped with opencv-contrib-python
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Number of face samples to capture per student during registration
SAMPLES_PER_STUDENT = 40

# LBPH prediction confidence threshold (lower = stricter match).
# Faces with a confidence score above this are labeled "Unknown".
RECOGNITION_CONFIDENCE_THRESHOLD = 70

# Minimum minutes between two attendance marks for the same student
# on the same day, to avoid duplicate rows if they linger in frame.
DUPLICATE_MARK_COOLDOWN_MINUTES = 0  # 0 = only once per day

FACE_SIZE = (200, 200)

for directory in (DATA_DIR, DATASET_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)
