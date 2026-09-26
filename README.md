# Face Recognition Based Attendance System

A college-semester project that marks classroom attendance automatically using
face recognition, and visualizes the resulting attendance data on an
interactive dashboard.

## How it works

1. **Register** each student — the webcam captures ~40 face samples, which are
   stored under `data/dataset/<student_id>/`.
2. **Train** an LBPH (Local Binary Patterns Histograms) face recognizer on the
   captured samples. OpenCV's Haar Cascade handles face *detection* first;
   LBPH then handles *recognition* (matching a detected face to a known
   student). This combo needs no GPU and no dlib/cmake build step, so it
   installs cleanly on a lab machine.
3. **Take attendance** — run the live recognizer against the webcam feed.
   Each recognized student is marked present in a local SQLite database, once
   per day.
4. **Visualize** — a Streamlit + Plotly dashboard reads the same database and
   shows attendance trends, per-student rates, a calendar heatmap, and a
   present/absent breakdown.

## Project structure

```
src/
  config.py                Paths and tunables (thresholds, sample counts)
  db.py                     SQLite schema + queries (students, attendance)
  capture_faces.py          Webcam face registration for a new student
  train_model.py            Trains the LBPH recognizer on data/dataset/
  attendance_system.py      Live recognition loop; marks attendance
  dashboard.py              Streamlit dashboard (data visualization)
  generate_sample_data.py   Backfills fake attendance so the dashboard can be
                             demoed without a webcam
data/
  dataset/                  Captured face images per student (gitignored)
  models/                   Trained LBPH model + label map (gitignored)
  attendance.db             SQLite database (gitignored, created on first run)
tests/
  test_db.py                Unit tests for the database layer
```

## Setup

Requires a webcam for registration and live attendance; the dashboard alone
does not need one.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Register students** (repeat once per student):

```bash
python src/capture_faces.py --roll 21CS001 --name "Jane Doe" --course "CS101"
```

Look at the camera; a green box confirms each captured sample. Press `q` to
abort early.

**2. Train the recognizer** (re-run any time you add/remove a student):

```bash
python src/train_model.py
```

**3. Take attendance:**

```bash
python src/attendance_system.py
```

Recognized faces get a green box with the student's name and are marked
present for today. Unrecognized faces show a red "Unknown" box and are not
marked. Press `q` to end the session.

**4. View the dashboard:**

```bash
streamlit run src/dashboard.py
```

Opens at `http://localhost:8501` with:

- KPI tiles — registered students, sessions logged, present today, today's
  attendance rate
- Daily attendance trend line
- Per-student attendance rate (horizontal bar, sorted)
- Attendance calendar heatmap (by week/weekday)
- Present vs. absent donut for today
- Filterable raw attendance log (by course, date range, student)

### Trying the dashboard without a webcam

To explore the dashboard's visualizations without registering real students:

```bash
python src/generate_sample_data.py --days 45
streamlit run src/dashboard.py
```

This inserts attendance history for 8 fictional students. It does **not**
create face data, so those students can't actually be recognized by the
camera — it's for exercising the dashboard/report only. Delete
`data/attendance.db` to start over with real data.

## Tuning

Key constants live in `src/config.py`:

- `SAMPLES_PER_STUDENT` — face images captured per registration (default 40)
- `RECOGNITION_CONFIDENCE_THRESHOLD` — LBPH distance cutoff; lower is
  stricter. Faces above this are labeled "Unknown" instead of a false match.
- `FACE_SIZE` — the size faces are normalized to before training/matching.

## Running tests

```bash
pip install pytest
pytest tests/
```

`test_db.py` covers the database layer (student registration, one-mark-per-day
attendance, duplicate roll number rejection) against a temporary SQLite file —
no camera or trained model required.

## Notes for the project report

- **Detection**: Haar Cascade (`haarcascade_frontalface_default.xml`, shipped
  with OpenCV).
- **Recognition**: LBPH (`cv2.face.LBPHFaceRecognizer_create`), chosen over
  deep-learning embeddings (e.g. `face_recognition`/dlib or FaceNet) for this
  project because it trains in seconds on a CPU with no external model
  download, at the cost of being less robust to lighting/pose changes than a
  deep embedding — a good tradeoff to call out as a limitation and possible
  extension in your writeup.
- **Storage**: SQLite, so the whole system runs off a single file with no
  external database server.
- **Visualization**: Streamlit for the app shell, Plotly for interactive
  charts (hover tooltips, zoom) rather than static matplotlib images.
