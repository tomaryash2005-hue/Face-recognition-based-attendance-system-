"""Streamlit dashboard visualizing attendance captured by the recognition system.

Usage:
    streamlit run src/dashboard.py
"""

from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import db

# Fixed-order categorical palette (colorblind-considered: blue / orange / teal / purple).
CATEGORICAL = ["#3B7DD8", "#E8833A", "#2FA37A", "#8768C8", "#C94F6D", "#5AA9C9"]
SEQUENTIAL_SCALE = "Blues"          # single hue, light -> dark, for magnitude
STATUS_GOOD = "#2FA37A"             # present
STATUS_CRITICAL = "#C94F6D"         # absent
MUTED_TEXT = "#6B7280"

st.set_page_config(page_title="Attendance Dashboard", layout="wide")
db.init_db()


@st.cache_data(ttl=30)
def load_data():
    students = pd.DataFrame(db.list_students())
    attendance = pd.DataFrame(db.get_all_attendance())
    if not attendance.empty:
        attendance["date"] = pd.to_datetime(attendance["date"])
    return students, attendance


students_df, attendance_df = load_data()

st.title("Face Recognition Attendance — Dashboard")

if students_df.empty:
    st.info(
        "No students registered yet. Run `python src/capture_faces.py --roll ... --name ...`, "
        "then `python src/train_model.py`, then take attendance with `python src/attendance_system.py`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Filters (one row above the charts)
# ---------------------------------------------------------------------------
col_f1, col_f2, col_f3 = st.columns([2, 2, 3])
with col_f1:
    courses = ["All"] + sorted([c for c in students_df["course"].unique() if c])
    course_filter = st.selectbox("Course", courses)
with col_f2:
    default_start = (
        attendance_df["date"].min().date() if not attendance_df.empty else datetime.now().date() - timedelta(days=30)
    )
    date_range = st.date_input(
        "Date range",
        value=(default_start, datetime.now().date()),
    )
with col_f3:
    student_filter = st.multiselect(
        "Students",
        options=sorted(students_df["name"].tolist()),
        default=[],
        placeholder="All students",
    )

filtered = attendance_df.copy()
if course_filter != "All" and not filtered.empty:
    filtered = filtered[filtered["course"] == course_filter]
if student_filter and not filtered.empty:
    filtered = filtered[filtered["name"].isin(student_filter)]
if not filtered.empty and isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[(filtered["date"].dt.date >= start) & (filtered["date"].dt.date <= end)]

total_students = len(students_df)
sessions_logged = filtered["date"].nunique() if not filtered.empty else 0
today_present = (
    filtered[filtered["date"].dt.date == datetime.now().date()]["roll_no"].nunique()
    if not filtered.empty
    else 0
)
today_rate = f"{(today_present / total_students * 100):.0f}%" if total_students else "0%"

# ---------------------------------------------------------------------------
# KPI stat tiles
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Registered students", total_students)
k2.metric("Sessions logged", sessions_logged)
k3.metric("Present today", today_present)
k4.metric("Today's attendance rate", today_rate)

st.divider()

if filtered.empty:
    st.warning("No attendance records match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Daily attendance trend (magnitude over time -> single hue line)
# ---------------------------------------------------------------------------
daily_counts = filtered.groupby(filtered["date"].dt.date).size().reset_index(name="present_count")
daily_counts.columns = ["date", "present_count"]

fig_trend = go.Figure()
fig_trend.add_trace(
    go.Scatter(
        x=daily_counts["date"],
        y=daily_counts["present_count"],
        mode="lines+markers",
        line=dict(color=CATEGORICAL[0], width=2, shape="spline"),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor="rgba(59,125,216,0.12)",
        hovertemplate="%{x|%b %d, %Y}<br><b>%{y}</b> present<extra></extra>",
    )
)
fig_trend.update_layout(
    title="Daily attendance count",
    xaxis_title=None,
    yaxis_title="Students present",
    template="plotly_white",
    height=380,
    margin=dict(t=50, l=10, r=10, b=10),
    hovermode="x unified",
)
st.plotly_chart(fig_trend, use_container_width=True)

col_a, col_b = st.columns(2)

# ---------------------------------------------------------------------------
# Per-student attendance rate (magnitude, sorted, single hue)
# ---------------------------------------------------------------------------
with col_a:
    per_student = filtered.groupby(["roll_no", "name"]).size().reset_index(name="days_present")
    per_student["attendance_pct"] = (per_student["days_present"] / max(sessions_logged, 1) * 100).round(1)
    per_student = per_student.sort_values("attendance_pct", ascending=True)

    fig_bar = px.bar(
        per_student,
        x="attendance_pct",
        y="name",
        orientation="h",
        color="attendance_pct",
        color_continuous_scale=SEQUENTIAL_SCALE,
        labels={"attendance_pct": "Attendance %", "name": ""},
        hover_data={"roll_no": True, "days_present": True, "attendance_pct": ":.1f"},
    )
    fig_bar.update_layout(
        title="Attendance rate by student",
        template="plotly_white",
        height=max(320, 28 * len(per_student)),
        coloraxis_showscale=False,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# Attendance calendar heatmap (sequential single hue)
# ---------------------------------------------------------------------------
with col_b:
    cal = filtered.groupby(filtered["date"].dt.date).size().reset_index(name="count")
    cal.columns = ["date", "count"]
    cal["date"] = pd.to_datetime(cal["date"])
    cal["week"] = cal["date"].dt.strftime("%Y-W%U")
    cal["weekday"] = cal["date"].dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    fig_heat = px.density_heatmap(
        cal,
        x="week",
        y="weekday",
        z="count",
        category_orders={"weekday": weekday_order},
        color_continuous_scale=SEQUENTIAL_SCALE,
        labels={"count": "Present", "week": "Week", "weekday": ""},
    )
    fig_heat.update_layout(
        title="Attendance calendar heatmap",
        template="plotly_white",
        height=max(320, 28 * len(per_student)),
        margin=dict(t=50, l=10, r=10, b=10),
    )
    fig_heat.update_traces(hovertemplate="Week %{x}<br>%{y}<br><b>%{z}</b> present<extra></extra>")
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Present vs absent today (status colors, icon + label via legend)
# ---------------------------------------------------------------------------
present_rolls = set(
    attendance_df[attendance_df["date"].dt.date == datetime.now().date()]["roll_no"]
) if not attendance_df.empty else set()
absent_count = total_students - len(present_rolls)

fig_status = go.Figure(
    data=[
        go.Pie(
            labels=["Present today", "Absent today"],
            values=[len(present_rolls), max(absent_count, 0)],
            marker=dict(colors=[STATUS_GOOD, STATUS_CRITICAL]),
            hole=0.55,
            textinfo="label+percent",
        )
    ]
)
fig_status.update_layout(
    title="Today's status",
    template="plotly_white",
    height=340,
    margin=dict(t=50, l=10, r=10, b=10),
    showlegend=True,
)
col_c, col_d = st.columns([1, 2])
with col_c:
    st.plotly_chart(fig_status, use_container_width=True)

with col_d:
    st.subheader("Raw attendance log")
    st.dataframe(
        filtered[["date", "time", "roll_no", "name", "course", "confidence"]]
        .sort_values(["date", "time"], ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=340,
    )
