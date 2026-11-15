
import sqlite3
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "medivoice.db"

AVAILABLE_TIMES = ("09:00", "10:00", "11:00", "14:00", "15:00", "16:00")


def connect():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def init_db():
    with connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                appointment_date TEXT NOT NULL,
                appointment_time TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'confirmed',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            unique_confirmed_slot
            ON appointments(appointment_date, appointment_time)
            WHERE status = 'confirmed'
        """)


def check_availability(appointment_date, appointment_time):
    try:
        requested_date = date.fromisoformat(appointment_date)
        requested_time = datetime.strptime(
            appointment_time, "%H:%M"
        ).time()
    except (ValueError, TypeError):
        return False, "Invalid date or time format."

    if requested_date < date.today():
        return False, "You cannot book an appointment in the past."

    if requested_date.weekday() >= 5:
        return False, "The demo clinic is closed on weekends."

    if appointment_time not in AVAILABLE_TIMES:
        return False, "That time is not an available demo slot."

    if datetime.combine(requested_date, requested_time) <= datetime.now():
        return False, "That appointment time has already passed."

    with connect() as db:
        existing = db.execute(
            """
            SELECT id FROM appointments
            WHERE appointment_date = ?
            AND appointment_time = ?
            AND status = 'confirmed'
            """,
            (appointment_date, appointment_time),
        ).fetchone()

    if existing:
        return False, "That appointment slot is already booked."

    return True, "Appointment slot is available."


def book_appointment(patient_name, appointment_date,
                     appointment_time, reason):
    if not isinstance(patient_name, str) or not patient_name.strip():
        return {"success": False, "message": "Patient name is required."}

    if not isinstance(reason, str) or not reason.strip():
        return {"success": False, "message": "Appointment reason is required."}

    init_db()

    available, message = check_availability(
        appointment_date, appointment_time
    )

    if not available:
        return {"success": False, "message": message}

    try:
        with connect() as db:
            cursor = db.execute(
                """
                INSERT INTO appointments
                (patient_name, appointment_date, appointment_time, reason)
                VALUES (?, ?, ?, ?)
                """,
                (
                    patient_name.strip(),
                    appointment_date,
                    appointment_time,
                    reason.strip(),
                ),
            )
            appointment_id = cursor.lastrowid

        return {
            "success": True,
            "appointment_id": appointment_id,
            "message": "Demo appointment successfully booked.",
        }

    except sqlite3.IntegrityError:
        return {
            "success": False,
            "message": "That appointment slot was just booked.",
        }


def list_appointments():
    init_db()

    with connect() as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """
            SELECT * FROM appointments
            ORDER BY appointment_date, appointment_time
            """
        ).fetchall()

    return [dict(row) for row in rows]


def get_appointment(appointment_id):
    """Find an appointment using its booking reference."""
    init_db()

    try:
        appointment_id = int(appointment_id)
    except (TypeError, ValueError):
        return None

    with connect() as db:
        db.row_factory = sqlite3.Row

        row = db.execute(
            """
            SELECT * FROM appointments
            WHERE id = ?
            """,
            (appointment_id,),
        ).fetchone()

    return dict(row) if row else None


def cancel_appointment(appointment_id):
    """Cancel a confirmed appointment without deleting its record."""
    init_db()

    try:
        appointment_id = int(appointment_id)
    except (TypeError, ValueError):
        return {
            "success": False,
            "message": "Invalid booking reference.",
        }

    with connect() as db:
        cursor = db.execute(
            """
            UPDATE appointments
            SET status = 'cancelled'
            WHERE id = ?
            AND status = 'confirmed'
            """,
            (appointment_id,),
        )

        if cursor.rowcount == 1:
            return {
                "success": True,
                "appointment_id": appointment_id,
                "message": "Your demo appointment has been cancelled.",
            }

        row = db.execute(
            "SELECT status FROM appointments WHERE id = ?",
            (appointment_id,),
        ).fetchone()

    if row is None:
        message = "No appointment found with that booking reference."
    elif row[0] == "cancelled":
        message = "This appointment has already been cancelled."
    else:
        message = "This appointment cannot be cancelled."

    return {
        "success": False,
        "message": message,
    }
