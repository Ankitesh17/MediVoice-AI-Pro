
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
import sqlite3


def reschedule_appointment(appointment_id, appointment_date, appointment_time):
    """Move a confirmed demo booking without changing its reference."""
    init_db()
    try:
        appointment_id = int(appointment_id)
    except (TypeError, ValueError):
        return {"success": False, "message": "Invalid booking reference."}

    # Use one write transaction to protect against competing bookings.
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT appointment_date, appointment_time, status "
            "FROM appointments WHERE id = ?", (appointment_id,)
        ).fetchone()
        if row is None:
            return {"success": False, "message": "Booking reference not found."}
        old_date, old_time, status = row
        if status != "confirmed":
            return {"success": False, "message": "Only confirmed appointments can be rescheduled."}
        if (old_date, old_time) == (appointment_date, appointment_time):
            return {"success": False, "message": "That is already your current appointment slot."}

        # check_availability opens a separate connection, so check validation
        # and occupancy here within the same write transaction.
        try:
            requested_date = date.fromisoformat(appointment_date)
            requested_time = datetime.strptime(appointment_time, "%H:%M").time()
        except (TypeError, ValueError):
            return {"success": False, "message": "Invalid date or time format."}
        if requested_date.weekday() >= 5:
            return {"success": False, "message": "The demo clinic is closed on weekends."}
        if datetime.combine(requested_date, requested_time) <= datetime.now():
            return {"success": False, "message": "That appointment time has already passed."}
        if appointment_time not in AVAILABLE_TIMES:
            return {"success": False, "message": "That time is not an available demo slot."}
        taken = db.execute(
            "SELECT 1 FROM appointments WHERE appointment_date = ? "
            "AND appointment_time = ? AND status = 'confirmed' AND id != ?",
            (appointment_date, appointment_time, appointment_id),
        ).fetchone()
        if taken:
            return {"success": False, "message": "That appointment slot is already booked."}
        try:
            cursor = db.execute(
                "UPDATE appointments SET appointment_date = ?, appointment_time = ? "
                "WHERE id = ? AND status = 'confirmed'",
                (appointment_date, appointment_time, appointment_id),
            )
        except sqlite3.IntegrityError:
            return {"success": False, "message": "That appointment slot was just booked."}
        if cursor.rowcount != 1:
            return {"success": False, "message": "The appointment could not be updated."}
    return {
        "success": True,
        "appointment_id": appointment_id,
        "message": f"Appointment {appointment_id} has been rescheduled to {appointment_date} at {appointment_time}.",
    }
