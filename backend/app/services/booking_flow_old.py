
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .appointments import book_appointment, check_availability


@dataclass
class BookingFlow:
    stage: str = "idle"
    name: str = ""
    appointment_date: str = ""
    appointment_time: str = ""
    reason: str = ""

    def reset(self):
        self.stage = "idle"
        self.name = ""
        self.appointment_date = ""
        self.appointment_time = ""
        self.reason = ""

    def handle(self, text: str) -> str | None:
        text = text.strip()
        lower = text.lower()

        if self.stage == "idle":
            if re.search(
                r"\b(book|schedule|make)\b.*\bappointment\b"
                r"|\bappointment\b.*\b(book|schedule)\b",
                lower,
            ):
                self.stage = "name"
                return "Certainly. May I have your full name, please?"
            return None

        if re.search(r"\b(cancel|stop|never mind)\b", lower):
            self.reset()
            return "I've stopped this booking request. No appointment was created."

        if self.stage == "name":
            name = re.sub(
                r"^(my name is|i am|i'm|it's|this is)\s+",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip(" .,!")

            if len(name) < 2 or len(name) > 80:
                return "Could you please tell me your full name?"

            self.name = name
            self.stage = "date"
            return "Thank you. Which date would you prefer?"

        if self.stage == "date":
            today = date.today()
            parsed = None

            if "tomorrow" in lower:
                parsed = today + timedelta(days=1)
            elif "today" in lower:
                parsed = today
            else:
                match = re.search(
                    r"\b(\d{4}-\d{2}-\d{2})\b", text
                )
                if match:
                    try:
                        parsed = date.fromisoformat(match.group(1))
                    except ValueError:
                        pass

                if parsed is None:
                    match = re.search(
                        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
                        r"(january|february|march|april|may|june|"
                        r"july|august|september|october|november|december)"
                        r"(?:\s+(\d{4}))?\b",
                        lower,
                    )
                    if match:
                        day = int(match.group(1))
                        month = datetime.strptime(
                            match.group(2), "%B"
                        ).month
                        year = int(match.group(3) or today.year)
                        try:
                            parsed = date(year, month, day)
                        except ValueError:
                            pass

            if parsed is None:
                return (
                    "Please say the appointment date, "
                    "for example, 17 November 2026."
                )

            if parsed < today:
                return (
                    "That date is in the past. "
                    "Please choose a future appointment date."
                )

            if parsed.weekday() >= 5:
                return (
                    "Our demo clinic is closed on weekends. "
                    "Please choose a weekday."
                )

            self.appointment_date = parsed.isoformat()
            self.stage = "time"
            return "What time would you prefer?"

        if self.stage == "time":
            match = re.search(
                r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
                lower,
            )

            if not match:
                return "Please tell me a specific time, such as 10 AM."

            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            meridiem = match.group(3)

            if minute > 59:
                return "Please tell me a valid time."

            if meridiem:
                if not 1 <= hour <= 12:
                    return "Please tell me a valid time."
                hour = hour % 12 + (12 if meridiem == "pm" else 0)
            elif hour > 23:
                return "Please tell me a valid time."

            appointment_time = f"{hour:02d}:{minute:02d}"

            available, message = check_availability(
                self.appointment_date,
                appointment_time,
            )

            if not available:
                return f"{message} Please choose another time."

            self.appointment_time = appointment_time
            self.stage = "reason"
            return "What is the reason for your appointment?"

        
        if self.stage == "reason":
            if not text:
                return "Could you briefly describe the reason for your visit?"

            clean_reason = re.sub(
                r"^(the reason for (?:my |the )?appointment is|"
                r"my reason is|i need an appointment for)\s*",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip(" .,!")

            if not clean_reason:
                return "Could you briefly describe the reason for your visit?"

            self.reason = clean_reason[:200]
            self.stage = "confirm"

            return (
                f"I have {self.name}, on {self.appointment_date}, "
                f"at {self.appointment_time}, for {self.reason}. "
                "Shall I book this demo appointment?"
            )


        if self.stage == "confirm":
            if re.fullmatch(
             r"(yes|yeah|yep|sure|okay|ok|confirm|book it)"
             r"([,\s.!]*(please|book it|confirm it|go ahead))*[.!]*",
             lower,
        ):
                result = book_appointment(
                    self.name,
                    self.appointment_date,
                    self.appointment_time,
                    self.reason,
                )
                self.reset()

                if result["success"]:
                    return (
                        "Your demo appointment is confirmed. "
                        f"Your booking reference is {result['appointment_id']}."
                    )

                return (
                    f"I couldn't complete the booking. "
                    f"{result['message']}"
                )

            if lower in {"no", "no thanks", "not yet"}:
                self.reset()
                return "No problem. I haven't booked anything."

            return "Please say yes to confirm, or no to cancel."

        return None
