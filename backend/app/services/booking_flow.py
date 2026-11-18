import re
from dataclasses import dataclass
from datetime import date

from .appointments import (
    book_appointment,
    cancel_appointment,
    check_availability,
    get_appointment,
    reschedule_appointment,
)
from .date_parser import parse_natural_date


def _affirmative(text: str) -> bool:
    text = text.lower().strip()
    text = re.sub(r"[.!?,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    positive = (
        r"(?:yes|yeah|yep|sure|okay|ok|confirm|"
        r"book it|go ahead|absolutely|correct|"
        r"that's correct|that's right|that is correct|"
        r"that is right|it's correct|it is correct)"
    )

    suffix = (
        r"(?:please|book it|confirm it|go ahead|"
        r"cancel it|correct|it's correct|it is correct|"
        r"that's correct|that's right|that is correct|"
        r"that is right)"
    )

    return bool(
        re.fullmatch(
            rf"{positive}(?: {suffix})*",
            text,
        )
    )


def _negative(text: str) -> bool:
    return bool(re.fullmatch(r"(?:no|no thanks|not yet|don't|do not)[\s,.!]*", text))


def _reference(text: str) -> int | None:
    """Extract a booking reference from natural speech."""

    words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }

    lower = text.lower().strip()
    lower = re.sub(r"[.,!?]+", " ", lower)
    lower = re.sub(r"\s+", " ", lower).strip()

    # Never interpret an appointment time as a reference.
    if re.search(r"\b\d{1,2}\s*(?:am|pm)\b", lower):
        return None

    # Accept natural conversational prefixes.
    prefix = (
        r"(?:(?:my|the)\s+)?"
        r"(?:(?:appointment|booking)\s+){0,2}"
        r"(?:reference|ref|number|id)"
        r"(?:\s+number)?"
        r"(?:\s+is)?"
        r"\s*[:#]?\s*"
    )

    lower = re.sub(rf"^{prefix}", "", lower).strip()

    # Also accept "it's 11" and "it is eleven".
    lower = re.sub(r"^it(?:'s| is)\s+", "", lower).strip()

    # Accept "number eleven" after conversational prefixes.
    lower = re.sub(r"^(?:number|no)\s+", "", lower).strip()

    # Accept digits of any length.
    if re.fullmatch(r"\d+", lower):
        value = int(lower)
        return value if value > 0 else None

    # Accept spoken numbers from one to twenty.
    spoken_value = words.get(lower)
    return spoken_value if spoken_value is not None and spoken_value > 0 else None


@dataclass
class BookingFlow:
    stage: str = "idle"
    name: str = ""
    appointment_date: str = ""
    appointment_time: str = ""
    reason: str = ""
    appointment_id: int | None = None
    action: str = ""
    name_attempts: int = 0
    new_date: str = ""
    new_time: str = ""

    def reset(self) -> None:
        self.stage = "idle"
        self.name = ""
        self.appointment_date = ""
        self.appointment_time = ""
        self.reason = ""
        self.appointment_id = None
        self.action = ""
        self.name_attempts = 0
        self.new_date = ""
        self.new_time = ""

    def handle(self, text: str) -> str | None:
        text = text.strip()
        lower = text.lower()

        # Ignore known STT prompt leakage.
        # Keep this narrow to avoid rejecting genuine speech.
        leaked_prompts = {
            "a conversation with a uk gp clinic receptionist",
            "a conversation with a uk gp clinic receptionist.",
        }

        if lower.strip() in leaked_prompts:
            return ""

        # Handle follow-up after a declined reschedule
        if self.stage == "reschedule_declined":
            self.reset()

            if re.search(
                r"\b(thank you|thanks|thank|okay|ok|cheers)\b",
                lower,
            ):
                return (
                    "You're very welcome! Your original "
                    "appointment remains unchanged. "
                    "Is there anything else I can help you with?"
                )

            # Allow new requests to be processed normally

        # Global polite conversation ending.
        # Do not treat "no" as a goodbye because it is
        # also used to decline booking changes.
        if re.fullmatch(
            r"(?:bye|goodbye|bye bye|see you|"
            r"that's all|end conversation|"
            r"no thanks|no thank you|"
            r"nothing else|that will be all)"
            r"[\s,.!]*",
            lower,
        ):
            # Keep "no thanks" available as a negative
            # answer during confirmation stages.
            if self.stage not in {
                "reschedule_confirm",
                "cancel_confirm",
                "confirm",
            }:
                self.reset()
                return "You're very welcome. Thank you for calling MediVoice AI. Have a lovely day!"

            self.reset()
            return "Goodbye! Thank you for calling MediVoice. Take care."

        if self.stage == "idle":
            # End the conversation when the patient has
            # no further requests.
            if re.fullmatch(
                r"(?:no|nope|nah|nothing|nothing else|"
                r"no thanks|no thank you|that's all|"
                r"that is all)[\s,.!]*",
                lower,
            ):
                self.reset()
                return (
                    "You're very welcome. Thank you for "
                    "calling MediVoice Clinic. Have a lovely day!"
                )

            # Respond naturally after completing an appointment.
            if re.fullmatch(
                r"(?:thank you|thanks|thank you so much|"
                r"thanks a lot|cheers|okay|ok)"
                r"[\s,.!]*",
                lower,
            ):
                return "You're very welcome! Is there anything else I can help you with today?"

            if re.search(
                r"\b(reschedul\w*|re[\s-]?schedul\w*|"
                r"move|change|shift|postpone)\b"
                r".*\b(appointment|booking)\b"
                r"|\b(appointment|booking)\b"
                r".*\b(reschedul\w*|re[\s-]?schedul\w*|"
                r"move|change|shift|postpone)\b",
                lower,
            ):
                self.action = "reschedule"
                self.stage = "manage_reference"
                ref = _reference(lower)
                if ref is not None:
                    return self._manage_reference(ref)
                return "Certainly. What is your appointment booking reference?"
            if re.search(
                r"\b(cancel|delete)\b.*\b(appointment|booking)\b|\b(appointment|booking)\b.*\b(cancel|delete)\b",  # noqa: E501
                lower,
            ):
                self.action = "cancel"
                self.stage = "manage_reference"
                ref = _reference(lower)
                if ref is not None:
                    return self._manage_reference(ref)
                return "Certainly. What is your appointment booking reference?"

            if re.search(
                r"\b(check|find|look up|lookup|view|status|details)\b.*\b(appointment|booking)\b|\b(appointment|booking)\b.*\b(check|status|details)\b",  # noqa: E501
                lower,
            ):
                self.action = "check"
                self.stage = "manage_reference"
                ref = _reference(lower)
                if ref is not None:
                    return self._manage_reference(ref)
                return "Certainly. What is your appointment booking reference?"

            if lower in {
                "book",
                "book it",
                "book an appointment",
                "schedule an appointment",
                "make an appointment",
            } or re.search(
                r"\b(book|schedule|make)\b.*\bappointment\b"
                r"|\bappointment\b.*\b(book|schedule)\b",
                lower,
            ):
                self.stage = "name"
                return "Certainly. May I have your full name, please?"

            return None

        # Do not interpret a request to cancel an existing appointment as
        # an instruction to abandon the cancellation flow.
        if re.fullmatch(r"(?:stop|never mind|start over|abort)[\s,.!]*", lower):
            self.reset()
            return "I've stopped this request. No changes were made."

        if self.stage == "manage_reference":
            ref = _reference(lower)
            if ref is None:
                return "Please tell me your numeric booking reference, such as 5."
            return self._manage_reference(ref)

        if self.stage == "manage_name":
            # Remove natural speech prefixes before comparing the name.
            name = re.sub(
                r"^(?:the patient(?:'s)? name is|patient(?:'s)? name is|"
                r"the name is|my name is|i am|i'm|it's|this is)\s+",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip(" .,!?")

            appointment = get_appointment(self.appointment_id)
            if appointment is None:
                self.reset()
                return "I couldn't find that booking reference. No changes were made."

            expected = appointment["patient_name"].strip()
            # A simple name match is for a fictional demo only, not real authentication.
            matches = re.sub(r"\s+", "", name).casefold() == re.sub(r"\s+", "", expected).casefold()
            if not matches:
                self.name_attempts += 1
                if self.name_attempts >= 3:
                    self.reset()
                    return (
                        "I couldn't verify the patient name after three attempts. "
                        "Please start a new request. No changes were made."
                    )
                return (
                    "I couldn't match that patient name. Please say the full "
                    "name on the booking again. No changes were made."
                )

            if self.action == "check":
                self.reset()
                return (
                    f"Appointment {appointment['id']} is {appointment['status']}. "
                    f"It is for {appointment['patient_name']} on "
                    f"{appointment['appointment_date']} at {appointment['appointment_time']}. "
                    f"The recorded reason is {appointment['reason']}."
                )

            if appointment["status"] != "confirmed":
                self.reset()
                return (
                    f"Appointment {appointment['id']} is already "
                    f"{appointment['status']}. No changes were made."
                )

            if self.action == "reschedule":
                self.stage = "reschedule_date"
                return (
                    f"I found appointment {appointment['id']} for {appointment['patient_name']} "
                    f"on {appointment['appointment_date']} at {appointment['appointment_time']}. "
                    "What new date would you prefer?"
                )

            self.stage = "cancel_confirm"
            return (
                f"I found the confirmed appointment for {appointment['patient_name']} "
                f"on {appointment['appointment_date']} at {appointment['appointment_time']}. "
                "Are you sure you want to cancel this demo appointment?"
            )

        if self.stage == "reschedule_date":
            # Ask for clarification if speech contains
            # more than one possible calendar date.
            date_pattern = (
                r"\b\d{1,2}(?:st|nd|rd|th)?[\s,]+"
                r"(?:january|february|march|april|may|"
                r"june|july|august|september|october|"
                r"november|december)\b"
            )

            mentioned_dates = re.findall(date_pattern, lower)

            if len(mentioned_dates) > 1:
                return (
                    "I heard more than one possible date. "
                    "Could you repeat just the final date, "
                    "including the day, month, and year?"
                )

            today = date.today()
            parsed = parse_natural_date(text, today=today)

            if parsed is None:
                return "Please say the new date, for example, 20 November 2026."
            if parsed < today:
                return "That date is in the past. Please choose a future date."
            if parsed.weekday() >= 5:
                return "The demo clinic is closed on weekends. Please choose a weekday."
            self.new_date = parsed.isoformat()
            self.stage = "reschedule_time"
            return "What new time would you prefer?"

        if self.stage == "reschedule_time":
            match = re.search(
                r"(?<!\d)(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?!\w)"
                r"|(?<!\d)(\d{1,2}):(\d{2})(?!\d)",
                lower,
            )
            if not match:
                return "Please say a specific time, such as 10 AM or 14:00."
            if match.group(1) is not None:
                hour = int(match.group(1))
                minute = int(match.group(2) or 0)
                meridiem = match.group(3).replace(".", "")
                if not 1 <= hour <= 12:
                    return "Please tell me a valid time."
                hour = hour % 12 + (12 if meridiem == "pm" else 0)
            else:
                hour = int(match.group(4))
                minute = int(match.group(5))
            if hour > 23 or minute > 59:
                return "Please tell me a valid time."
            new_time = f"{hour:02d}:{minute:02d}"
            current = get_appointment(self.appointment_id)
            if current is None or current["status"] != "confirmed":
                self.reset()
                return (
                    "This appointment is no longer available to reschedule. No changes were made."
                )
            if (current["appointment_date"], current["appointment_time"]) == (
                self.new_date,
                new_time,
            ):
                return "That is already your current appointment time. Please choose another time."
            available, message = check_availability(self.new_date, new_time)
            if not available:
                return f"{message} Please choose another time."
            self.new_time = new_time
            self.stage = "reschedule_confirm"
            return (
                f"You want to move appointment {self.appointment_id} to "
                f"{self.new_date} at {self.new_time}. "
                "Should I confirm this change?"
            )

        if self.stage == "reschedule_confirm":
            if _affirmative(lower) or re.fullmatch(
                r"(?:yes,?\s*)?reschedule(?: it)?[\s,.!]*", lower
            ):
                result = reschedule_appointment(self.appointment_id, self.new_date, self.new_time)
                self.reset()
                return str(result["message"])

            if _negative(lower):
                self.reset()
                self.stage = "reschedule_declined"
                return "Okay, your original appointment remains unchanged."
            return "Please say yes to reschedule, or no to keep your original appointment."

        if self.stage == "cancel_confirm":
            if _affirmative(lower) or bool(
                re.fullmatch(r"(?:yes,?\s*)?cancel(?: it| appointment)?[\s,.!]*", lower)
            ):
                result = cancel_appointment(self.appointment_id)
                self.reset()
                return str(result["message"])
            if _negative(lower):
                self.reset()
                return "Okay, your appointment has not been cancelled."
            return "Please say yes to cancel this appointment, or no to keep it."

        if re.fullmatch(r"(?:cancel|stop|never mind)[\s,.!]*", lower):
            self.reset()
            return "I've stopped this booking request. No appointment was created."

        if self.stage == "name":
            name = re.sub(
                r"^(?:my (?:full )?name is|"
                r"the (?:patient(?:'s)? )?name is|"
                r"i am|i'm|it's|this is)\s+",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip(" .,!?")

            if len(name) < 2 or len(name) > 80:
                return "Could you please tell me your full name?"

            self.name = name
            self.stage = "name_confirm"

            return f"I heard your name as {self.name}. Is that correct? Please say yes or no."

        if self.stage == "name_confirm":
            if _affirmative(lower):
                self.stage = "date"
                return "Thank you. Which date would you prefer?"

            if _negative(lower):
                self.name = ""
                self.stage = "name_spell"
                return (
                    "No problem. Please spell your full name "
                    "slowly, one letter at a time. "
                    "You can say the first and last names separately."
                )

            return (
                f"I heard {self.name}. "
                "Please say yes if that is correct, "
                "or no if you would like to correct it."
            )

        if self.stage == "name_spell":
            spoken = text.strip()

            # Accept letters separated by spaces or hyphens.
            letters = re.findall(r"[A-Za-z]", spoken)

            # Only accept a spelling if every token is
            # a single letter, separated by spaces or hyphens.
            tokens = re.split(r"[\s-]+", spoken.strip(" .,!?"))
            valid = len(tokens) >= 2 and all(
                len(token) == 1 and token.isalpha() for token in tokens
            )

            if not valid:
                return (
                    "I couldn't reliably understand the spelling. "
                    "Please say one letter at a time, "
                    "with a short pause between letters."
                )

            self.name = "".join(letters).title()
            self.stage = "name_confirm"

            return f"I have spelled your name as {self.name}. Is that correct?"

        if self.stage == "date":
            today = date.today()
            parsed = parse_natural_date(text, today=today)
            if parsed is None:
                return "Please say the appointment date, for example, 17 November 2026."
            if parsed < today:
                return "That date is in the past. Please choose a future appointment date."
            if parsed.weekday() >= 5:
                return "Our demo clinic is closed on weekends. Please choose a weekday."
            self.appointment_date = parsed.isoformat()
            self.stage = "time"
            return "What time would you prefer?"

        if self.stage == "time":
            match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", lower)
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
            available, message = check_availability(self.appointment_date, appointment_time)
            if not available:
                return f"{message} Please choose another time."
            self.appointment_time = appointment_time
            self.stage = "reason"
            return "What is the reason for your appointment?"

        if self.stage == "reason":
            if not text:
                return "Could you briefly describe the reason for your visit?"

            clean_reason = re.sub(
                r"^(?:the reason for (?:my |the )?appointment is|"
                r"the reason is|my reason is|"
                r"i need an appointment for)\s*",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip(" .,!?")

            if not clean_reason:
                return "Could you briefly describe the reason for your visit?"

            if re.fullmatch(
                r"(?:thank you|thanks|thank you so much|"
                r"thanks a lot|okay|ok|yes|no|"
                r"hello|hi|bye|goodbye|cheers|"
                r"please|sure|that's all)"
                r"[\s,.!]*",
                clean_reason,
                flags=re.IGNORECASE,
            ):
                return (
                    "Could you briefly tell me the reason for "
                    "your appointment, such as a general "
                    "consultation or routine check-up?"
                )

            self.reason = clean_reason[:200]
            self.stage = "confirm"

            return (
                f"I have {self.name}, on {self.appointment_date}, "
                f"at {self.appointment_time}, for {self.reason}. "
                "Shall I book this demo appointment?"
            )

        if self.stage == "confirm":
            if _affirmative(lower):
                result = book_appointment(
                    self.name, self.appointment_date, self.appointment_time, self.reason
                )
                self.reset()
                if result["success"]:
                    return f"Your demo appointment is confirmed. Your booking reference is {result['appointment_id']}."  # noqa: E501
                return f"I couldn't complete the booking. {result['message']}"
            if _negative(lower):
                self.reset()
                return "No problem. I haven't booked anything."
            return "Please say yes to confirm, or no to cancel."
        return None

    def _manage_reference(self, appointment_id: int) -> str:
        appointment = get_appointment(appointment_id)
        if appointment is None:
            self.reset()
            return "I couldn't find that booking reference. No changes were made."
        self.appointment_id = appointment_id
        self.name_attempts = 0
        self.stage = "manage_name"
        return "Please tell me the patient name on that booking."
