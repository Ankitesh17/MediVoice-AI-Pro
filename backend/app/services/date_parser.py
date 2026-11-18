
import re
from datetime import date, timedelta

MONTHS = {
    name: number
    for number, name in enumerate(
        [
            "january", "february", "march", "april",
            "may", "june", "july", "august",
            "september", "october", "november", "december",
        ],
        start=1,
    )
}

WEEKDAYS = {
    name: number
    for number, name in enumerate(
        ["monday", "tuesday", "wednesday", "thursday",
         "friday", "saturday", "sunday"]
    )
}

ORDINALS = {
    "first": 1, "second": 2, "third": 3,
    "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9,
    "tenth": 10, "eleventh": 11, "twelfth": 12,
    "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16,
    "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
    "twenty-first": 21, "twenty-second": 22,
    "twenty-third": 23, "twenty-fourth": 24,
    "twenty-fifth": 25, "twenty-sixth": 26,
    "twenty-seventh": 27, "twenty-eighth": 28,
    "twenty-ninth": 29, "thirtieth": 30,
    "thirty-first": 31,
}

DAY = r"(?:[0-9]{1,2}(?:st|nd|rd|th)?|" + "|".join(
    sorted(ORDINALS, key=len, reverse=True)
) + r")"

MONTH = r"(?:" + "|".join(MONTHS) + r")"


def parse_natural_date(
    text: str,
    today: date | None = None,
) -> date | None:
    """Parse common English appointment dates.

    Return None for unrecognized or invalid dates.
    Clinic availability is validated separately.
    """
    today = today or date.today()
    value = text.lower().strip()
    value = re.sub(r"\s+", " ", value)

    if re.search(r"\bday after tomorrow\b", value):
        return today + timedelta(days=2)

    if re.search(r"\btomorrow\b", value):
        return today + timedelta(days=1)

    if re.search(r"\btoday\b", value):
        return today

    # ISO dates, such as 2026-12-02.
    iso = re.search(r"\b\d{4}-\d{2}-\d{2}\b", value)
    if iso:
        try:
            return date.fromisoformat(iso.group())
        except ValueError:
            return None

    # Next Monday, this Friday, Monday.
    weekday = re.search(
        r"\b(?:(next|this)\s+)?"
        r"(monday|tuesday|wednesday|thursday|friday|"
        r"saturday|sunday)\b",
        value,
    )
    if weekday:
        modifier = weekday.group(1)
        target = WEEKDAYS[weekday.group(2)]
        days = (target - today.weekday()) % 7

        if modifier == "next":
            # "Next Monday" means the Monday of next
            # calendar week.
            days = 7 - today.weekday() + target
        elif days == 0:
            days = 7

        return today + timedelta(days=days)

    # Normalize spoken ordinal days.
    for word, number in sorted(
        ORDINALS.items(), key=lambda item: -len(item[0])
    ):
        value = re.sub(
            rf"\b{re.escape(word)}\b",
            str(number),
            value,
        )

    value = re.sub(r"\bof\b", " ", value)
    value = re.sub(r"[,]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    # 2 December 2026 or December 2 2026.
    day_first = re.search(
        rf"\b({DAY})\s+({MONTH})(?:\s+(\d{{4}}))?\b",
        value,
    )
    month_first = re.search(
        rf"\b({MONTH})\s+({DAY})(?:\s+(\d{{4}}))?\b",
        value,
    )

    if day_first:
        day_text = day_first.group(1)
        month_text = day_first.group(2)
        year_text = day_first.group(3)
    elif month_first:
        month_text = month_first.group(1)
        day_text = month_first.group(2)
        year_text = month_first.group(3)
    else:
        return None

    day_number = int(
        re.sub(r"(st|nd|rd|th)$", "", day_text)
    )
    month_number = MONTHS[month_text]
    year = int(year_text) if year_text else today.year

    try:
        result = date(year, month_number, day_number)
        if year_text is None and result < today:
            result = date(year + 1, month_number, day_number)
        return result
    except ValueError:
        return None
