import json
import logging
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger("mewshifts-local")

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is missing.")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=120.0,
    max_retries=1,
)

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".xls", ".xlsx"}
ALLOWED_PERMISSION_TIMES = {"shift start", "shift end", "during shift"}

SYSTEM_PROMPT = """
You extract attendance records from an uploaded PDF or Excel file.

Return STRICT JSON only in this exact schema:

{
  "days": [
    {
      "date": "YYYY-MM-DD",
      "attend1": "HH:mm" | null,
      "attend2": "HH:mm" | null,
      "attend3": "HH:mm" | null,
      "leave1": "HH:mm" | null,
      "leave2": "HH:mm" | null,
      "permission_time": "shift end" | "shift start" | "during shift" | null,
      "permission_period": "HH:mm" | null
    }
  ]
}

Rules:
- Include only real dates found in the document.
- Normalize all dates to YYYY-MM-DD.
- Normalize all times to HH:mm 24-hour format.
- If a value is missing, return null.
- Do not invent days.
- Do not output markdown.
- If you are uncertain, prefer null rather than guessing.
- The document may contain Arabic.
"""

TIME_RE = re.compile(r"^\d{2}:\d{2}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_time(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"null", "none", "nan", "-"}:
        return None

    match = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if match:
        hh = int(match.group(1))
        mm = int(match.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"

    if TIME_RE.fullmatch(s):
        hh, mm = map(int, s.split(":"))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return s

    return None


def normalize_duration(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"null", "none", "nan", "-"}:
        return None

    match = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if match:
        hh = int(match.group(1))
        mm = int(match.group(2))
        if hh >= 0 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"

    return None


def normalize_permission_time(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s in {"null", "none", "nan", "-"}:
        return None
    return s if s in ALLOWED_PERMISSION_TIMES else None


def normalize_date(value: Any) -> str:
    if value is None:
        raise ValueError("Missing date")

    s = str(value).strip()
    if not s:
        raise ValueError("Empty date")

    if DATE_RE.fullmatch(s):
        datetime.strptime(s, "%Y-%m-%d")
        return s

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    raise ValueError("Invalid date format: {}".format(s))


def extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_and_normalize_model_output(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object.")

    days = payload.get("days")
    if not isinstance(days, list):
        raise ValueError("Model output must contain a 'days' list.")

    normalized_days = []
    seen_dates = set()

    for index, item in enumerate(days):
        if not isinstance(item, dict):
            raise ValueError("days[{}] must be an object.".format(index))

        date_value = normalize_date(item.get("date"))
        attend1 = normalize_time(item.get("attend1"))
        attend2 = normalize_time(item.get("attend2"))
        attend3 = normalize_time(item.get("attend3"))
        leave1 = normalize_time(item.get("leave1"))
        leave2 = normalize_time(item.get("leave2"))
        permission_time = normalize_permission_time(item.get("permission_time"))
        permission_period = normalize_duration(item.get("permission_period"))

        if date_value in seen_dates:
            continue

        seen_dates.add(date_value)
        normalized_days.append(
            {
                "date": date_value,
                "attend1": attend1,
                "attend2": attend2,
                "attend3": attend3,
                "leave1": leave1,
                "leave2": leave2,
                "permission_time": permission_time,
                "permission_period": permission_period,
            }
        )

    normalized_days.sort(key=lambda x: x["date"])
    return {"days": normalized_days}


def pick_test_file(script_dir: Path) -> Path:
    for name in ("test.pdf", "test.xlsx", "test.xls"):
        path = script_dir / name
        if path.exists():
            return path
    raise FileNotFoundError("Put test.pdf or test.xlsx or test.xls beside local.py")


def upload_file_to_openai(file_path: Path) -> str:
    logger.info("Uploading file to OpenAI: %s", file_path.name)
    with open(file_path, "rb") as f:
        uploaded = client.files.create(
            file=f,
            purpose="user_data",
        )
    logger.info("Uploaded successfully. file_id=%s", uploaded.id)
    return uploaded.id


def ask_openai_to_parse_file(file_id: str) -> dict:
    logger.info("Sending file reference to OpenAI for parsing...")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": file_id,
                        },
                        {
                            "type": "input_text",
                            "text": SYSTEM_PROMPT,
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        print("\n" + "=" * 80)
        print("OPENAI ERROR")
        print("=" * 80)
        print(repr(e))
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                print("status_code:", resp.status_code)
                print("response_text:", resp.text)
            except Exception:
                pass
        print("=" * 80 + "\n")
        raise

    raw_output = response.output_text or ""

    print("\n" + "=" * 80)
    print("OPENAI RAW RESPONSE")
    print("=" * 80)
    print(raw_output)
    print("=" * 80 + "\n")

    if not raw_output.strip():
        raise RuntimeError("OpenAI returned empty output.")

    json_text = extract_json_text(raw_output)
    parsed = json.loads(json_text)
    return validate_and_normalize_model_output(parsed)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    test_file = pick_test_file(script_dir)

    suffix = test_file.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type: {}".format(suffix))

    file_size = test_file.stat().st_size
    if file_size == 0:
        raise ValueError("{} is empty".format(test_file.name))
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError("{} is too large ({} bytes)".format(test_file.name, file_size))

    logger.info("Using file: %s", test_file.name)
    logger.info("File size: %s bytes", file_size)

    file_id = upload_file_to_openai(test_file)
    result = ask_openai_to_parse_file(file_id)

    print("\n" + "=" * 80)
    print("VALIDATED JSON RESULT")
    print("=" * 80)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Failed: %s", e)
        logger.error(traceback.format_exc())
        sys.exit(1)