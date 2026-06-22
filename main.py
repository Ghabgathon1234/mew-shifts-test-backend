import asyncio
import base64
import json
import logging
import os
import re
import threading
import traceback
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from typing import Any, Generator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mewshifts-import")

# Load .env for local development (Railway injects env vars directly).
load_dotenv()


# =============================================================================
# APP
# =============================================================================

app = FastAPI(title="MEW Shifts Import Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
IMPORT_SEMAPHORE = asyncio.Semaphore(2)

ALLOWED_EXTENSIONS = (".pdf", ".xls", ".xlsx")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")
INSERT_ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS + IMAGE_EXTENSIONS
ALLOWED_PERMISSION_TIMES = {"shift start", "shift end", "during shift"}
ALLOWED_STATUSES = {"onDuty", "Vacation", "Sick Leave", "Casual Leave", "Training Course", "Other"}
ALLOWED_SHIFTS = {"day", "night"}
MAX_OUTPUT_TOKENS = 32768
PDF_RENDER_DPI = 150

PAGE_CHUNK_SUFFIX = (
    "\n\nExtract EVERY attendance/leave row visible on THIS page/chunk only. "
    "Infer column meaning from this document's headers — layouts differ by file. "
    "For overnight shifts, put clock-out on the shift-start date. "
    "Do not skip rows."
)

SYSTEM_PROMPT_TEMPLATE = """
You extract attendance records from an uploaded PDF or Excel file.

Return STRICT JSON only in this exact schema:

{{
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "attend1": "HH:mm" | null,
      "attend2": "HH:mm" | null,
      "attend3": "HH:mm" | null,
      "leave1": "HH:mm" | null,
      "leave2": "HH:mm" | null,
      "status": "onDuty" | "Vacation" | "Sick Leave" | "Casual Leave" | "Training Course" | "Other" | null,
      "customized_vacation": "string" | null,
      "shift": "day" | "night" | null,
      "permission_time": "shift end" | "shift start" | "during shift" | null,
      "permission_period": "HH:mm" | null
    }}
  ]
}}

{presence_context}

CRITICAL — column mapping (layouts vary; infer from this document's headers):
- Documents may be tables, grids, lists, or system exports in any language.
- Do NOT assume fixed column names — read the actual headers and labels on THIS file.
- Date column examples: Date, التاريخ, Tarikh, Day, تاريخ, etc. → normalize to YYYY-MM-DD.
- First clock-in / entry / arrival / in / الدخول / sign-in column → attend1.
- Second clock-in (only when presence check enabled and two in-punches exist) → attend2.
- Clock-out / exit / departure / out / الخروج / sign-out column → leave1 (never attend2).
- leave2 only when the document shows two separate out-punches on the same day.
- Works for fingerprint reports, Excel/PDF exports, ZKTeco, HR portals, handwritten timesheets, etc.

Overnight / cross-day shifts:
- Some shifts start in the evening and end after midnight (e.g. attend at 22:00, leave at 06:00 next day).
- The document may show clock-in on one date and clock-out on the NEXT date's row.
- Assign leave1 to the day the shift STARTED (the attend1 day), not the next day.
- leave1 may be earlier than attend1 on the same record (e.g. attend1=18:51, leave1=07:05) — that is correct.
- Do NOT create a separate entry for the next day when it only has a clock-out from the previous night.
- A row with only clock-out and no clock-in is usually the leave punch of the previous shift.

Multi-page documents:
- Files may have many pages (e.g. "page 1 of 11", "عدد صفحات الاستعلام").
- Extract EVERY row from EVERY page through the full period — never stop after the first page or month.
- Do not summarize or sample; return every date row found in the entire document or page chunk.

{shift_pattern_context}

Status field:
- Use "onDuty" for normal work days (or null, same effect).
- Use "Vacation", "Sick Leave", "Casual Leave", or "Training Course" when the document indicates the employee was on leave or training that day. These days should have null times.
- Use "Other" with "customized_vacation" (short label, e.g. "Maternity Leave", "Hajj") when leave type does not match the standard types above.
- If a day exists in the pattern as an OFF day and has no data in the document, do NOT include it.

Shift field:
- "day" if shift starts roughly between 06:00–17:59, "night" if 18:00–05:59. Use null if unknown.

Permission fields:
- permission_time: "shift start", "shift end", or "during shift" — only if the document shows permission/excuse data.
- permission_period: Duration as "HH:mm".

Rules:
- Include only real dates found in the document.
- Normalize all dates to YYYY-MM-DD and all times to HH:mm (24-hour).
- If a value is missing, return null.
- Do not invent days or data.
- Do not output markdown — ONLY the JSON object.
- The document may be in any language including Arabic (العربية).
- Extract ALL available data: attendance times, leave types, permissions, vacations, sick leave, etc.
"""

INSERT_SYSTEM_PROMPT_TEMPLATE = """
You extract leave, permission, and attendance updates from an uploaded image, PDF, or Excel file.

The user wants to INSERT records such as sick leave, vacation, casual leave, training course, permissions, or attendance times into their shift calendar.

Return STRICT JSON only in this exact schema:

{{
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "attend1": "HH:mm" | null,
      "attend2": "HH:mm" | null,
      "attend3": "HH:mm" | null,
      "leave1": "HH:mm" | null,
      "leave2": "HH:mm" | null,
      "status": "onDuty" | "Vacation" | "Sick Leave" | "Casual Leave" | "Training Course" | "Other" | null,
      "customized_vacation": "string" | null,
      "shift": "day" | "night" | null,
      "permission_time": "shift end" | "shift start" | "during shift" | null,
      "permission_period": "HH:mm" | null
    }}
  ]
}}

{presence_context}

Focus on:
- Sick leave, vacation, casual leave, training course approvals or notices
- Permission / excuse requests with duration
- Attendance times if the document is a timesheet or clock record
- Photos of HR messages, WhatsApp approvals, paper forms, or screenshots

Status field:
- Use "Sick Leave", "Vacation", "Casual Leave", or "Training Course" when the document indicates leave/training for that date.
- Use "Other" with "customized_vacation" when the leave type is not one of the standard types.
- Leave days should usually have null times unless clock times are also shown.
- Use "onDuty" only when normal attendance times are provided without a leave status.

Permission fields:
- permission_time: "shift start", "shift end", or "during shift"
- permission_period: Duration as "HH:mm"

{shift_pattern_context}

Rules:
- Include only real dates found in the document.
- Normalize all dates to YYYY-MM-DD and all times to HH:mm (24-hour).
- If a value is missing, return null.
- Do not invent days or data.
- Do not output markdown — ONLY the JSON object.
- The document may be in any language including Arabic (العربية).
"""


# =============================================================================
# CONFIG (lazy OpenAI client — app must boot even if key is missing)
# =============================================================================


def _openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY")


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = _openai_api_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Import service is not configured (OPENAI_API_KEY missing).",
        )
    return OpenAI(
        api_key=api_key,
        timeout=180.0,
        max_retries=2,
    )


# =============================================================================
# HELPERS
# =============================================================================

TIME_RE = re.compile(r"^\d{2}:\d{2}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def safe_preview(text: str, max_len: int = 1000) -> str:
    if not text:
        return ""
    return text.replace("\x00", " ")[:max_len]


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


def normalize_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"null", "none", "nan", "-"}:
        return None
    for allowed in ALLOWED_STATUSES:
        if s.lower() == allowed.lower():
            return allowed
    return None


def normalize_shift(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s in {"null", "none", "nan", "-"}:
        return None
    return s if s in ALLOWED_SHIFTS else None


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

    raise ValueError(f"Invalid date format: {s}")


def extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_customized_vacation(value: Any, status: Optional[str]) -> Optional[str]:
    if status != "Other":
        return None
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"null", "none", "nan", "-"}:
        return None
    if len(s) > 40:
        s = s[:40]
    return s


def merge_day_dict(existing: dict, new: dict) -> dict:
    """Merge two normalized day dicts, preferring non-null fields."""
    merged = dict(existing)
    for key in (
        "attend1",
        "attend2",
        "attend3",
        "leave1",
        "leave2",
        "status",
        "customized_vacation",
        "shift",
        "permission_time",
        "permission_period",
    ):
        if merged.get(key) is None and new.get(key) is not None:
            merged[key] = new[key]
    return merged


def normalize_day_item(item: dict) -> Optional[dict]:
    if not isinstance(item, dict):
        return None

    try:
        date_value = normalize_date(item.get("date"))
    except ValueError as e:
        logger.warning("Skipping day item: %s", e)
        return None

    status = normalize_status(item.get("status"))
    return {
        "date": date_value,
        "attend1": normalize_time(item.get("attend1")),
        "attend2": normalize_time(item.get("attend2")),
        "attend3": normalize_time(item.get("attend3")),
        "leave1": normalize_time(item.get("leave1")),
        "leave2": normalize_time(item.get("leave2")),
        "status": status,
        "customized_vacation": normalize_customized_vacation(
            item.get("customized_vacation"), status
        ),
        "shift": normalize_shift(item.get("shift")),
        "permission_time": normalize_permission_time(item.get("permission_time")),
        "permission_period": normalize_duration(item.get("permission_period")),
    }


def validate_and_normalize_model_output(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object.")

    days = payload.get("days")
    if not isinstance(days, list):
        raise ValueError("Model output must contain a 'days' list.")

    by_date: dict[str, dict] = {}

    for index, item in enumerate(days):
        normalized = normalize_day_item(item)
        if normalized is None:
            logger.warning("Skipping days[%s]: invalid item", index)
            continue

        date_value = normalized["date"]
        if date_value in by_date:
            by_date[date_value] = merge_day_dict(by_date[date_value], normalized)
            logger.info("Merged duplicate date from model output: %s", date_value)
        else:
            by_date[date_value] = normalized

    normalized_days = sorted(by_date.values(), key=lambda x: x["date"])
    return {"days": normalized_days}


def upload_file_to_openai(client: OpenAI, filename: str, file_bytes: bytes) -> str:
    logger.info("Uploading file to OpenAI: %s", filename)

    uploaded = client.files.create(
        file=(filename, file_bytes),
        purpose="user_data",
    )

    logger.info("Uploaded successfully. file_id=%s", uploaded.id)
    return uploaded.id


def delete_openai_file(client: OpenAI, file_id: Optional[str]) -> None:
    if not file_id:
        return
    try:
        client.files.delete(file_id)
        logger.info("Deleted OpenAI file_id=%s", file_id)
    except Exception as e:
        logger.warning("Failed to delete OpenAI file_id=%s: %s", file_id, e)


def _format_single_pattern_lines(pattern: dict, title: str = "Pattern") -> list[str]:
    lines: list[str] = []
    name = pattern.get("name")
    eff_from = pattern.get("effective_from")
    eff_to = pattern.get("effective_to")
    if name or eff_from:
        span = f"{eff_from or '?'} to {eff_to or 'open'}"
        lines.append(f"- {title}: {name or 'Shift pattern'} ({span})")

    cycle_len = pattern.get("cycle_length", 0)
    duration = pattern.get("shift_duration_minutes", 0)
    if cycle_len:
        lines.append(
            f"  Cycle: {cycle_len} days, shift duration: {duration // 60}h{duration % 60:02d}m"
        )

    day_labels = []
    days = pattern.get("days", [])
    for d in days:
        day_num = d.get("day")
        is_off = d.get("off", False)
        start = d.get("start", "07:00")
        label = d.get("label", "")
        if is_off:
            day_labels.append(f"Day{day_num}=OFF")
        else:
            lbl = f"({label})" if label else ""
            day_labels.append(f"Day{day_num}=Work{lbl}@{start}")
    if day_labels:
        lines.append(f"  Layout: {', '.join(day_labels)}")

    anchor = pattern.get("anchor_date")
    anchor_cycle_day = pattern.get("anchor_cycle_day")
    if anchor:
        lines.append(f"  Anchor: {anchor} is cycle day {anchor_cycle_day or 1}")
    return lines


def build_pattern_context(shift_pattern_json: Optional[str] = None) -> str:
    if not shift_pattern_json:
        return ""

    try:
        data = json.loads(shift_pattern_json)
        lines = [
            "User shift patterns (use the pattern effective on each row's date for off days and overnight context):",
        ]

        if isinstance(data, dict) and isinstance(data.get("patterns"), list):
            for pattern in data["patterns"]:
                lines.extend(_format_single_pattern_lines(pattern))
        elif isinstance(data, dict):
            lines.extend(_format_single_pattern_lines(data, title="Pattern"))
        else:
            return ""

        lines.append(
            "- OFF cycle days should not have standalone work entries unless vacation/leave."
        )
        lines.append(
            "- Night shifts starting ~18:00-19:00 often end after midnight; merge exit onto the entry date."
        )
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Failed to parse shift_pattern_json: %s", e)
        return ""


def build_system_prompt(
    shift_pattern_json: Optional[str] = None,
    has_presence_check: bool = False,
) -> str:
    if has_presence_check:
        presence_context = (
            "PRESENCE CHECK IS ENABLED for this user.\n"
            "Each work day may have TWO clock-in punches and ONE clock-out:\n"
            "  attend1 = first clock-in (arrival)\n"
            "  attend2 = second clock-in (presence re-scan, mid-shift)\n"
            "  leave1  = clock-out (departure)\n"
            "Do NOT confuse the clock-out with attend2."
        )
    else:
        presence_context = (
            "Presence check is NOT enabled. Each work day has ONE clock-in and ONE clock-out:\n"
            "  attend1 = clock-in\n"
            "  leave1  = clock-out\n"
            "attend2/attend3 should be null unless the document clearly shows multiple in-punches."
        )

    pattern_context = build_pattern_context(shift_pattern_json)

    return SYSTEM_PROMPT_TEMPLATE.format(
        presence_context=presence_context,
        shift_pattern_context=pattern_context,
    )


def build_insert_prompt(
    shift_pattern_json: Optional[str] = None,
    has_presence_check: bool = False,
) -> str:
    if has_presence_check:
        presence_context = (
            "PRESENCE CHECK IS ENABLED for this user.\n"
            "If attendance times appear: attend1 = first clock-in, attend2 = presence re-scan, leave1 = clock-out."
        )
    else:
        presence_context = (
            "Presence check is NOT enabled. If attendance times appear: attend1 = clock-in, leave1 = clock-out."
        )

    pattern_context = build_pattern_context(shift_pattern_json)

    return INSERT_SYSTEM_PROMPT_TEMPLATE.format(
        presence_context=presence_context,
        shift_pattern_context=pattern_context,
    )


def merge_chunk_days(combined_by_date: dict[str, dict], chunk_result: dict) -> None:
    for day in chunk_result.get("days", []):
        date_value = day.get("date")
        if not date_value:
            continue
        if date_value in combined_by_date:
            combined_by_date[date_value] = merge_day_dict(
                combined_by_date[date_value], day
            )
        else:
            combined_by_date[date_value] = day


def count_pdf_pages(file_bytes: bytes) -> int:
    import fitz

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return doc.page_count


def extract_single_page_pdf(file_bytes: bytes, page_index: int) -> bytes:
    """Build a one-page PDF from a multi-page document (no text extraction)."""
    import fitz

    src = fitz.open(stream=file_bytes, filetype="pdf")
    if page_index < 0 or page_index >= src.page_count:
        raise IndexError(f"PDF page index out of range: {page_index}")

    dst = fitz.open()
    dst.insert_pdf(src, from_page=page_index, to_page=page_index)
    return dst.tobytes()


def render_pdf_page_png(file_bytes: bytes, page_index: int, dpi: int = PDF_RENDER_DPI) -> bytes:
    """Fallback when single-page PDF upload fails (scanned/low-quality pages)."""
    import fitz

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    if page_index < 0 or page_index >= doc.page_count:
        raise IndexError(f"PDF page index out of range: {page_index}")
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def ask_openai_to_parse_pdf_bytes(
    client: OpenAI,
    filename: str,
    pdf_bytes: bytes,
    system_prompt: str,
) -> dict:
    """Upload a (usually single-page) PDF to OpenAI and parse via vision/file input."""
    file_id: Optional[str] = None
    try:
        file_id = upload_file_to_openai(client, filename, pdf_bytes)
        return ask_openai_to_parse_file(client, file_id, system_prompt)
    finally:
        delete_openai_file(client, file_id)


def parse_pdf_document(
    client: OpenAI,
    file_bytes: bytes,
    system_prompt: str,
    filename: str,
) -> dict:
    """Non-streaming wrapper around [parse_pdf_document_stream]."""
    for event in parse_pdf_document_stream(
        client, file_bytes, system_prompt, filename
    ):
        if event.get("event") == "done":
            return {"days": event.get("days", [])}
    raise ValueError("PDF parse produced no result")


def parse_pdf_document_stream(
    client: OpenAI,
    file_bytes: bytes,
    system_prompt: str,
    filename: str,
) -> Generator[dict, None, None]:
    """
    Parse PDFs page-by-page: each page is sent as its own PDF to OpenAI
    (layout/vision) without local text extraction.
    """
    total_pages = count_pdf_pages(file_bytes)
    if total_pages == 0:
        raise ValueError("PDF has no pages")

    logger.info(
        "Parsing PDF (%s pages) — one OpenAI file per page (no text extraction)",
        total_pages,
    )
    yield {"event": "start", "total_pages": total_pages}

    combined_by_date: dict[str, dict] = {}
    pdf_pages_parsed = 0
    vision_pages_parsed = 0
    base_name = (filename or "document").rsplit(".", 1)[0]

    for page_index in range(total_pages):
        page_num = page_index + 1
        chunk_prompt = (
            system_prompt
            + PAGE_CHUNK_SUFFIX
            + f"\n(This chunk is page {page_num} of {total_pages}.)"
        )
        chunk_days = 0
        parsed = False

        try:
            page_pdf = extract_single_page_pdf(file_bytes, page_index)
            chunk_result = ask_openai_to_parse_pdf_bytes(
                client,
                f"{base_name}_page_{page_num}.pdf",
                page_pdf,
                chunk_prompt,
            )
            chunk_days = len(chunk_result.get("days", []))
            merge_chunk_days(combined_by_date, chunk_result)
            pdf_pages_parsed += 1
            parsed = True
            logger.info(
                "PDF page %s file: chunk_days=%s total_unique=%s",
                page_num,
                chunk_days,
                len(combined_by_date),
            )
        except Exception as e:
            logger.warning(
                "PDF page %s file parse failed, trying page image: %s", page_num, e
            )

        if not parsed:
            try:
                png_bytes = render_pdf_page_png(file_bytes, page_index)
                chunk_result = ask_openai_to_parse_image(
                    client,
                    f"{base_name}_page_{page_num}.png",
                    png_bytes,
                    chunk_prompt,
                )
                chunk_days = len(chunk_result.get("days", []))
                merge_chunk_days(combined_by_date, chunk_result)
                vision_pages_parsed += 1
                logger.info(
                    "PDF page %s image: chunk_days=%s total_unique=%s",
                    page_num,
                    chunk_days,
                    len(combined_by_date),
                )
            except Exception as e:
                logger.warning("PDF page %s image parse failed: %s", page_num, e)
                yield {
                    "event": "page",
                    "page": page_num,
                    "total_pages": total_pages,
                    "chunk_days": 0,
                    "total_days": len(combined_by_date),
                    "warning": str(e),
                }
                continue

        yield {
            "event": "page",
            "page": page_num,
            "total_pages": total_pages,
            "chunk_days": chunk_days,
            "total_days": len(combined_by_date),
        }

    if combined_by_date:
        normalized_days = sorted(combined_by_date.values(), key=lambda x: x["date"])
        logger.info(
            "PDF page parse complete: total_days=%s pdf_pages=%s image_pages=%s",
            len(normalized_days),
            pdf_pages_parsed,
            vision_pages_parsed,
        )
        yield {"event": "done", "days": normalized_days}
        return

    logger.warning("PDF per-page parse found no rows; falling back to whole-file upload")
    file_id = upload_file_to_openai(client, filename, file_bytes)
    try:
        result = ask_openai_to_parse_file(client, file_id, system_prompt)
        yield {
            "event": "done",
            "days": result.get("days", []),
        }
    finally:
        delete_openai_file(client, file_id)


def ask_openai_to_parse_file(client: OpenAI, file_id: str, system_prompt: str) -> dict:
    logger.info("Sending file reference to OpenAI for parsing...")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            max_output_tokens=MAX_OUTPUT_TOKENS,
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
                            "text": system_prompt,
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        logger.error("OpenAI request failed: %s", repr(e))
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {repr(e)}")

    raw_output = response.output_text or ""
    logger.info("OpenAI raw output chars: %s", len(raw_output))
    logger.info("OpenAI raw output preview: %s", safe_preview(raw_output, 2000))

    if not raw_output.strip():
        raise HTTPException(status_code=500, detail="OpenAI returned empty output.")

    json_text = extract_json_text(raw_output)

    try:
        parsed = json.loads(json_text)
    except Exception as e:
        logger.error("Model returned invalid JSON: %s", str(e))
        logger.error("Bad JSON preview: %s", safe_preview(json_text, 3000))
        raise HTTPException(status_code=500, detail=f"Model returned invalid JSON: {str(e)}")

    try:
        validated = validate_and_normalize_model_output(parsed)
        logger.info("Validated JSON successfully. days=%s", len(validated.get("days", [])))
        return validated
    except Exception as e:
        logger.error("Model JSON validation failed: %s", str(e))
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Model returned invalid attendance schema: {str(e)}",
        )


def is_image_filename(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def guess_image_mime(filename: str) -> str:
    lowered = filename.lower()
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith(".webp"):
        return "image/webp"
    if lowered.endswith((".heic", ".heif")):
        return "image/heic"
    return "image/jpeg"


def ask_openai_to_parse_image(
    client: OpenAI,
    filename: str,
    file_bytes: bytes,
    system_prompt: str,
) -> dict:
    logger.info("Sending image to OpenAI for parsing: %s", filename)

    mime = guess_image_mime(filename)
    b64 = base64.b64encode(file_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            max_output_tokens=MAX_OUTPUT_TOKENS,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": data_url,
                        },
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        logger.error("OpenAI image request failed: %s", repr(e))
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {repr(e)}")

    raw_output = response.output_text or ""
    logger.info("OpenAI image output chars: %s", len(raw_output))

    if not raw_output.strip():
        raise HTTPException(status_code=500, detail="OpenAI returned empty output.")

    json_text = extract_json_text(raw_output)

    try:
        parsed = json.loads(json_text)
    except Exception as e:
        logger.error("Model returned invalid JSON: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Model returned invalid JSON: {str(e)}")

    try:
        validated = validate_and_normalize_model_output(parsed)
        logger.info("Validated image JSON successfully. days=%s", len(validated.get("days", [])))
        return validated
    except Exception as e:
        logger.error("Model JSON validation failed: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Model returned invalid attendance schema: {str(e)}",
        )


def resolve_prompt_builder(filename: str, mode: str):
    """Bulk PDF/Excel use the full import extractor; insert prompt is for photos."""
    if mode == "insert" and is_image_filename(filename):
        return build_insert_prompt
    return build_system_prompt


def run_import_pipeline_stream(
    filename: str,
    file_bytes: bytes,
    shift_pattern_json: Optional[str] = None,
    has_presence_check: bool = False,
    mode: str = "import",
) -> Generator[dict, None, None]:
    """Yield NDJSON events while parsing (page progress for PDFs)."""
    client = get_openai_client()
    prompt_builder = resolve_prompt_builder(filename, mode)
    system_prompt = prompt_builder(shift_pattern_json, has_presence_check)

    if is_image_filename(filename):
        result = ask_openai_to_parse_image(client, filename, file_bytes, system_prompt)
        yield {"event": "done", "days": result.get("days", [])}
        return

    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        for event in parse_pdf_document_stream(
            client, file_bytes, system_prompt, filename
        ):
            yield event
        return

    file_id: Optional[str] = None
    try:
        file_id = upload_file_to_openai(client, filename, file_bytes)
        result = ask_openai_to_parse_file(client, file_id, system_prompt)
        yield {"event": "done", "days": result.get("days", [])}
    finally:
        delete_openai_file(client, file_id)


def run_import_pipeline(
    filename: str,
    file_bytes: bytes,
    shift_pattern_json: Optional[str] = None,
    has_presence_check: bool = False,
    mode: str = "import",
) -> dict:
    """Blocking OpenAI work — run inside asyncio.to_thread."""
    client = get_openai_client()
    prompt_builder = resolve_prompt_builder(filename, mode)
    system_prompt = prompt_builder(shift_pattern_json, has_presence_check)

    if is_image_filename(filename):
        return ask_openai_to_parse_image(client, filename, file_bytes, system_prompt)

    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        return parse_pdf_document(client, file_bytes, system_prompt, filename)

    file_id: Optional[str] = None
    try:
        file_id = upload_file_to_openai(client, filename, file_bytes)
        return ask_openai_to_parse_file(client, file_id, system_prompt)
    finally:
        delete_openai_file(client, file_id)


# =============================================================================
# ROUTES
# =============================================================================


@app.on_event("startup")
async def on_startup() -> None:
    port = os.getenv("PORT", "8000")
    key_present = bool(_openai_api_key())
    logger.info("MEW Shifts import backend starting on port=%s openai_key_present=%s", port, key_present)
    if not key_present:
        logger.error(
            "OPENAI_API_KEY is not set — /health will work but /import-attendance will return 503."
        )


@app.get("/")
def root() -> dict:
    return {"ok": True, "service": "mewshifts-import"}


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "mewshifts-import",
        "openai_key_present": bool(_openai_api_key()),
        "max_file_size_bytes": MAX_FILE_SIZE_BYTES,
        "max_parallel_imports": 2,
    }


@app.post("/import-attendance")
async def import_attendance(
    file: UploadFile = File(...),
    shift_pattern: Optional[str] = Form(None),
    has_presence_check: Optional[str] = Form(None),
    mode: Optional[str] = Form("import"),
    stream: Optional[str] = Form("true"),
) -> Any:
    logger.info("POST /import-attendance called")

    filename = (file.filename or "").strip()
    lowered = filename.lower()
    presence_check = has_presence_check in ("true", "1", "yes") if has_presence_check else False
    request_mode = (mode or "import").strip().lower()
    if request_mode not in {"import", "insert"}:
        request_mode = "import"
    use_stream = stream not in ("false", "0", "no")

    allowed_extensions = INSERT_ALLOWED_EXTENSIONS if request_mode == "insert" else ALLOWED_EXTENSIONS

    logger.info(
        "Incoming filename: %s  mode: %s  stream: %s  shift_pattern_present: %s  presence_check: %s",
        filename, request_mode, use_stream, shift_pattern is not None, presence_check,
    )

    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    if not lowered.endswith(allowed_extensions):
        if request_mode == "insert":
            raise HTTPException(
                status_code=400,
                detail="Only PDF, XLS, XLSX, JPG, JPEG, PNG, WEBP, or HEIC are supported.",
            )
        raise HTTPException(status_code=400, detail="Only PDF, XLS, XLSX are supported.")

    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.error("Failed reading uploaded file: %s", str(e))
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed reading uploaded file: {str(e)}")

    size = len(file_bytes)
    logger.info("Uploaded file size: %s bytes", size)

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    if size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    if not _openai_api_key():
        raise HTTPException(
            status_code=503,
            detail="Import service is not configured (OPENAI_API_KEY missing).",
        )

    logger.info("Waiting for semaphore slot...")
    async with IMPORT_SEMAPHORE:
        logger.info("Semaphore acquired for filename=%s", filename)

        if use_stream:
            loop = asyncio.get_running_loop()
            event_queue: asyncio.Queue = asyncio.Queue()

            def worker() -> None:
                try:
                    for event in run_import_pipeline_stream(
                        filename,
                        file_bytes,
                        shift_pattern,
                        presence_check,
                        request_mode,
                    ):
                        loop.call_soon_threadsafe(event_queue.put_nowait, event)
                except Exception as e:
                    logger.error("Stream import worker failed: %s", e)
                    loop.call_soon_threadsafe(
                        event_queue.put_nowait,
                        {"event": "error", "detail": str(e)},
                    )
                finally:
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)

            threading.Thread(target=worker, daemon=True).start()

            async def ndjson_generator():
                try:
                    while True:
                        event = await event_queue.get()
                        if event is None:
                            break
                        if event.get("event") == "done":
                            logger.info(
                                "Import stream complete for filename=%s | days_count=%s",
                                filename,
                                len(event.get("days", [])),
                            )
                        yield json.dumps(event, ensure_ascii=False) + "\n"
                finally:
                    logger.info("Import request finished for filename=%s", filename)

            return StreamingResponse(
                ndjson_generator(),
                media_type="application/x-ndjson",
            )

        try:
            parsed = await asyncio.to_thread(
                run_import_pipeline,
                filename,
                file_bytes,
                shift_pattern,
                presence_check,
                request_mode,
            )

            logger.info(
                "Import successful for filename=%s | days_count=%s",
                filename,
                len(parsed.get("days", [])),
            )

            return parsed

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Unhandled import failure: %s", str(e))
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            logger.info("Import request finished for filename=%s", filename)
