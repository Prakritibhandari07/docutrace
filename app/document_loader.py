from pathlib import Path
import re
from pypdf import PdfReader


def fix_split_words(text: str) -> str:
    """Repair common words split by PDF text extraction."""

    if not text:
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join words split by a hyphen at the end of a line.
    text = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*", "", text)

    # Repair common words split across lines.
    split_word_fixes = [
        (r"\bt\s*\n\s*ransmission\b", "transmission"),
        (r"\bt\s*\n\s*ypes\b", "types"),
        (r"\bCircui\s*\n\s*t\b", "Circuit"),
        (r"\bmess\s*\n\s*age\b", "message"),
        (r"\bpacke\s*\n\s*t\b", "packet"),
        (r"\bdestina\s*\n\s*tion\b", "destination"),
        (r"\bcommunica\s*\n\s*tion\b", "communication"),
        (r"\bconnec\s*\n\s*tion\b", "connection"),
        (r"\binforma\s*\n\s*tion\b", "information"),
        (r"\bapplica\s*\n\s*tion\b", "application"),
        (r"\bdedica\s*\n\s*ted\b", "dedicated"),
        (r"\bphysi\s*\n\s*cal\b", "physical"),
        (r"\btechni\s*\n\s*que\b", "technique"),
        (r"\bresour\s*\n\s*ces\b", "resources"),
        (r"\bnet\s*\n\s*work\b", "network"),
        (r"\bsw\s*\n\s*itching\b", "switching"),
        (r"\bestabl\s*\n\s*ished\b", "established"),
        (r"\btermina\s*\n\s*ted\b", "terminated"),
        (r"\btermina\s*\n\s*tes\b", "terminates"),
        (r"\bcompar\s*\n\s*able\b", "comparable"),
        (r"\binsp\s*\n\s*ects\b", "inspects"),
        (r"\bsigna\s*\n\s*l\b", "signal"),
        (r"\bba\s*\n\s*nd\b", "band"),
        (r"\bf\s*\n\s*our\b", "four"),
        (r"\bI\s*\n\s*llustrating\b", "Illustrating"),
        (r"\bTh\s*\n\s*e issue\b", "The issue"),
        (r"\bco\s*\n\s*mmunication\b", "communication"),
        (r"\bensur\s*\n\s*es\b", "ensures"),
        (r"\bunlimi\s*\n\s*ted\b", "unlimited"),
        (r"\bar\s*\n\s*rives\b", "arrives"),
        (r"\bVirtual C\s*\n\s*ircuit\b", "Virtual Circuit"),
        (r"\bsimultaneous\s*\n\s*ly\b", "simultaneously"),
    ]

    for pattern, replacement in split_word_fixes:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    # Repair common words split by spaces on the same line.
    same_line_fixes = [
        (r"\bt\s+r ansmission\b", "transmission"),
        (r"\bt\s+ypes\b", "types"),
        (r"\bCircui\s+t\b", "Circuit"),
        (r"\bmess\s+age\b", "message"),
        (r"\bpacke\s+t\b", "packet"),
        (r"\bdestina\s+tion\b", "destination"),
        (r"\bcommunica\s+tion\b", "communication"),
        (r"\bconnec\s+tion\b", "connection"),
        (r"\binforma\s+tion\b", "information"),
        (r"\bapplica\s+tion\b", "application"),
        (r"\bdedica\s+ted\b", "dedicated"),
        (r"\bphysi\s+cal\b", "physical"),
        (r"\btechni\s+que\b", "technique"),
        (r"\bresour\s+ces\b", "resources"),
        (r"\bnet\s+work\b", "network"),
        (r"\bsw\s+itching\b", "switching"),
        (r"\bVirtual C\s+ircuit\b", "Virtual Circuit"),
        (r"\bI\s+llustrating\b", "Illustrating"),
        (r"\bTh\s+e issue\b", "The issue"),
        (r"\bi\s+t\s+is\b", "it is"),
        (r"\bi\s+t\b", "it"),
    ]

    for pattern, replacement in same_line_fixes:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    # Repair specific line breaks around hyphenated expressions.
    text = re.sub(
        r"\bconnection\s*-\s*less\b",
        "connectionless",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bconnection\s*-\s*oriented\b",
        "connection-oriented",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\breceived\s+in\s*-\s*order\b",
        "received in order",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bcost\s*-\s*efficient\b",
        "cost-efficient",
        text,
        flags=re.IGNORECASE,
    )

    return text


def is_heading(line: str) -> bool:
    """Identify headings, document headers, and figure captions."""

    headings = {
        "Switching:",
        "Methods of Switching:",
        "Circuit Switched Network:",
        "Message Switched Network:",
        "Packet Switched Network:",
        "Packet Switching Techniques:",
        "Datagram Approach:",
        "Virtual Circuit Approach:",
        "Advantages",
        "Advantages:",
        "Disadvantages",
        "Disadvantages:",
    }

    if line in headings:
        return True

    if re.match(r"^(Figure|Fig)\s*:", line, re.IGNORECASE):
        return True

    if line.upper() == "DATA COMMUNICATION AND COMPUTER NETWORK":
        return True

    if line.upper() == "BIM II SEMESTER":
        return True

    if re.match(r"^Compiled By\s*:", line, re.IGNORECASE):
        return True

    return False


def clean_pdf_text(text: str) -> str:
    """Clean PDF text while preserving headings, bullets, and list items."""

    if not text:
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize common PDF bullet characters.
    bullet_characters = (
        "",
        "",
        "▪",
        "●",
        "◦",
        "",
        "\uf0d8",
    )

    for bullet in bullet_characters:
        text = text.replace(bullet, "•")

    text = fix_split_words(text)

    raw_lines = text.splitlines()
    lines = []

    for raw_line in raw_lines:
        line = raw_line.replace("\t", " ")
        line = re.sub(r" {2,}", " ", line).strip()

        # Remove whitespace before punctuation.
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)

        if line:
            lines.append(line)

    output = []
    current = ""

    def flush_current():
        nonlocal current

        if current.strip():
            output.append(current.strip())

        current = ""

    i = 0

    while i < len(lines):
        line = lines[i]

        # Join a standalone main bullet to the next line.
        if line == "•":
            flush_current()

            if i + 1 < len(lines):
                output.append("• " + lines[i + 1].strip())
                i += 2
                continue

            output.append("•")
            i += 1
            continue

        # Join a standalone sub-list marker to its description.
        if line.lower() == "o":
            flush_current()

            if i + 1 < len(lines):
                output.append("o " + lines[i + 1].strip())
                i += 2
                continue

            output.append("o")
            i += 1
            continue

        # Join a standalone number marker to the next line.
        if re.fullmatch(r"\d+\.", line):
            flush_current()

            if i + 1 < len(lines):
                output.append(line + " " + lines[i + 1].strip())
                i += 2
                continue

            output.append(line)
            i += 1
            continue

        # Preserve headings and figure captions.
        if is_heading(line):
            flush_current()
            output.append(line)
            i += 1
            continue

        # Preserve complete main bullets as separate entries.
        if line.startswith("•"):
            flush_current()

            bullet_text = line[1:].strip()

            if bullet_text:
                output.append("• " + bullet_text)
            else:
                output.append("•")

            i += 1
            continue

        # Preserve numbered entries such as "1. Datagram Approach".
        if re.match(r"^\d+\.\s+\S", line):
            flush_current()
            output.append(line)
            i += 1
            continue

        # Join wrapped lines into a single paragraph.
        if current:
            current += " " + line
        else:
            current = line

        i += 1

    flush_current()

    cleaned_lines = []

    for line in output:
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def load_pdf(pdf_path: str) -> list[dict]:
    """Extract cleaned text from a PDF while preserving page numbers."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(extraction_mode="plain")
        except TypeError:
            text = page.extract_text()

        if not text or not text.strip():
            continue

        cleaned_text = clean_pdf_text(text)

        if cleaned_text:
            documents.append(
                {
                    "page": page_number,
                    "text": cleaned_text,
                }
            )

    return documents