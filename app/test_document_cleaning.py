
from app.document_loader import fix_split_words, clean_pdf_text


def test_fix_split_words_repairs_line_breaks():
    text = "transmis-\nsion and packe\nt switching"

    result = fix_split_words(text)

    assert "transmission" in result.lower()
    assert "packet switching" in result.lower()


def test_fix_split_words_repairs_common_split_words():
    text = "communica\ntion and destina\ntion"

    result = fix_split_words(text)

    assert "communication" in result.lower()
    assert "destination" in result.lower()


def test_clean_pdf_text_removes_null_characters():
    text = "Circuit\x00 switching"

    result = clean_pdf_text(text)

    assert "\x00" not in result
    assert "Circuit switching" in result


def test_clean_pdf_text_preserves_headings():
    text = "Switching:\nIntroduction to switching"

    result = clean_pdf_text(text)

    assert "Switching:" in result
    assert "Introduction to switching" in result


def test_clean_pdf_text_preserves_bullets():
    text = "• Circuit switching establishes a dedicated path."

    result = clean_pdf_text(text)

    assert "Circuit switching establishes a dedicated path." in result
    assert "•" in result


def test_clean_pdf_text_joins_wrapped_paragraphs():
    text = (
        "Circuit switching establishes a dedicated\n"
        "path before data transmission begins."
    )

    result = clean_pdf_text(text)

    assert (
        "Circuit switching establishes a dedicated path "
        "before data transmission begins."
    ) in result


def test_clean_pdf_text_handles_empty_input():
    assert clean_pdf_text("") == ""
    assert fix_split_words("") == ""
