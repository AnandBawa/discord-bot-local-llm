"""Pure PDF extraction entry point for the bot's single spawned worker."""

import pymupdf


def extract_pdf_text(pdf_bytes, max_pages=15):
    text = ""
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            for index, page in enumerate(document):
                if index >= max_pages:
                    text += "\n...[Additional pages skipped to save memory]"
                    break
                text += page.get_text() + "\n"
        return text.strip()
    except Exception as exc:
        return f"Error reading PDF: {exc}"
