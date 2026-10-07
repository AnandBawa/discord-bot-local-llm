"""Bounded PDF text extraction, run in a fresh child process for each upload."""

PDF_MEMORY_LIMIT = 512 * 1024 * 1024
PDF_CPU_SECONDS = 10
PDF_RESOURCE_ERROR = (
    "Error reading PDF: This PDF is too complex or took too long to process. Try a simpler file."
)


def extract_pdf_text(pdf_bytes, max_pages=15, max_chars=40000):
    import pymupdf

    # Parser diagnostics must not become part of the extracted document.
    pymupdf.TOOLS.mupdf_display_errors(False)
    pymupdf.TOOLS.mupdf_display_warnings(False)
    text = ""
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            for index, page in enumerate(document):
                if index >= max_pages:
                    text += "\n...[Additional pages skipped to save memory]"
                    break
                page_text = page.get_text() + "\n"
                remaining = max_chars - len(text)
                if len(page_text) > remaining:
                    return text + page_text[:remaining] + "\n...[Content Truncated]"
                text += page_text
                if len(text) == max_chars and index + 1 < len(document):
                    return text + "\n...[Content Truncated]"
        return text.strip()
    except Exception:
        return "Error reading PDF: The file is invalid or too complex to read. Try a simpler PDF."


def main():
    import sys

    try:
        import resource
    except ImportError:
        print("Error reading PDF: Safe PDF processing requires Linux/WSL.")
        return
    # Apply limits before importing the native parser or accepting document data.
    # Retain any stricter limits inherited from the host/test environment.
    for kind, maximum in ((resource.RLIMIT_AS, PDF_MEMORY_LIMIT),
                          (resource.RLIMIT_CPU, PDF_CPU_SECONDS), (resource.RLIMIT_CORE, 0)):
        inherited = resource.getrlimit(kind)
        cap = min([maximum, *(value for value in inherited if value != resource.RLIM_INFINITY)])
        resource.setrlimit(kind, (cap, cap))
    data = sys.stdin.buffer.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        text = "Error reading PDF: The file exceeds the 10 MiB limit."
    else:
        text = extract_pdf_text(data, int(sys.argv[1]), int(sys.argv[2]))
    sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    main()
