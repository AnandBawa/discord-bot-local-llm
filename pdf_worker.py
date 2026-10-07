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


def limit_windows_process():
    """Attach only this PDF worker to a Windows job before loading the parser."""
    import ctypes as c

    class BasicLimits(c.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", c.c_int64), ("PerJobUserTimeLimit", c.c_int64),
            ("LimitFlags", c.c_uint32), ("MinimumWorkingSetSize", c.c_size_t),
            ("MaximumWorkingSetSize", c.c_size_t), ("ActiveProcessLimit", c.c_uint32),
            ("Affinity", c.c_size_t), ("PriorityClass", c.c_uint32), ("SchedulingClass", c.c_uint32),
        ]

    class ExtendedLimits(c.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits), ("IoInfo", c.c_uint64 * 6),
            ("ProcessMemoryLimit", c.c_size_t), ("JobMemoryLimit", c.c_size_t),
            ("PeakProcessMemoryUsed", c.c_size_t), ("PeakJobMemoryUsed", c.c_size_t),
        ]

    kernel = c.WinDLL("kernel32", use_last_error=True)
    for name, args, result in (
        ("CreateJobObjectW", [c.c_void_p, c.c_wchar_p], c.c_void_p),
        ("SetInformationJobObject", [c.c_void_p, c.c_int, c.c_void_p, c.c_uint32], c.c_int32),
        ("GetCurrentProcess", [], c.c_void_p),
        ("AssignProcessToJobObject", [c.c_void_p, c.c_void_p], c.c_int32),
        ("CloseHandle", [c.c_void_p], c.c_int32),
    ):
        function = getattr(kernel, name)
        function.argtypes, function.restype = args, result

    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise c.WinError()
    try:
        limits = ExtendedLimits()
        # JOB_OBJECT_LIMIT_PROCESS_TIME | JOB_OBJECT_LIMIT_PROCESS_MEMORY.
        limits.BasicLimitInformation.LimitFlags = 0x00000002 | 0x00000100
        limits.BasicLimitInformation.PerProcessUserTimeLimit = PDF_CPU_SECONDS * 10_000_000
        limits.ProcessMemoryLimit = PDF_MEMORY_LIMIT
        # JobObjectExtendedLimitInformation = 9.
        if not kernel.SetInformationJobObject(job, 9, c.byref(limits), c.sizeof(limits)):
            raise c.WinError()
        if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
            raise c.WinError()
    finally:
        # With no KILL_ON_JOB_CLOSE flag, the limits survive handle closure until
        # the attached process exits. Closing must not kill the worker's output.
        kernel.CloseHandle(job)


def limit_pdf_process():
    import sys

    if sys.platform == "win32":
        limit_windows_process()
        return
    import resource

    # Apply limits before importing the native parser or accepting document data.
    # Retain any stricter limits inherited from the host/test environment.
    for kind, maximum in ((resource.RLIMIT_AS, PDF_MEMORY_LIMIT),
                          (resource.RLIMIT_CPU, PDF_CPU_SECONDS), (resource.RLIMIT_CORE, 0)):
        inherited = resource.getrlimit(kind)
        cap = min([maximum, *(value for value in inherited if value != resource.RLIM_INFINITY)])
        resource.setrlimit(kind, (cap, cap))


def main():
    import sys

    try:
        limit_pdf_process()
    except (ImportError, AttributeError, OSError, ValueError):
        print("Error reading PDF: The reader could not enable its resource limits. Check the bot's environment.")
        return
    data = sys.stdin.buffer.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        text = "Error reading PDF: The file exceeds the 10 MiB limit."
    else:
        text = extract_pdf_text(data, int(sys.argv[1]), int(sys.argv[2]))
    sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    main()
