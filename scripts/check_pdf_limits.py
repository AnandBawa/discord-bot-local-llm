"""PDF resource limits: mocked API failures everywhere, real Job Objects on Windows.

Run: python scripts/check_pdf_limits.py
Native resource limits are applied only to owned, short-lived child processes.
Tests use synthetic PDFs and never import bot.py or load its configuration.
"""

import contextlib
import ctypes
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pdf_worker


class PDFLimitChecks(unittest.TestCase):
    def kernel(self):
        kernel = Mock()
        kernel.CreateJobObjectW.return_value = 0x1234567887654321
        kernel.GetCurrentProcess.return_value = ctypes.c_void_p(-1).value
        kernel.SetInformationJobObject.return_value = 1
        kernel.AssignProcessToJobObject.return_value = 1
        kernel.CloseHandle.return_value = 1
        return kernel

    def test_windows_limits_use_pointer_safe_abi_and_close_without_terminating_worker(self):
        kernel = self.kernel()
        with patch.object(ctypes, "WinDLL", return_value=kernel, create=True) as dll:
            pdf_worker.limit_windows_process()
        dll.assert_called_once_with("kernel32", use_last_error=True)
        self.assertIs(kernel.CreateJobObjectW.restype, ctypes.c_void_p)
        self.assertIs(kernel.GetCurrentProcess.restype, ctypes.c_void_p)
        self.assertEqual(kernel.AssignProcessToJobObject.argtypes, [ctypes.c_void_p] * 2)
        handle, info_class, pointer, size = kernel.SetInformationJobObject.call_args.args
        limits = pointer._obj
        self.assertEqual(handle, kernel.CreateJobObjectW.return_value)
        self.assertEqual(info_class, 9)
        self.assertEqual(size, ctypes.sizeof(limits))
        self.assertEqual(size, 144 if ctypes.sizeof(ctypes.c_void_p) == 8 else 112)
        self.assertEqual(limits.BasicLimitInformation.LimitFlags, 0x102)
        self.assertEqual(limits.BasicLimitInformation.PerProcessUserTimeLimit,
                         pdf_worker.PDF_CPU_SECONDS * 10_000_000)
        self.assertEqual(limits.ProcessMemoryLimit, pdf_worker.PDF_MEMORY_LIMIT)
        kernel.AssignProcessToJobObject.assert_called_once_with(handle, kernel.GetCurrentProcess.return_value)
        kernel.CloseHandle.assert_called_once_with(handle)

    def test_each_windows_api_setup_failure_propagates_and_closes_created_handle(self):
        for failed in ("CreateJobObjectW", "SetInformationJobObject", "AssignProcessToJobObject"):
            with self.subTest(failed=failed):
                kernel = self.kernel()
                getattr(kernel, failed).return_value = 0
                with patch.object(ctypes, "WinDLL", return_value=kernel, create=True), \
                        patch.object(ctypes, "WinError", return_value=OSError("synthetic failure"), create=True):
                    with self.assertRaises(OSError):
                        pdf_worker.limit_windows_process()
                if failed == "CreateJobObjectW":
                    kernel.CloseHandle.assert_not_called()
                else:
                    kernel.CloseHandle.assert_called_once_with(kernel.CreateJobObjectW.return_value)
                if failed != "AssignProcessToJobObject":
                    kernel.AssignProcessToJobObject.assert_not_called()

    def test_windows_selects_job_limits_without_importing_resource(self):
        with patch.object(sys, "platform", "win32"), \
                patch.object(pdf_worker, "limit_windows_process") as apply:
            pdf_worker.limit_pdf_process()
        apply.assert_called_once_with()

    def test_limit_setup_failure_does_not_read_or_parse_document(self):
        for error in (OSError, ImportError, AttributeError, ValueError):
            with self.subTest(error=error.__name__), \
                    patch.object(pdf_worker, "limit_pdf_process", side_effect=error("synthetic")), \
                    patch.object(pdf_worker, "extract_pdf_text") as extract, \
                    patch.object(sys, "stdin") as incoming, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                pdf_worker.main()
                incoming.buffer.read.assert_not_called()
                extract.assert_not_called()
                self.assertIn("could not enable its resource limits", output.getvalue())


@unittest.skipUnless(sys.platform == "win32", "Requires the native Windows kernel")
class WindowsPDFProcessChecks(unittest.TestCase):
    def child(self, code, *, data=b"", timeout=8):
        # Never set limits in the test runner or bot process.
        bootstrap = (
            "import runpy, socket, sys\n"
            "def denied(*args, **kwargs):\n    raise AssertionError('External sockets disabled')\n"
            "socket.socket.connect = socket.socket.connect_ex = denied\n"
            f"worker = runpy.run_path({str(ROOT / 'pdf_worker.py')!r})\n"
            "namespace = worker['main'].__globals__\n" + code
        )
        return subprocess.run([sys.executable, "-I", "-c", bootstrap], input=data,
                              capture_output=True, timeout=timeout)

    def test_native_job_limits_block_large_allocation_after_handle_closes(self):
        result = self.child(
            "namespace['PDF_MEMORY_LIMIT'] = 96 * 1024 * 1024\n"
            "worker['limit_pdf_process']()\n"
            "try:\n    value = bytearray(192 * 1024 * 1024)\n"
            "except MemoryError:\n    print('allocation blocked')\n"
            "else:\n    raise AssertionError('Memory limit was not enforced')\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(result.stdout.strip(), b"allocation blocked")

    def test_native_cpu_limit_terminates_only_busy_child_and_next_child_works(self):
        result = self.child(
            "namespace['PDF_CPU_SECONDS'] = 1\n"
            "worker['limit_pdf_process']()\n"
            "print('limits ready', flush=True)\n"
            "while True:\n    pass\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"limits ready", result.stdout)
        recovered = self.child("worker['limit_pdf_process']()\nprint('recovered')")
        self.assertEqual(recovered.returncode, 0, recovered.stderr.decode(errors="replace"))
        self.assertEqual(recovered.stdout.strip(), b"recovered")

    def test_native_worker_flushes_large_unicode_output_before_exit(self):
        result = self.child(
            "sys.argv = ['pdf_worker.py', '15', '40000']\n"
            "namespace['extract_pdf_text'] = lambda *args: '\\u754c' * 40000\n"
            "worker['main']()\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(result.stdout.decode("utf-8"), "\u754c" * 40000)

    def test_native_worker_reads_pdf_with_limits_then_accepts_later_file(self):
        import pymupdf

        with pymupdf.open() as document:
            document.new_page().insert_text((72, 72), "Synthetic Windows PDF")
            data = document.tobytes()
        for contents, expected in ((b"invalid", "Error reading PDF"), (data, "Synthetic Windows PDF")):
            with self.subTest(expected=expected):
                result = self.child("sys.argv = ['pdf_worker.py', '15', '40000']\nworker['main']()",
                                    data=contents)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                self.assertIn(expected, result.stdout.decode("utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
