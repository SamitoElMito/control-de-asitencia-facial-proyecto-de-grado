import ctypes
import platform
from pathlib import Path


def short_path(path) -> str:
    """
    Convert a path to the Windows 8.3 short form before passing it to OpenCV.
    OpenCV's C++ layer (FileStorage, CascadeClassifier) cannot open paths that
    contain non-ASCII characters such as accented letters.
    On non-Windows systems the path is returned unchanged.
    """
    resolved = str(Path(path).resolve())
    if platform.system() != "Windows":
        return resolved
    buf = ctypes.create_unicode_buffer(512)
    ret = ctypes.windll.kernel32.GetShortPathNameW(resolved, buf, len(buf))
    if ret == 0 or not buf.value:
        return resolved
    return buf.value
