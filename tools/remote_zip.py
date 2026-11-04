"""Lists or extracts individual files from a remote zip archive over HTTP range
requests, so a single scene can be pulled out of the multi-GB 3DGS reference
archives without downloading all of them.

Usage:
  python tools/remote_zip.py list    <url> [substring]
  python tools/remote_zip.py extract <url> <out_dir> <prefix> [<prefix> ...]
"""
import io
import os
import shutil
import sys
import time
import urllib.request
import zipfile

TIMEOUT_S = 60
MAX_ATTEMPTS = 8


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file backed by HTTP range requests."""

    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        # A stalled connection must not hang forever: time out and retry.
        for attempt in range(1, MAX_ATTEMPTS + 1):
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                    data = resp.read()
                if len(data) != n:
                    raise IOError(f"short read ({len(data)} of {n} bytes)")
                break
            except (OSError, IOError) as e:
                if attempt == MAX_ATTEMPTS:
                    raise
                print(f"  retry {attempt}/{MAX_ATTEMPTS - 1} at byte {self.pos}: {e}", flush=True)
                time.sleep(min(2 ** attempt, 30))
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("list", "extract"):
        print(__doc__)
        sys.exit(1)
    url = sys.argv[2]
    # A large read-ahead buffer keeps the number of HTTP requests low.
    archive = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=8 << 20))
    if sys.argv[1] == "list":
        needle = sys.argv[3] if len(sys.argv) > 3 else ""
        for info in archive.infolist():
            if needle in info.filename:
                print(f"{info.file_size:>14,}  {info.filename}")
        return

    out_dir, prefixes = sys.argv[3], sys.argv[4:]
    for info in archive.infolist():
        if info.is_dir() or not any(info.filename.startswith(p) for p in prefixes):
            continue
        target = os.path.join(out_dir, info.filename)
        if os.path.exists(target) and os.path.getsize(target) == info.file_size:
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"extracting {info.filename} ({info.file_size:,} bytes)", flush=True)
        with archive.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, 8 << 20)


if __name__ == "__main__":
    main()
