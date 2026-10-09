"""Read members of a remote zip (e.g. a 12.6 GB Zenodo archive) with HTTP range requests,
so only the needed files are downloaded instead of the whole archive."""
import io
import urllib.request


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20):
        self.url, self.pos, self.block = url, 0, block
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "acousticwatch/1.0"})
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
        self._cache = (-1, b"")

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _get(self, start, end):
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}",
                                                                "User-Agent": "acousticwatch/1.0"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception:
                if attempt == 5:
                    raise
                import time; time.sleep(5 * (attempt + 1))

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        cs, cb = self._cache
        if cs <= self.pos and self.pos + n <= cs + len(cb):
            out = cb[self.pos - cs:self.pos - cs + n]
        else:
            fetch = max(n, self.block)
            data = self._get(self.pos, min(self.pos + fetch, self.size))
            self._cache = (self.pos, data)
            out = data[:n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)
