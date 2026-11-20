import hashlib
import io
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RemoteFile:
    url: str
    name: str
    md5: str | None = None
    unzip: tuple[str, ...] = ()  # zip members to keep (flat in the folder); the zip is deleted
    member: str | None = None  # url is a zip: fetch only this member (HTTP range requests), saved as name


def is_downloaded(file: RemoteFile, folder: Path) -> bool:
    if file.unzip:
        return all((folder / member).exists() for member in file.unzip)
    return (folder / file.name).exists()


HEADERS = {"User-Agent": "femg-dataset-builder"}

# url kaggle://<owner>/<dataset>: the whole public dataset, extracted into the folder; name: its top entry
KAGGLE = "kaggle://"


def download_file(file: RemoteFile, folder: Path, retries: int = 3) -> None:
    if is_downloaded(file, folder):
        return
    folder.mkdir(parents=True, exist_ok=True)
    if file.url.startswith(KAGGLE):
        fetch_kaggle(file, folder)
        return
    part = folder / (file.name + ".part")  # renamed when complete: an interrupted download never looks done

    for attempt in range(1, retries + 1):
        try:
            if file.member:
                fetch_member(file.url, file.member, part)
            else:
                fetch(file.url, part)
            break
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)

    if file.md5 and md5_of(part) != file.md5:
        part.unlink()
        raise OSError(f"{file.url}: md5 mismatch")

    if file.unzip:
        extract_members(part, folder, file.unzip)
        part.unlink()
    else:
        part.rename(folder / file.name)


def fetch(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out, 8 << 20)
        expected = response.headers.get("Content-Length")
    if expected is not None and dest.stat().st_size != int(expected):
        raise OSError(f"{url}: truncated download ({dest.stat().st_size} of {expected} bytes)")


class HTTPRange(io.RawIOBase):
    """A remote file read with HTTP range requests: enough for zipfile to read one member of a large archive."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=HEADERS), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        headers = {**HEADERS, "Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=60) as r:
            if r.status != 206:
                raise OSError(f"{self.url}: the server ignores range requests")
            data = r.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_member(url: str, member: str, dest: Path) -> None:
    """One member of a remote zip, without downloading the archive."""
    with zipfile.ZipFile(io.BufferedReader(HTTPRange(url), buffer_size=8 << 20)) as z:
        with z.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out, 8 << 20)


def fetch_kaggle(file: RemoteFile, folder: Path) -> None:
    """A public Kaggle dataset through kagglehub (no credentials needed for public datasets)."""
    import kagglehub

    kagglehub.dataset_download(file.url.removeprefix(KAGGLE), output_dir=str(folder))
    if not (folder / file.name).exists():
        raise FileNotFoundError(f"{file.url}: downloaded, but {folder / file.name} is not in it")


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_members(archive: Path, folder: Path, members: tuple[str, ...]) -> None:
    missing = set(members)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            name = Path(entry.filename).name
            if name not in missing:
                continue
            tmp = folder / (name + ".part")
            with z.open(entry) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
            tmp.rename(folder / name)
            missing.discard(name)
    if missing:
        raise FileNotFoundError(f"{archive.name} lacks {sorted(missing)}")
