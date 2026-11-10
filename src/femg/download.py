"""Download one file into a folder. Used by scripts/pull_data.py."""

import hashlib
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RemoteFile:
    """A file to download.

    If `unzip` is not empty, the file is a zip archive: only the listed members (matched by file name) are kept,
    extracted directly into the folder, and the zip itself is deleted.
    """

    url: str
    name: str
    md5: str | None = None
    unzip: tuple[str, ...] = ()


def is_downloaded(file: RemoteFile, folder: Path) -> bool:
    if file.unzip:
        return all((folder / member).exists() for member in file.unzip)
    return (folder / file.name).exists()


def download_file(file: RemoteFile, folder: Path, retries: int = 3) -> None:
    """Download `file` into `folder` (does nothing if it is already there).

    The data goes to a `.part` file first and is renamed at the end, so an interrupted download never leaves
    a file that looks complete.
    """
    if is_downloaded(file, folder):
        return
    folder.mkdir(parents=True, exist_ok=True)
    part = folder / (file.name + ".part")

    for attempt in range(1, retries + 1):
        try:
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
    request = urllib.request.Request(url, headers={"User-Agent": "femg-dataset-builder"})
    with urllib.request.urlopen(request, timeout=60) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out, 8 << 20)
        expected = response.headers.get("Content-Length")
    if expected is not None and dest.stat().st_size != int(expected):
        raise OSError(f"{url}: truncated download ({dest.stat().st_size} of {expected} bytes)")


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_members(archive: Path, folder: Path, members: tuple[str, ...]) -> None:
    """Extract the wanted members flat into `folder` (archives nest them in folders that vary)."""
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
