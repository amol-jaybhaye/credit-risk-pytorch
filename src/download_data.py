# src/download_data.py
from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout, Timeout

UCI_URLS = [
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default+of+credit+card+clients.zip",
]

ZIP_NAME = "default_of_credit_card_clients.zip"


def _zip_sig_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 4:
        return False
    with path.open("rb") as f:
        return f.read(4)[:2] == b"PK"


def _sniff_first_bytes(url: str, timeout: int = 30) -> tuple[int, str, bytes]:
    """
    Fetch a tiny portion to detect HTML/error responses early.
    """
    headers = {"User-Agent": "credit-risk-pytorch/1.0", "Range": "bytes=0-1023"}
    r = requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True)
    ct = (r.headers.get("Content-Type") or "").lower()
    data = next(r.iter_content(chunk_size=1024), b"")
    status = r.status_code
    r.close()
    return status, ct, data


def _download_with_retries(url: str, dst: Path, *, timeout: int = 60, max_retries: int = 6) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")

    backoff = 1.0
    last_err: Optional[Exception] = None

    # Quick sniff: if server is giving HTML, fail fast
    status, ct, head = _sniff_first_bytes(url)
    if status >= 400:
        raise RuntimeError(f"HTTP {status} from {url}")
    if b"<html" in head.lower() or "text/html" in ct:
        # This is NOT the zip
        raise RuntimeError(f"Got HTML instead of ZIP from {url} (content-type={ct})")

    for attempt in range(1, max_retries + 1):
        try:
            resume_from = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "credit-risk-pytorch/1.0"}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"

            with requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"Bad status {r.status_code} from {url}")

                mode = "ab" if (resume_from > 0 and r.status_code == 206) else "wb"
                if mode == "wb" and part.exists():
                    part.unlink(missing_ok=True)

                with part.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            # Validate ZIP signature
            if not _zip_sig_ok(part):
                # show first bytes to make debugging obvious
                with part.open("rb") as f:
                    first = f.read(200)
                raise RuntimeError(
                    f"Downloaded file is not a ZIP (first bytes={first[:60]!r}). "
                    f"Likely HTML/error page or blocked download."
                )

            part.replace(dst)
            return dst

        except (ChunkedEncodingError, ConnectionError, ReadTimeout, Timeout, RuntimeError) as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 2, 20.0)

    raise RuntimeError(f"Failed downloading {url}. Last error: {last_err}")


def download_uci_zip(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / ZIP_NAME

    if zip_path.exists() and _zip_sig_ok(zip_path):
        return zip_path

    errors = []
    for url in UCI_URLS:
        try:
            return _download_with_retries(url, zip_path)
        except Exception as e:
            errors.append((url, str(e)))

    raise RuntimeError("All download URLs failed:\n" + "\n".join([f"- {u}: {err}" for u, err in errors]))


def extract_zip(zip_path: Path, data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # This will now only run if the signature is correct
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    # find .xls/.xlsx
    for p in data_dir.rglob("*.xls*"):
        return p

    raise FileNotFoundError(f"No .xls/.xlsx found after extracting {zip_path}")


def ensure_dataset(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # If user already manually put the excel file, use it.
    for p in data_dir.glob("*.xls*"):
        return p

    zip_path = download_uci_zip(data_dir)
    xls_path = extract_zip(zip_path, data_dir)

    # normalize name
    stable = data_dir / "default_of_credit_card_clients.xls"
    try:
        if xls_path != stable:
            os.replace(xls_path, stable)
            xls_path = stable
    except Exception:
        pass

    return xls_path
