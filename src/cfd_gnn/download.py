from __future__ import annotations

import argparse
import io
import json
import ssl
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import certifi

DATASET = "waitongchung/inert-ch4o2-hit-dns"
API_ROOT = "https://www.kaggle.com/api/v1"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def _dataset_files() -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    token: str | None = None
    while True:
        query = urlencode({"pageToken": token}) if token else ""
        url = f"{API_ROOT}/datasets/list/{DATASET}"
        if query:
            url = f"{url}?{query}"
        with urlopen(url, timeout=60, context=TLS_CONTEXT) as response:
            page = json.load(response)
        files.extend(page["datasetFiles"])
        token = page.get("nextPageToken")
        if not token:
            return files


def _safe_destination(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe dataset path: {name}")
    destination = root.joinpath(*relative.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes destination: {name}")
    return destination


def _extract_file(
    name: str, destination: Path, body: bytes, content_type: str
) -> bytes:
    if not body.startswith(b"PK"):
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type in {"application/json", "text/xml", "application/xml"}:
            return body
        raise ValueError(
            f"Unexpected response for {name}: {content_type}, "
            f"body starts with {body[:32]!r}"
        )

    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        members = archive.namelist()
        if len(members) != 1 or PurePosixPath(members[0]).name != destination.name:
            raise ValueError(f"Unexpected archive contents for {name}: {members}")
        return archive.read(members[0])


def _download_file(root: Path, item: dict[str, object]) -> tuple[str, bool]:
    name = str(item["name"])
    expected_size = int(item["totalBytes"])
    destination = _safe_destination(root, name)
    if destination.exists() and destination.stat().st_size == expected_size:
        return name, False

    url = f"{API_ROOT}/datasets/download/{DATASET}?{urlencode({'fileName': name})}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=180, context=TLS_CONTEXT) as response:
                body = response.read()
                content_type = response.headers.get("Content-Type", "")
            break
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)

    contents = _extract_file(name, destination, body, content_type)

    if len(contents) != expected_size:
        raise ValueError(
            f"Downloaded {name} has {len(contents)} bytes; expected {expected_size}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(contents)
    temporary.replace(destination)
    return name, True


def download_dataset(root: Path, workers: int = 4) -> None:
    if workers < 1:
        raise ValueError("--workers must be positive")
    root = root.resolve()
    files = _dataset_files()
    downloaded = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download_file, root, item) for item in files]
        for index, future in enumerate(as_completed(futures), start=1):
            name, changed = future.result()
            downloaded += int(changed)
            if index % 20 == 0 or index == len(futures):
                print(
                    f"{index}/{len(futures)} files checked; "
                    f"{downloaded} downloaded (latest: {name})",
                    flush=True,
                )
    print(f"Dataset ready at {root}; downloaded {downloaded} files.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and extract the public BLASTNet HIT Kaggle dataset"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = _parser().parse_args()
    download_dataset(args.data_dir, args.workers)


if __name__ == "__main__":
    main()
