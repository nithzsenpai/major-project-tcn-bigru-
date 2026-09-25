from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .constants import ATTACK_TO_CLASS, SCENARIOS, ZENODO_RECORD_ID


PROFILE_SCENARIOS = {
    "quick": ("highway_2",),
    "recommended": ("highway_2", "urban_2"),
    "full": SCENARIOS,
}


def build_http_session() -> requests.Session:
    """Create a session that tolerates temporary Zenodo/API failures."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "vanet-sybil-illusion-detector/1.1.0 (dataset downloader)"}
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _wanted_names(scenarios: Iterable[str], include_ground_truth: bool) -> set[str]:
    wanted: set[str] = set()
    for scenario in scenarios:
        wanted.add(f"InTAS_{scenario}.zip")
        for attack in ATTACK_TO_CLASS:
            wanted.add(f"InTAS_{scenario}_{attack}.zip")
        if include_ground_truth:
            wanted.update(
                {
                    f"InTAS_{scenario}_Train_groundTruth.json",
                    f"InTAS_{scenario}_Validation_groundTruth.json",
                    f"InTAS_{scenario}_Test_ground_truth.json",
                }
            )
    return wanted


def fetch_record_manifest(
    record_id: str = ZENODO_RECORD_ID,
    session: requests.Session | None = None,
) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    client = session or build_http_session()
    try:
        response = client.get(url, timeout=(15, 120))
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            "Zenodo did not respond after automatic retries. This is normally a "
            "temporary server or rate-limit problem. Wait 10-20 minutes and rerun "
            "the same command; already verified files will be kept."
        ) from error
    return response.json()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    file_info: dict,
    output_dir: Path,
    force: bool = False,
    session: requests.Session | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / file_info["key"]
    checksum = str(file_info.get("checksum", ""))
    expected_md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None

    if destination.exists() and not force:
        if destination.stat().st_size == int(file_info["size"]):
            if expected_md5 is None or _md5(destination) == expected_md5:
                print(f"Already verified: {destination.name}")
                return destination
        print(f"Existing file is incomplete or invalid; downloading again: {destination.name}")

    url = file_info.get("links", {}).get("content") or file_info.get("links", {}).get("self")
    if not url:
        raise ValueError(f"No download URL for {file_info['key']}")

    temporary = destination.with_suffix(destination.suffix + ".part")
    client = session or build_http_session()
    try:
        response = client.get(url, stream=True, timeout=(15, 300))
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Zenodo could not download {destination.name} after automatic retries. "
            "Rerun the same command later; completed files will not be downloaded again."
        ) from error
    with response:
        total = int(response.headers.get("content-length") or file_info["size"])
        with temporary.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))

    temporary.replace(destination)
    if destination.stat().st_size != int(file_info["size"]):
        raise RuntimeError(f"Size check failed for {destination.name}")
    if expected_md5 and _md5(destination) != expected_md5:
        raise RuntimeError(f"Checksum check failed for {destination.name}")
    return destination


def download_profile(
    profile: str,
    output_dir: Path,
    include_ground_truth: bool = False,
    force: bool = False,
) -> list[Path]:
    if profile not in PROFILE_SCENARIOS:
        raise ValueError(f"Unknown profile {profile!r}; choose {sorted(PROFILE_SCENARIOS)}")
    session = build_http_session()
    print("Contacting Zenodo (temporary 429/5xx errors are retried automatically)...")
    record = fetch_record_manifest(session=session)
    wanted = _wanted_names(PROFILE_SCENARIOS[profile], include_ground_truth)
    available = {item["key"]: item for item in record["files"]}
    missing = sorted(wanted - available.keys())
    if missing:
        raise RuntimeError(f"Files missing from Zenodo record: {missing}")

    selected = [available[name] for name in sorted(wanted)]
    total_gib = sum(int(item["size"]) for item in selected) / (1024**3)
    print(f"Profile {profile}: {len(selected)} files, approximately {total_gib:.2f} GiB")
    paths = [
        download_file(item, output_dir, force=force, session=session)
        for item in selected
    ]

    manifest = {
        "record_id": ZENODO_RECORD_ID,
        "profile": profile,
        "scenarios": list(PROFILE_SCENARIOS[profile]),
        "files": [
            {"name": item["key"], "size": item["size"], "checksum": item.get("checksum")}
            for item in selected
        ],
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download verified VeReMi NextGen subsets")
    parser.add_argument("--profile", choices=sorted(PROFILE_SCENARIOS), default="recommended")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--include-ground-truth", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    download_profile(args.profile, args.output, args.include_ground_truth, args.force)


if __name__ == "__main__":
    main()
