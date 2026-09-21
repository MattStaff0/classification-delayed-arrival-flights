"""
Fetch BTS On-Time Performance monthly data and name it for this project.

Downloads the prezipped monthly files from TranStats, extracts just the CSV
(not the bundled readme.html, which would clobber the project's copy), and
names it YYYY-MM-flight-data.csv so filename order is chronological.

Already-present months are skipped, so re-running only fetches what's missing.

Usage:
    python3 fetch_bts_data.py                          # 2025-07 .. 2026-06
    python3 fetch_bts_data.py --start 2025-07 --end 2026-06
    python3 fetch_bts_data.py --start 2026-06 --end 2026-06 --force
    python3 fetch_bts_data.py --dry-run
"""
import argparse
import csv
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile

BASE_URL = ("https://transtats.bts.gov/PREZIP/"
            "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")

DEFAULT_START = (2025, 7)
DEFAULT_END = (2026, 6)

USER_AGENT = "Mozilla/5.0"          # TranStats rejects some default agents
RETRIES = 3
RETRY_WAIT = 5                      # seconds, doubled each retry
EST_CSV_BYTES = 290_000_000         # a month of extracted CSV, for the disk check


def parse_month(text):
    """'2025-07' -> (2025, 7)."""
    try:
        year, month = text.split("-")
        year, month = int(year), int(month)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM, got {text!r}")
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError(f"month out of range in {text!r}")
    return year, month


def month_range(start, end):
    """Inclusive list of (year, month) from start to end."""
    if start > end:
        sys.exit(f"error: start {start} is after end {end}")
    months = []
    year, month = start
    while (year, month) <= end:
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def target_name(year, month):
    """ISO order, so sorting filenames sorts chronologically."""
    return f"{year}-{month:02d}-flight-data.csv"


def download(url, dest):
    """Download url to dest, retrying on network errors. Raises on failure."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                expected = response.headers.get("Content-Length")
                expected = int(expected) if expected else None
                with open(dest, "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            got = os.path.getsize(dest)
            if expected is not None and got != expected:
                raise OSError(f"truncated: got {got:,} bytes, expected {expected:,}")
            return got
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            if os.path.exists(dest):
                os.remove(dest)
            if attempt == RETRIES:
                raise
            wait = RETRY_WAIT * 2 ** (attempt - 1)
            print(f"    attempt {attempt} failed ({err}); retrying in {wait}s")
            time.sleep(wait)


def extract_csv(zip_path, year, month, dest):
    """Extract the single data CSV from the archive to dest. Raises on mismatch."""
    with zipfile.ZipFile(zip_path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected 1 CSV in {zip_path}, found {members}")
        partial = f"{dest}.part"
        with archive.open(members[0]) as source, open(partial, "wb") as handle:
            shutil.copyfileobj(source, handle, length=1024 * 1024)
    verify_month(partial, year, month)
    os.replace(partial, dest)       # atomic: a killed run never leaves a short CSV
    return os.path.getsize(dest)


def verify_month(path, year, month):
    """Confirm the first data row really is the year and month we asked for."""
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        row = next(reader, None)
    if not header or not row:
        raise ValueError(f"{path} has no data rows")
    fields = dict(zip(header, row))
    got = (fields.get("Year"), fields.get("Month"))
    if got != (str(year), str(month)):
        raise ValueError(f"{path} contains Year/Month {got}, expected {(year, month)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=parse_month, default=DEFAULT_START,
                        help="first month, YYYY-MM (default 2025-07)")
    parser.add_argument("--end", type=parse_month, default=DEFAULT_END,
                        help="last month, YYYY-MM (default 2026-06)")
    parser.add_argument("--out", default=".", help="directory for the CSVs")
    parser.add_argument("--keep-zip", action="store_true", help="keep the downloaded archives")
    parser.add_argument("--force", action="store_true", help="re-download months already present")
    parser.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    months = month_range(args.start, args.end)
    todo = [(y, m) for y, m in months
            if args.force or not os.path.exists(os.path.join(args.out, target_name(y, m)))]

    print(f"{len(months)} months requested, {len(todo)} to fetch "
          f"({len(months) - len(todo)} already present)")
    for year, month in todo:
        print(f"  will fetch {year}-{month:02d} -> {target_name(year, month)}")
    if not todo:
        return 0

    free = shutil.disk_usage(args.out).free
    needed = len(todo) * EST_CSV_BYTES
    print(f"Disk: {free / 1024**3:.1f} GB free, ~{needed / 1024**3:.1f} GB needed")
    if free < needed:
        sys.exit("error: not enough free disk space")
    if args.dry_run:
        return 0

    failures = []
    for index, (year, month) in enumerate(todo, start=1):
        name = target_name(year, month)
        dest = os.path.join(args.out, name)
        zip_path = os.path.join(args.out, f"bts_{year}_{month:02d}.zip")
        print(f"\n[{index}/{len(todo)}] {year}-{month:02d}")
        try:
            url = BASE_URL.format(year=year, month=month)
            start = time.perf_counter()
            size = download(url, zip_path)
            print(f"    downloaded {size / 1024**2:.0f} MB in {time.perf_counter() - start:.0f}s")
            size = extract_csv(zip_path, year, month, dest)
            print(f"    extracted  {size / 1024**2:.0f} MB -> {name}")
        except Exception as err:
            print(f"    FAILED: {err}")
            failures.append((year, month, err))
            if os.path.exists(f"{dest}.part"):
                os.remove(f"{dest}.part")
        finally:
            if os.path.exists(zip_path) and not args.keep_zip:
                os.remove(zip_path)

    print(f"\nDone: {len(todo) - len(failures)}/{len(todo)} fetched")
    for year, month, err in failures:
        print(f"  FAILED {year}-{month:02d}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
