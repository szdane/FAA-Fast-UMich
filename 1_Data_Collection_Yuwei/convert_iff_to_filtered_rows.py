r"""Convert a single NASA Sherlock facility/day IFF file to the legacy 47-column CSV.

Run from the repository root with Python 3.9 or newer:
    python "1_Data_Collection_Yuwei\convert_iff_to_filtered_rows.py" INPUT.csv OUTPUT.csv
    python "1_Data_Collection_Yuwei\convert_iff_to_filtered_rows.py" INPUT.csv OUTPUT.csv --skip-missing-airports

Type-2 headers supply Orig/Dest, joined to type-3 points by fltKey, not callsign.
Flight-plan updates and geometry-estimated airports are not substituted for the
header's reported airports. Missing headers or airports stop conversion unless
--skip-missing-airports is explicitly selected; skipped flight keys are logged.

recTime is written without a timezone suffix for compatibility. America/Detroit
is the default: UTC-05:00 in winter, UTC-04:00 in summer. Use --timezone UTC for
UTC timestamps, or --timezone Etc/GMT+5 for fixed UTC-05:00 throughout the year.
Windows installations may need the tzdata package for IANA timezone definitions.

All other track fields, units, row order, and source duplicates are preserved.
In particular, alt remains in hundreds of feet. Each accepted source point is
written once; no joins multiply rows, no blanks are forward-filled, and no
filename-date filter is applied. Existing output files are never overwritten.
Two streaming passes keep memory proportional to flights, not track points.

Format: https://sherlock.opendata.arc.nasa.gov/api/docs/#/products/IFF
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


TRACK_COLUMNS = [
    "recType", "recTime", "fltKey", "bcnCode", "cid", "Source", "msgType", "acId",
    "recTypeCat", "coord1", "coord2", "alt", "significance", "coord1Accur",
    "coord2Accur", "altAccur", "groundSpeed", "course", "rateOfClimb",
    "altQualifier", "altIndicator", "trackPtStatus", "leaderDir", "scratchPad",
    "msawInhibitInd", "assignedAltString", "controllingFac", "controllingSec",
    "receivingFac", "receivingSec", "activeContr", "primaryContr", "kybrdSubset",
    "kybrdSymbol", "adsCode", "opsType", "airportCode", "trackNumber",
    "tptReturnType", "modeSCode", "sensorTrackNumberList", "spi", "dvs", "dupM3a",
    "tid",
]
OUTPUT_COLUMNS = TRACK_COLUMNS + ["dep", "arrival"]
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
LOGGER = logging.getLogger(__name__)


def read_records(path: Path) -> Iterator[tuple[int, list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, [])[:2] != ["1", "IFF"]:
            raise ValueError(f"{path} is not a raw IFF file (expected '1,IFF,...').")
        for row in reader:
            if not row or row[0] not in ("2", "3"):
                continue
            if (row[0] == "2" and len(row) < 12) or (
                row[0] == "3" and len(row) not in (36, 45)
            ):
                raise ValueError(
                    f"Line {reader.line_num}: unsupported type-{row[0]} "
                    f"record with {len(row)} fields."
                )
            if not row[2].strip() or not row[7].strip():
                raise ValueError(f"Line {reader.line_num}: missing fltKey or acId.")
            yield reader.line_num, row


def format_time(value: str, zone: ZoneInfo) -> str:
    microseconds = Decimal(value) * 1_000_000
    if not microseconds.is_finite() or microseconds != microseconds.to_integral_value():
        raise ValueError("Expected finite Unix seconds with at most microsecond precision.")
    instant = EPOCH + timedelta(microseconds=int(microseconds))
    precision = "milliseconds" if int(microseconds) % 1000 == 0 else "microseconds"
    return instant.astimezone(zone).replace(tzinfo=None).isoformat(
        sep=" ", timespec=precision
    )


def convert_iff(
    input_path: Path,
    output_path: Path,
    *,
    timezone_name: str = "America/Detroit",
    skip_missing_airports: bool = False,
) -> tuple[int, int]:
    """Return (written points, skipped points); reject ambiguous flight metadata."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    zone = ZoneInfo(timezone_name)
    headers: dict[str, tuple[str, str, str]] = {}
    callsigns: dict[str, str] = {}
    track_counts: Counter[str] = Counter()

    for line_number, row in read_records(input_path):
        key, acid = row[2], row[7]
        if row[0] == "2":
            metadata = (acid, row[10].strip(), row[11].strip())
            if key in headers and headers[key] != metadata:
                raise ValueError(f"Line {line_number}: conflicting headers for fltKey {key}.")
            headers[key] = metadata
        else:
            if key in callsigns and callsigns[key] != acid:
                raise ValueError(f"Line {line_number}: conflicting callsigns for fltKey {key}.")
            callsigns[key] = acid
            track_counts[key] += 1

    if not track_counts:
        raise ValueError("The IFF file contains no type-3 track points.")
    for key, acid in callsigns.items():
        if key in headers and headers[key][0] != acid:
            raise ValueError(f"Header/track callsign mismatch for fltKey {key}.")

    missing = {
        key for key in track_counts
        if key not in headers or any(value in ("", "?") for value in headers[key][1:])
    }
    skipped = sum(track_counts[key] for key in missing)
    if missing:
        message = (
            f"{len(missing):,} tracked flights ({skipped:,} points) lack reported "
            f"origin/destination metadata. Flight keys: {', '.join(sorted(missing))}."
        )
        if not skip_missing_airports:
            raise ValueError(message + " Supply the missing airports or use --skip-missing-airports.")
        if len(missing) == len(track_counts):
            raise ValueError(message + " No complete flights remain to export.")
        LOGGER.warning("%s These flights will be skipped; airports are not guessed.", message)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with TemporaryDirectory(prefix=".iff-conversion-", dir=output_path.parent) as directory:
        temporary_output = Path(directory) / "filtered_rows.csv"
        with temporary_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(OUTPUT_COLUMNS)
            for line_number, row in read_records(input_path):
                if row[0] != "3" or row[2] in missing:
                    continue
                _, departure, arrival = headers[row[2]]
                converted = row + [""] * (len(TRACK_COLUMNS) - len(row))
                try:
                    converted[1] = format_time(row[1], zone)
                except (InvalidOperation, ValueError, OverflowError) as error:
                    raise ValueError(
                        f"Line {line_number}: invalid recTime {row[1]!r}: {error}"
                    ) from error
                converted[7] = f"{row[7]}_{departure}to{arrival}"
                writer.writerow(converted + [departure, arrival])
                written += 1
        if output_path.exists():
            raise FileExistsError(f"Output already exists: {output_path}")
        temporary_output.rename(output_path)
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Raw IFF CSV containing type-2 and type-3 records.")
    parser.add_argument("output", type=Path, help="New output CSV; existing files are not overwritten.")
    parser.add_argument("--timezone", default="America/Detroit",
                        help="Output IANA timezone (default: America/Detroit, including DST).")
    parser.add_argument("--skip-missing-airports", action="store_true",
                        help="Explicitly exclude flights with unknown header origin/destination.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        written, skipped = convert_iff(
            args.input, args.output, timezone_name=args.timezone,
            skip_missing_airports=args.skip_missing_airports,
        )
    except ZoneInfoNotFoundError:
        parser.exit(1, f"Error: timezone {args.timezone!r} is unavailable. Check its name "
                    "and install tzdata if timezone definitions are missing.\n")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved {written:,} rows and {len(OUTPUT_COLUMNS)} columns to {args.output}")
    print(f"recTime timezone: {args.timezone} (suffix omitted); skipped points: {skipped:,}")


if __name__ == "__main__":
    main()
