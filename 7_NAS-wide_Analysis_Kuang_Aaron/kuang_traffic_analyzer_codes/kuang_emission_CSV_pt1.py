# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 16:07:36 2026

@author: anomi
"""

import zipfile
import gzip
from pathlib import Path
import pandas as pd

# ----------------------------
# Inputs
# ----------------------------
# change to your local path if needed
zip_path = r"C:\Users\anomi\Downloads\IFF_RD_2023_12_22_2023_12_23.zip"
# where extracted .gz files will go
work_dir = Path(
    r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\CSV\SHERLOCK_data\Kuang_emission")
out_path = Path(
    r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\CSV\SHERLOCK_data\Kuang_emission\tpa_filtered.csv")
# ----------------------------
# Helpers
# ----------------------------


def extract_zip(zip_path: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            # Extract each file into out_dir
            zf.extract(name, out_dir)
            paths.append(out_dir / name)
    return paths


def read_iff_record3_from_gz(gz_path: Path, skip_first_lines: int = 4) -> pd.DataFrame:
    """
    IFF file: mixed record types. Keep only record type '3'.
    Ignores first `skip_first_lines` lines (metadata).
    Returns a DataFrame of strings (variable columns prevented by filtering).
    """
    rows = []
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i < skip_first_lines:
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if parts and parts[0] == "3":
                rows.append(parts)
    return pd.DataFrame(rows)


def read_rd_from_gz(gz_path: Path) -> pd.DataFrame:
    """RD is a normal CSV with headers; pandas can read gzip directly."""
    return pd.read_csv(gz_path, compression="gzip")


# ----------------------------
# 1) Extract and discover files
# ----------------------------
extracted = extract_zip(zip_path, work_dir)

iff_files = sorted([p for p in extracted if p.name.startswith(
    "IFF_") and p.suffixes[-2:] == [".csv", ".gz"]])
rd_files = sorted([p for p in extracted if p.name.startswith(
    "RD_") and p.suffixes[-2:] == [".csv", ".gz"]])

if not iff_files:
    raise FileNotFoundError("No IFF_*.csv.gz files found in the ZIP.")
if not rd_files:
    raise FileNotFoundError("No RD_*.csv.gz files found in the ZIP.")

print("IFF files:", [p.name for p in iff_files])
print("RD files:", [p.name for p in rd_files])

# ----------------------------
# 2) Read + concat time windows
# ----------------------------
df1 = pd.concat([read_iff_record3_from_gz(p, skip_first_lines=4)
                 for p in iff_files], ignore_index=True)
df2 = pd.concat([read_rd_from_gz(p) for p in rd_files], ignore_index=True)

print("Combined IFF record-3 shape:", df1.shape)
print("Combined RD shape:", df2.shape)

# ----------------------------
# 3) Select required IFF columns: 2,3,4,10,11,12,17 (1-based)
# ----------------------------
cols_0based = [1, 2, 3, 9, 10, 11, 16]
if df1.shape[1] < max(cols_0based) + 1:
    raise ValueError(
        f"IFF record-3 has {df1.shape[1]} columns; need at least 17.")

df1_sub = df1.iloc[:, cols_0based].copy()

# Key is original 3rd column -> within df1_sub it's the 2nd selected col
key_col = df1_sub.columns[1]
df1_sub[key_col] = df1_sub[key_col].astype("string").str.strip()

# ----------------------------
# 4) Prepare RD fields (use headers explicitly)
# ----------------------------
required_rd = ["Msn", "AcType", "EstDest", "Airline"]
missing = [c for c in required_rd if c not in df2.columns]
if missing:
    raise ValueError(f"RD missing expected columns: {missing}")

df2_small = df2[required_rd].copy()
df2_small["Msn"] = df2_small["Msn"].astype("string").str.strip()

# ----------------------------
# 5) Merge (safe even if Msn duplicates)
# ----------------------------
merged = df1_sub.merge(df2_small, left_on=key_col,
                       right_on="Msn", how="left").drop(columns=["Msn"])

# ----------------------------
# 6) Filter: allowed AcType + EstDest == "DTW"
# ----------------------------
allowed = {
    "A19N", "A20N", "A21N", "A300", "A306", "A30B", "A310", "A318", "A319", "A320", "A321", "A332", "A333", "A338",
    "A339", "A342", "A343", "A345", "A346", "A359", "A35K", "A388", "BCS1", "BCS3", "B703", "B712", "B720", "B721",
    "B722", "B732", "B733", "B734", "B735", "B736", "B737", "B738", "B739", "B37M", "B38M", "B39M", "B3XM", "B741",
    "B742", "B743", "B744", "B748", "B74R", "B74S", "B752", "B753", "B762", "B763", "B764", "B772", "B773", "B778",
    "B779", "B77L", "B77W", "B788", "B789", "B78X", "DC8", "DC9", "DC10", "MD80", "MD87", "MD90", "MD11", "CRJ1",
    "CRJ2", "CRJ7", "CRJ9", "CRJX", "DH8A", "DH8B", "DH8C", "DH8D", "E135", "E140", "E145", "E170", "E75S", "E75L",
    "E190", "E195", "E290", "E295", "AT43", "AT45", "AT46", "AT72", "AT75", "AT76", "JS31", "JS32", "JS41", "SF34",
    "SW4", "D328", "J328", "F50", "F70", "F100", "L101"
}

ac = merged["AcType"].astype("string").str.strip()
dest = merged["EstDest"].astype("string").str.strip()

mask = (
    ac.notna() & (ac != "") & (ac != "?") & ac.isin(allowed) &
    (dest == "TPA")
)

out = merged.loc[mask].copy()

print("Rows after filtering:", len(out))

# ----------------------------
# 7) Save
# ----------------------------
out.to_csv(out_path, index=False)
print("Saved:", out_path.resolve())
