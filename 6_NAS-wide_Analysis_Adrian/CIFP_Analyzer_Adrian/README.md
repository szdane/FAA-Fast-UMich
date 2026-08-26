# CIFP Analyzer

`CIFP_analyzer.py` builds a waypoint database from an FAA CIFP fixed-width text file, extracts waypoint/fix records, optionally filters them to a pre-TRACON region, allows manually defined fixes to be added, and plots the resulting waypoint set.

The current script is configured for an MSP example, but the same workflow can be adapted for other airports by changing the airport center coordinates, boundary fixes, ordered boundary list, manual waypoints, and output paths.

---

## What the folder contains

Graph_waypoint_csv Folder: Contains CSV's of the waypoints located in pre-TRACON regions surrounding DTW, ORD, ATL and MSP.

Figures Folder: Contain's visual depiction of pre-TRACON region + waypoints

CIFP_analyzer.py: The main file used to manipulate and extract data from the FAA CIFP file


## What the script does

The script has three main stages:

1. **Parse FAA CIFP waypoint records**
   - Reads a fixed-width FAA CIFP file named `FAACIFP18`.
   - Extracts waypoint/fix records from the CIFP text.
   - Converts CIFP coordinate strings into decimal latitude/longitude.
   - Stores all extracted waypoints in a SQLite database.
   - Writes a deduplicated waypoint CSV.

2. **Extract pre-TRACON waypoints**
   - Builds a geodesic circle around an airport center point.
   - Builds a polygon from user-defined boundary fixes.
   - Subtracts the polygon from the circle to define a pre-TRACON region.
   - Extracts waypoints from the SQLite database that fall inside that region.
   - Optionally appends manually defined waypoints.
   - Writes the final pre-TRACON waypoint list to CSV.

3. **Plot the extracted region and waypoints**
   - Loads the pre-TRACON waypoint CSV.
   - Plots extracted waypoints.
   - Plots the airport center.
   - Plots the TRACON boundary.
   - Plots the 300 km geodesic circle.
   - Saves the figure as a PNG.

---

## Dependencies

The script uses the following Python packages:

```bash
pip install numpy pandas pyproj shapely matplotlib
```

It also uses Python standard-library modules:

```python
csv
re
sqlite3
sys
pathlib
typing
```

---

## Required input file

The script expects an FAA CIFP fixed-width text file (file location dependent on user environment)

This file is parsed line-by-line to identify waypoint records.

To use a different CIFP file, edit this line inside `main()`:

```python
input_path = Path(
    r"/Users/XXXX/Documents/LATTICE/FAST/Misc/FAACIFP18"
)
```

---

## Main outputs

The script writes several outputs.

### 1. SQLite waypoint database

```text
/Users/XXXX/Documents/LATTICE/FAST/CSV/Graph_waypoint/faacifp18.sqlite
```

This database contains a `waypoints` table with all extracted CIFP waypoint records.

### 2. All-waypoints CSV

```text
/Users/XXXX/Documents/LATTICE/FAST/CSV/Graph_waypoint/all_waypoints.csv
```

This CSV contains deduplicated waypoint coordinate entries.

### 3. Pre-TRACON waypoint CSV

For the current MSP setup:

```text
/Users/XXXX/Documents/LATTICE/FAST/CSV/Graph_waypoint/msp_pre_tracon_graph.csv
```

This CSV contains waypoints inside the generated pre-TRACON region, plus any manually added waypoints.

### 4. Pre-TRACON plot

For the current MSP setup:

```text
/Users/XXXX/Documents/LATTICE/FAST/Figures/Graph_waypoint/MSP_pre_tracon_waypoints.png
```

This plot shows the extracted waypoint set, TRACON boundary, 300 km circle, and airport center.

---

## Coordinate conventions

The script uses two coordinate conventions depending on context.

### User-defined fixes

Boundary fixes and manual waypoints should be defined as:

```python
"FIX": (lat, lon)
```

Example:

```python
msp_fixes = {
    "BAINY": (45.7536, -93.6994),
    "MUSCL": (45.0288, -91.7768),
}
```

### Shapely geometry

Shapely expects coordinates as:

```python
(lon, lat)
```

The script handles this conversion internally when building polygons and points.

---

## How CIFP coordinates are parsed

FAA CIFP coordinates are parsed from strings in this format:

```text
NDDMMSSssWDDDMMSSss
```

For example, a coordinate such as:

```text
N4212750W08321200
```

is converted into decimal degrees:

```python
(lat, lon)
```

The regular expression used for parsing is:

```python
COORD_RE = re.compile(
    r"([NS])(\d{2})(\d{2})(\d{4})([EW])(\d{3})(\d{2})(\d{4})"
)
```

---

## Waypoint records extracted

The function `classify_waypoint_record()` keeps two types of CIFP records:

### Enroute waypoints

```text
section E, subsection A
```

These are labeled as:

```text
ENROUTE_WAYPOINT
```

### Terminal waypoints/fixes

```text
section P, terminal waypoint subsection C
```

These are labeled as:

```text
TERMINAL_WAYPOINT
```

All other records are ignored.

---

## Database schema

The SQLite database contains a table named `waypoints`.

Columns:

| Column | Description |
|---|---|
| `id` | SQLite autoincrement ID |
| `line_number` | Original line number in the CIFP file |
| `record_type` | First five characters of the CIFP record |
| `area_code` | CIFP area code |
| `section_code` | CIFP section code |
| `subsection_code` | CIFP subsection code |
| `waypoint_class` | `ENROUTE_WAYPOINT` or `TERMINAL_WAYPOINT` |
| `airport_or_enroute` | Airport/enroute identifier field |
| `ident` | Waypoint/fix identifier |
| `lat` | Decimal latitude |
| `lon` | Decimal longitude |
| `coord_raw` | Raw coordinate substring from the CIFP record |
| `raw_record` | Full original CIFP line |

Indexes are created on:

```text
ident
lat, lon
waypoint_class
```

---

## Running the script

From a terminal:

```bash
python CIFP_analyzer.py
```

The script will first create the SQLite database and all-waypoints CSV. Then it will run the MSP pre-TRACON extraction and generate a plot.

When complete, it prints summary information such as:

```text
Created SQLite database: .../faacifp18.sqlite
Created waypoint CSV: .../all_waypoints.csv
Waypoint records extracted: ...
Unique waypoint coordinate entries: ...
Plot saved to:
.../MSP_pre_tracon_waypoints.png
```

---

## Configuring the airport example

The current airport center is MSP:

```python
msp_lat, msp_lon = 44.8820, -93.2217
```

To adapt the script to another airport, change:

```python
msp_lat, msp_lon
msp_fixes
msp_ordered_fixes
manual_msp_waypoints
output_csv
airport_id
plot title/labels if desired
```

---

## Boundary fixes

The pre-TRACON boundary is defined by a dictionary of fixes:

```python
msp_fixes = {
    "BAINY": (45.7536, -93.6994),
    "MUSCL": (45.0288, -91.7768),
    "KASPR": (43.9655, -93.2470),
    "TORGY": (44.6436, -94.3753),
}
```

The polygon is built using `msp_ordered_fixes`:

```python
msp_ordered_fixes = [
    "BAINY",
    "MUSCL",
    "KASPR",
    "TORGY",
]
```

The order matters. The fixes should be listed in the order they should be connected around the boundary.

---

## Manual waypoints

Manual waypoints can be appended to the output even if they are not present in the CIFP database or are outside the region.

Example:

```python
manual_msp_waypoints = {
    "GEP":   (45.145694, -93.373194),
    "KKILR": (44.852839, -92.185589),
    "BLUEM": (44.181350, -93.223797),
    "NITZR": (44.186244, -93.466072),
}
```

The option:

```python
include_manual_outside_region=True
```

means manual waypoints are always included, even if they fall outside the generated pre-TRACON area.

Set it to `False` to only keep manual waypoints that fall inside the region.

---

## Pre-TRACON region logic

The function `build_pre_tracon_area()` constructs the region as:

```text
300 km geodesic circle around airport - TRACON boundary polygon
```

The geodesic circle is created using WGS84 ellipsoid geometry through `pyproj.Geod`.

The default radius is:

```python
radius_m = 300_000
```

This corresponds to 300 km.

---

## CSV output fields

The all-waypoints CSV contains:

| Column | Description |
|---|---|
| `ident` | Waypoint/fix name |
| `lat` | Latitude in decimal degrees |
| `lon` | Longitude in decimal degrees |
| `waypoint_classes` | Distinct waypoint classes associated with that fix |
| `area_codes` | Distinct CIFP area codes associated with that fix |
| `record_count` | Number of source records grouped into this row |

The pre-TRACON CSV includes more detailed fields inherited from the database, including record type, section code, subsection code, raw coordinate string, raw CIFP record, and line number.

---

## Notes and cautions

- The script currently uses absolute file paths. Update these paths before running on another machine.
- The `main()` function deletes the existing SQLite database if it already exists:

```python
if db_path.exists():
    db_path.unlink()
```

- Boundary and manual waypoint dictionaries use `(lat, lon)`, but Shapely plotting and geometry calculations use `(lon, lat)`.
- The polygon boundary fix order is important. An incorrect order can create an invalid or self-crossing polygon.
- The plotting section is currently tied to the MSP variable names. Rename these variables consistently if adapting to another airport.

---

