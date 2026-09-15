# -*- coding: utf-8 -*-
"""Spyder-friendly launcher for the CWAM weather pipeline.

Open this file in Spyder and press Run (F5). Edit the options below,
or just edit config.yaml and leave these as-is.
"""

import sys
from pathlib import Path

# Make the package importable no matter what Spyder's working directory is
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cwam_pipeline.run import main  # noqa: E402

# ----------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------
CONFIG = HERE / "config.yaml"
SKIP_DOWNLOAD = False        # True -> reuse files already in data_dir
ECHOTOP_LOCAL_DIR = None     # e.g. r"C:\...\my_echotop_files" to skip S3

argv = ["--config", str(CONFIG)]
if SKIP_DOWNLOAD:
    argv.append("--skip-download")
if ECHOTOP_LOCAL_DIR:
    argv += ["--echotop-local-dir", str(ECHOTOP_LOCAL_DIR)]

main(argv)
