# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 23:19:46 2026

@author: anomi
"""

import os
import re
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont

# %%

# -----------------------
# User settings
# -----------------------
png_dir = r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\Weather_stuff\mesonet_img\final_masked\150kt_test"
gif_path = r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\Weather_stuff\mesonet_img\geoTIFF\screenshot_for_gif\output_gif\presentation.gif"

frame_duration_ms = 800  # playback speed

timestamp_position = (20, 20)  # (x, y) in pixels
font_size = 48
font_color = "white"
font_path = r"C:\Windows\Fonts\arial.ttf"  # e.g. r"C:\Windows\Fonts\arial.ttf"

# OPTIONAL: if you want absolute timestamps instead of T+MM min
use_absolute_time = False
# only used if use_absolute_time=True
start_time = datetime(2026, 1, 25, 18, 0, 0)
timezone_suffix = "EST"  # or "UTC", etc.

# -----------------------
# Helpers
# -----------------------
# Matches "...t00..." "...t05..." "...t120..." etc. (supports 2+ digits)
TMIN_RE = re.compile(r"t(\d{2,})", re.IGNORECASE)


def extract_minutes(filename: str) -> int:
    m = TMIN_RE.search(filename)
    if not m:
        raise ValueError(
            f"Could not find 't##' minutes in filename: {filename}")
    return int(m.group(1))


def load_font():
    if font_path:
        return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def draw_timestamp(img, text, pos):
    draw = ImageDraw.Draw(img)
    x, y = pos
    # outline for readability
    outline = "black"
    for dx in (-1, 1):
        for dy in (-1, 1):
            draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=font_color)


# -----------------------
# Collect + sort frames by extracted minutes
# -----------------------
png_files = [f for f in os.listdir(png_dir) if f.lower().endswith(".png")]
frames_with_t = []

for f in png_files:
    try:
        tmin = extract_minutes(f)
        frames_with_t.append((tmin, f))
    except ValueError:
        # skip files that don't match the t## pattern
        pass

if not frames_with_t:
    raise RuntimeError("No PNGs with a 't##' pattern found in the directory.")

frames_with_t.sort(key=lambda x: x[0])  # sort by minutes

# -----------------------
# Build stamped frames
# -----------------------
font = load_font()
frames = []

for tmin, fname in frames_with_t:
    path = os.path.join(png_dir, fname)
    img = Image.open(path).convert("RGBA")

    if use_absolute_time:
        ts = start_time + timedelta(minutes=tmin)
        ts_str = ts.strftime(f"%Y-%m-%d %H:%M {timezone_suffix}")
    else:
        ts_str = f"T+{tmin:02d} min"

    draw_timestamp(img, ts_str, timestamp_position)
    frames.append(img)

# -----------------------
# Save GIF
# -----------------------
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=frame_duration_ms,
    loop=0
)

print(f"GIF saved to: {gif_path}")
print(f"Frames used: {len(frames)} (from t={
      frames_with_t[0][0]} to t={frames_with_t[-1][0]} minutes)")
