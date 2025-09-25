Canine Thorax Annotator

A fast, clinician-friendly Streamlit app for annotating canine lateral thoracic radiographs and computing derived metrics (VHS, VLAS, tracheal diameter ratio, SAX/LAX angle). Includes a Summary view for progress tracking, QC flags, CSV export, and thumbnail “jump to image”.

⸻

Features
	•	Point-and-click annotation of 11 landmarks (in order):
	•	T3_center, T4_cranial, T4_caudal, T5_center
	•	long_axis_start (carina), long_axis_end (apex)
	•	short_axis_start, short_axis_end
	•	LA_end (caudal LA bulge)
	•	trachea_upper, trachea_lower
	•	Auto-computed metrics:
VHS, VLAS, tracheal diameter ratio (TD/T4), SAX–LAX angle.
	•	Legend & colored labels drawn on the image (customizable).
	•	Robust image handling (EXIF transpose, RGB normalization).
	•	Summary mode:
	•	Overall progress bar
	•	Filterable table with QC flags (e.g., VHS out of plausible range)
	•	CSV export
	•	Thumbnail gallery with Open → jump to Annotate
	•	Resilient UI: single-click save, reliable Prev/Next, clear all, force redraw, and jump routing.

⸻

Quick start

1) Prerequisites
	•	Python 3.13 (tested on 3.13.6)
	•	macOS/Linux/Windows
	•	A folder of radiographs in PNG/JPG/JPEG format.
