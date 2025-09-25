# Canine Thorax Annotator 🐾

A **Streamlit-based annotation tool** for veterinary radiographs.  

This app lets you mark **thoracic landmarks** on canine lateral radiographs and automatically compute measurements such as **VHS (Vertebral Heart Score), VLAS, tracheal diameter ratio, and SAX/LAX angle**. 

It also includes a **Summary** view for progress tracking, QC flags, CSV export, and a thumbnail gallery with “Open” to jump to any image.

---

## ✨ Features
- Interactive annotation using `streamlit-drawable-canvas`.
- 11 standardized landmarks:
  - T3_center, T4_cranial, T4_caudal, T5_center  
  - long_axis_start (carina), long_axis_end (apex)  
  - short_axis_start, short_axis_end  
  - LA_end (caudal LA bulge)  
  - trachea_upper, trachea_lower
- Auto-computed measurements:
  - T4 length, VHS, VLAS, TD/T4, SAX–LAX angle
- Colored points + labels with a customizable legend
- Robust image handling (EXIF transpose, RGB normalization)
- **Summary** mode:
  - Progress bar
  - Filterable table with QC flags
  - CSV export
  - Thumbnail gallery with **Open** → jump to Annotate

⸻

## 🖼️ Usage

Annotate mode
	1.	Put .png/.jpg images in the Dataset/ folder.
	2.	Select an image from the sidebar.
	3.	Click once per landmark in the guided order (see “Annotation help”).
	4.	Buttons:
	•	Reset last point – remove the most recent landmark.
	•	Clear all points – remove all landmarks for this image.
	•	Force redraw – remount the canvas if a render looks stale.
	5.	Save annotations writes JSON to annotations/.

Summary mode
	•	Track dataset progress (completed vs total).
	•	Filter to incomplete cases, flag implausible VHS or large angles.
	•	Download all measurements as CSV.
	•	Use Open on a thumbnail to jump back to Annotate on that image.

⸻

## 🔎 Outputs

For each image X.png, the app writes annotations/X.json:

<pre>
json
{
  "image": "X.png",
  "points": { 
    "T3_center": [x, y], 
    "T4_cranial": [x, y], 
    "..." : [x, y] 
  },
  "derived": {
    "T4_length": float,
    "LAX": float,
    "SAX": float,
    "VHS": float,
    "VLAS_length": float,
    "VLAS": float,
    "TD": float,
    "TD_over_T4": float,
    "SAX_LAX_angle_deg": float
  }
}
</pre>


Coordinates are in original image pixels.

⸻

## 🏷️ Customizing the Legend

Edit LEGEND_TEXT in app.py to control the legend text.
You can also edit the color map and short labels for on-image tags.

⸻

## 🚀 Roadmap
	•	Export overlays with helper lines (LAX/SAX/T4).
	•	Multi-annotator support + agreement metrics (ICC, Bland–Altman).
	•	COCO-style export for ML pipelines.
	•	Keyboard shortcuts where Streamlit allows.

⸻

## 🛡️ License

This repository is proprietary (all rights reserved).

⸻

## 🙏 Acknowledgements

This project was made possible thanks to:
	•	Streamlit – for providing such a powerful and easy-to-use framework.
	•	streamlit - drawable-canvas – for enabling interactive annotations.
	•	Pillow and NumPy – for all image handling and geometry math.
	•	Pandas – for summary tables and dataset tracking.
	•	And of course, the open-source community, for building the tools that made this possible.

Special thanks to all colleagues and collaborators who inspired this project and provided feedback during its development.
