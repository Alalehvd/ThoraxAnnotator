# Canine Thorax Annotator 🐾

A **Streamlit-based annotation tool** for veterinary radiographs.  
This app lets you mark **thoracic landmarks** on canine lateral radiographs and automatically compute measurements such as **VHS (Vertebral Heart Score), VLAS, tracheal diameter ratio, and SAX/LAX angle**. It also includes a **Summary** view for progress tracking, QC flags, CSV export, and a thumbnail gallery with “Open” to jump to any image.

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

---

## 📂 Project Structure
