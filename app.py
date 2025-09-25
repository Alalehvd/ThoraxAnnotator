import json
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Canine Thorax Annotator", layout="wide")

# --------- Paths ---------
IMAGES_DIR = Path("Dataset")
ANN_DIR = Path("annotations")
ANN_DIR.mkdir(parents=True, exist_ok=True)

# --------- Landmarks order (11 points) ---------
POINT_ORDER = [
    "T3_center",
    "T4_cranial",
    "T4_caudal",
    "T5_center",
    "long_axis_start",  # carina
    "long_axis_end",    # cardiac apex
    "short_axis_start",
    "short_axis_end",
    "LA_end",           # caudal bulge of LA
    "trachea_upper",
    "trachea_lower",
]

HELP = {
    "T3_center": "Center of T3 vertebral body.",
    "T4_cranial": "Midpoint of T4 cranial endplate.",
    "T4_caudal": "Midpoint of T4 caudal endplate.",
    "T5_center": "Center of T5 vertebral body.",
    "long_axis_start": "Carina (tracheal bifurcation).",
    "long_axis_end": "Cardiac apex.",
    "short_axis_start": "Heart border point 1 (widest diameter; ⟂ to long axis).",
    "short_axis_end": "Heart border point 2 (opposite side of widest diameter).",
    "LA_end": "Most caudal point of the left atrium.",
    "trachea_upper": "Upper edge of tracheal lumen (thoracic inlet).",
    "trachea_lower": "Lower edge of tracheal lumen (thoracic inlet).",
}

# --------- Colors + short labels ---------
COLOR_MAP = {
    "T3_center":        (0, 153, 255),   # blue
    "T4_cranial":       (255, 64, 64),   # red
    "T4_caudal":        (200, 0, 0),     # dark red
    "T5_center":        (153, 102, 255), # purple
    "long_axis_start":  (0, 200, 0),     # green
    "long_axis_end":    (0, 140, 0),     # dark green
    "short_axis_start": (255, 200, 0),   # yellow
    "short_axis_end":   (220, 160, 0),   # dark yellow
    "LA_end":           (255, 128, 0),   # orange
    "trachea_upper":    (255, 0, 255),   # magenta
    "trachea_lower":    (180, 0, 180),   # dark magenta
}
SHORT = {
    "T3_center": "T3",
    "T4_cranial": "T4cr",
    "T4_caudal": "T4ca",
    "T5_center": "T5",
    "long_axis_start": "Carina",
    "long_axis_end": "Apex",
    "short_axis_start": "SAX1",
    "short_axis_end": "SAX2",
    "LA_end": "LAend",
    "trachea_upper": "TrUp",
    "trachea_lower": "TrLo",
}

def rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

# --------- Cached helpers ---------
@st.cache_data(show_spinner=False)
def cached_list_images(folder: str):
    """List images quickly (skip hidden/AppleDouble/empty)."""
    root = Path(folder)
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        imgs += sorted(root.glob(ext))
    imgs = [p for p in imgs if not (p.name.startswith("._") or p.name.startswith(".")) and p.stat().st_size > 0]
    return [str(p) for p in imgs]

@st.cache_data(show_spinner=False)
def load_base_image(path_str: str, target_w: int = 850):
    """Load file safely, normalize EXIF rotation, force RGB, resize, return (base, scale, h)."""
    p = Path(path_str)
    img = Image.open(p)
    img = ImageOps.exif_transpose(img).convert("RGB")  # avoid EXIF/white issues
    w, h = img.size
    canvas_w = min(target_w, w) if w > 0 else target_w
    scale = canvas_w / w if w > 0 else 1.0
    canvas_h = max(int(h * scale), 1)
    base = img.resize((canvas_w, canvas_h), resample=Image.BILINEAR).copy()  # fresh copy each run
    return base, float(scale), int(canvas_h)

# --------- Draw overlay directly on base image ---------
def make_display_image(base_img: Image.Image, points: dict, scale: float, show_legend: bool = True) -> Image.Image:
    """Draw colored landmarks, labels, and optional legend on a copy of base_img."""
    bg = base_img.copy()
    draw = ImageDraw.Draw(bg)

    try:
        r = 6
        for name, xy in points.items():
            if name not in COLOR_MAP:
                continue
            # Guard against malformed points
            if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                continue
            if xy[0] is None or xy[1] is None:
                continue

            sx = int(float(xy[0]) * scale)
            sy = int(float(xy[1]) * scale)
            color = COLOR_MAP[name]
            draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=color, outline=(0, 0, 0))
            label = SHORT.get(name, name)
            draw.text((sx + 10, sy - 10), label, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))

        if show_legend:
            lx, ly = 10, 10
            box, line_h = 12, 18
            for nm in POINT_ORDER:
                color = COLOR_MAP[nm]
                label = SHORT[nm]
                draw.rectangle((lx, ly, lx + box, ly + box), fill=color, outline=(0, 0, 0))
                draw.text(
                    (lx + box + 8, ly - 2),
                    f"{label} = {nm}",
                    fill=(255, 255, 255),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )
                ly += line_h
    except Exception as e:
        st.warning(f"Overlay drawing skipped due to error: {e}")

    return bg

# --------- Annotation file helpers ---------
def ann_path(img_path: Path) -> Path:
    return ANN_DIR / f"{img_path.stem}.json"

def load_ann(img_path: Path):
    p = ann_path(img_path)
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return None

def save_ann(img_path: Path, points: dict, derived: dict):
    data = {"image": img_path.name, "points": points, "derived": derived}
    with open(ann_path(img_path), "w") as f:
        json.dump(data, f, indent=2)

# --------- Math helpers ---------
def dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def angle_between(p1a, p1b, p2a, p2b):
    v1, v2 = np.array(p1b) - np.array(p1a), np.array(p2b) - np.array(p2a)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(abs(np.degrees(np.arccos(cosang))))

def compute(points):
    d = {}
    has = lambda *k: all(k_i in points for k_i in k)
    if has("T4_cranial", "T4_caudal"):
        d["T4_length"] = dist(points["T4_cranial"], points["T4_caudal"])
    if has("long_axis_start", "long_axis_end"):
        d["LAX"] = dist(points["long_axis_start"], points["long_axis_end"])
    if has("short_axis_start", "short_axis_end"):
        d["SAX"] = dist(points["short_axis_start"], points["short_axis_end"])
    if has("long_axis_start", "LA_end"):
        d["VLAS_length"] = dist(points["long_axis_start"], points["LA_end"])
    if has("trachea_upper", "trachea_lower"):
        d["TD"] = dist(points["trachea_upper"], points["trachea_lower"])
    T4 = d.get("T4_length", 0.0)
    if T4 > 0:
        if "LAX" in d and "SAX" in d:
            d["VHS"] = (d["LAX"] + d["SAX"]) / T4
        if "VLAS_length" in d:
            d["VLAS"] = d["VLAS_length"] / T4
        if "TD" in d:
            d["TD_over_T4"] = d["TD"] / T4
    if has("long_axis_start","long_axis_end","short_axis_start","short_axis_end"):
        d["SAX_LAX_angle_deg"] = angle_between(
            points["long_axis_start"], points["long_axis_end"],
            points["short_axis_start"], points["short_axis_end"]
        )
    return d

# ==================== Sidebar ====================
st.sidebar.title("Canine Thorax Annotator")
st.sidebar.markdown("---")
if st.sidebar.button("Refresh images"):
    cached_list_images.clear()

show_legend = st.sidebar.checkbox("Show legend", value=True)
canvas_target_w = st.sidebar.slider("Canvas width (px)", 700, 1000, 850, step=50)

# Mode widget (separate key) + our own route state
if "view" not in st.session_state:
    st.session_state.view = "Annotate"

# ---- Jump handler: process any requested route/image change BEFORE widgets ----
if "pending_jump" in st.session_state:
    pj = st.session_state.pending_jump
    if "idx" in pj:
        st.session_state.idx = int(pj["idx"])
    if "view" in pj:
        st.session_state.view = pj["view"]
        st.session_state["mode_widget"] = pj["view"]  # pre-set radio UI
    # NEW: force a two-phase canvas mount on next run
    st.session_state.pending_canvas_refresh = True
    del st.session_state.pending_jump
    st.rerun()

mode_widget = st.sidebar.radio(
    "Mode",
    ["Annotate", "Summary"],
    index=0 if st.session_state.view == "Annotate" else 1,
    key="mode_widget",
)

# Keep route in sync with the widget
if mode_widget != st.session_state.view:
    st.session_state.view = mode_widget
    st.rerun()

mode = st.session_state.view

# ==================== Session state ====================
if "canvas_nonce" not in st.session_state:
    st.session_state.canvas_nonce = 0
if "idx" not in st.session_state:
    st.session_state.idx = 0

# ==================== Images ====================
image_paths = [Path(p) for p in cached_list_images(str(IMAGES_DIR))]
if not image_paths:
    st.sidebar.warning("No valid images found. Add PNG/JPG radiographs to the `Dataset/` folder and refresh.")
    st.stop()
names = [p.name for p in image_paths]

# ==================== Annotate mode ====================
if mode == "Annotate":
    choice = st.sidebar.selectbox("Select image:", names, index=st.session_state.idx)
    new_idx = names.index(choice)
    if new_idx != st.session_state.idx:
        st.session_state.idx = new_idx
        st.session_state.canvas_nonce += 1
        st.rerun()

    current = image_paths[st.session_state.idx]

    # Base image
    try:
        display_base, scale, canvas_h = load_base_image(str(current), target_w=canvas_target_w)
    except Exception as e:
        st.error(f"Failed to open `{current.name}`: {e}")
        st.stop()
    canvas_w = display_base.size[0]

    # Load existing points
    if "points" not in st.session_state or st.session_state.get("points_img") != current.name:
        saved = load_ann(current)
        st.session_state.points = saved["points"] if saved else {}
        st.session_state.points_img = current.name
    points = dict(st.session_state.points)  # fresh copy

    # Next required point
    next_name = None
    for nm in POINT_ORDER:
        if nm not in points:
            next_name = nm
            break

    # Header & help
    st.markdown(f"### Image: `{current.name}`")
    with st.expander("Annotation help", expanded=True):
        if next_name:
            st.markdown(f"**Next point:** `{next_name}` — {HELP[next_name]}")
        else:
            st.success("All points placed. You can adjust using the buttons below or overwrite by clicking again.")

    # Overlay
    display_img = make_display_image(display_base, points, scale, show_legend)
    
    # If we just jumped, do a two-phase mount to avoid blank background
    if st.session_state.get("pending_canvas_refresh"):
        st.session_state.pending_canvas_refresh = False
        st.session_state.canvas_nonce += 1   # new canvas key
        st.rerun()

    # Canvas
    stroke = rgb_to_hex(COLOR_MAP.get(next_name, (0, 255, 0)))
    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color=stroke,
        background_image=display_img,   # pass PIL.Image (stable)
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="point",
        display_toolbar=False,
        key=f"canvas_{current.name}_{st.session_state.canvas_nonce}",
    )

    # Click handler
    def near(a, b, tol=5.0):
        return float(np.linalg.norm(np.array(a) - np.array(b))) <= tol

    if canvas.json_data is not None:
        objs = canvas.json_data.get("objects", [])
        if len(objs) > 0 and len(points) < len(POINT_ORDER):
            obj = objs[-1]
            left = float(obj.get("left", 0))
            top = float(obj.get("top", 0))
            radius = float(obj.get("radius", 0) or 0)
            cx = left + radius
            cy = top + radius
            x, y = cx / scale, cy / scale
            pname = POINT_ORDER[len(points)]

            # Debounce near previous point
            last_key = POINT_ORDER[len(points) - 1] if len(points) > 0 else None
            if last_key and last_key in points and near(points[last_key], (x, y)):
                pass
            else:
                points[pname] = [float(x), float(y)]
                st.session_state.points = points
                st.session_state.canvas_nonce += 1
                st.rerun()

    # Editing buttons
    col1, col2, col3 = st.columns(3)
    if col1.button("Reset last point"):
        last = None
        for nm in POINT_ORDER:
            if nm in points:
                last = nm
        if last:
            points.pop(last, None)
            st.session_state.points = points
            st.session_state.canvas_nonce += 1
            st.rerun()

    if col2.button("Clear all points"):
        st.session_state.points = {}
        st.session_state.canvas_nonce += 1
        st.rerun()

    if col3.button("Force redraw"):
        st.session_state.canvas_nonce += 1
        st.rerun()

    # Measurements
    derived = compute(points)
    st.subheader("Derived measurements")
    if derived:
        st.json(derived)
    else:
        st.info("Measurements appear after you place the required points.")

    # Save / Reload
    c1, c2, c3 = st.columns(3)
    if c1.button("Save annotations"):
        save_ann(current, points, derived)
        st.success(f"Saved: {ann_path(current)}")

    if c2.button("Reload saved"):
        loaded = load_ann(current)
        if loaded:
            st.session_state.points = loaded.get("points", {})
            st.session_state.canvas_nonce += 1
            st.rerun()
        else:
            st.warning("No saved annotation for this image.")

    # Prev / Next
    st.sidebar.markdown("---")
    if st.sidebar.button("⟵ Prev") and st.session_state.idx > 0:
        st.session_state.idx -= 1
        st.session_state.canvas_nonce += 1
        st.rerun()

    if st.sidebar.button("Next ⟶") and st.session_state.idx < len(image_paths) - 1:
        st.session_state.idx += 1
        st.session_state.canvas_nonce += 1
        st.rerun()

# ==================== Summary mode ====================
else:
    st.title("Dataset Summary")

    # Collect annotation status for all images (use saved derived if present; otherwise compute on the fly)
    def load_points_and_metrics(p: Path):
        data = load_ann(p)
        pts, drv = (data.get("points", {}), data.get("derived", {})) if data else ({}, {})
        if pts and not drv:
            drv = compute(pts)  # backfill
        return pts, drv

    rows = []
    completed = 0
    for p in image_paths:
        pts, drv = load_points_and_metrics(p)
        done = (len(pts) == len(POINT_ORDER))
        if done:
            completed += 1
        rows.append({
            "image": p.name,
            "completed": done,
            "num_points": len(pts),
            "VHS": drv.get("VHS"),
            "VLAS": drv.get("VLAS"),
            "TD_over_T4": drv.get("TD_over_T4"),
            "SAX_LAX_angle_deg": drv.get("SAX_LAX_angle_deg"),
        })

    df = pd.DataFrame(rows).sort_values(["completed", "image"], ascending=[False, True]).reset_index(drop=True)

    # Progress
    st.subheader("Progress")
    total = len(image_paths)
    st.progress((completed / total) if total else 0.0, text=f"{completed} / {total} images annotated")

    # Filters
    colf1, colf2, colf3 = st.columns(3)
    show_only_incomplete = colf1.checkbox("Show only incomplete", value=False)
    vhs_warn = colf2.checkbox("Flag implausible VHS (<5 or >13)", value=True)
    angle_warn = colf3.checkbox("Flag angle > 95°", value=False)

    df_view = df.copy()
    if show_only_incomplete:
        df_view = df_view[~df_view["completed"]]

    def qc_vhs(v):
        if v is None or pd.isna(v):
            return ""
        return "⚠️" if (v < 5 or v > 13) else ""

    def qc_angle(a):
        if a is None or pd.isna(a):
            return ""
        return "⚠️" if a > 95 else ""

    if vhs_warn:
        df_view["VHS_flag"] = df_view["VHS"].map(qc_vhs)
    if angle_warn:
        df_view["Angle_flag"] = df_view["SAX_LAX_angle_deg"].map(qc_angle)

    # Table
    st.subheader("Table")
    st.dataframe(df_view, use_container_width=True)

    # Download CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download summary CSV", data=csv_bytes, file_name="annotation_summary.csv", mime="text/csv")

    # Thumbnail gallery
    st.subheader("Gallery")
    cols_per_row = 4
    cols = st.columns(cols_per_row)

    @st.cache_data(show_spinner=False)
    def make_thumb(path_str: str, max_w: int = 220):
        try:
            img = Image.open(path_str)
            img = ImageOps.exif_transpose(img).convert("RGB")
            w, h = img.size
            if w > max_w:
                scale = max_w / w
                img = img.resize((max_w, max(1, int(h * scale))), resample=Image.BILINEAR)
            return img
        except Exception:
            return Image.new("RGB", (max_w, int(max_w * 0.75)), color=(30, 30, 30))

    for i, p in enumerate(image_paths):
        c = cols[i % cols_per_row]
        pts, drv = load_points_and_metrics(p)
        done = (len(pts) == len(POINT_ORDER))

        with c.container():
            st.image(make_thumb(str(p)), caption=p.name, use_column_width=True)
            if done:
                st.markdown("✅ **Completed**")
            else:
                st.markdown(f"🟡 {len(pts)}/{len(POINT_ORDER)} points")

            # Key stats
            bits = []
            if drv.get("VHS") is not None:
                bits.append(f"VHS: {drv['VHS']:.2f}")
            if drv.get("VLAS") is not None:
                bits.append(f"VLAS: {drv['VLAS']:.2f}")
            if drv.get("SAX_LAX_angle_deg") is not None:
                bits.append(f"∠: {drv['SAX_LAX_angle_deg']:.1f}°")
            if bits:
                st.caption(" | ".join(bits))

            # Open button: jump to Annotate mode on this image
            if st.button("Open", key=f"open_{p.name}"):
                st.session_state.pending_jump = {
                    "idx": names.index(p.name),
                    "view": "Annotate",
                }
                st.rerun()