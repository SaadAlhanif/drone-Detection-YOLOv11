import os
import tempfile
import subprocess

import streamlit as st
from ultralytics import YOLO
import cv2
import imageio_ffmpeg


# =========================
# Page UI
# =========================
st.set_page_config(page_title="Drone Detection", layout="wide")
st.title("🛸 Drone Detection (Video)")
st.write("ارفع فيديو، والنظام بيطلع لك فيديو عليه كشف الدرون + كلمة Drone فوقه.")


# =========================
# Model
# =========================
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ ما لقيت ملف المودل: {MODEL_PATH}\n"
            f"تأكد ان best.pt موجود بنفس مجلد app.py"
        )
    return YOLO(MODEL_PATH)

model = load_model()


# =========================
# Controls
# =========================
st.sidebar.header("⚙️ Settings")
conf_thres = st.sidebar.slider("Confidence", 0.05, 0.95, 0.30, 0.05)
iou_thres  = st.sidebar.slider("IoU", 0.05, 0.95, 0.50, 0.05)

uploaded = st.file_uploader("📤 ارفع فيديو (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"])

if uploaded is None:
    st.info("ارفع فيديو عشان نبدأ.")
    st.stop()


# =========================
# Save input to temp
# =========================
tmp_dir = tempfile.mkdtemp()
input_path = os.path.join(tmp_dir, uploaded.name)

with open(input_path, "wb") as f:
    f.write(uploaded.getbuffer())

st.success("✅ تم رفع الفيديو")


# =========================
# Read video info
# =========================
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    st.error("❌ ما قدرت أفتح الفيديو. جرّب فيديو ثاني.")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30.0

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0


# =========================
# Output writer (temp)
# =========================
raw_output_path = os.path.join(tmp_dir, "output_raw.mp4")

# mp4v works for writing, but may not play in browser -> we convert later to H.264
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

if not writer.isOpened():
    st.error("❌ ما قدرت أفتح VideoWriter. جرّب فيديو ثاني.")
    cap.release()
    st.stop()


# =========================
# UI layout
# =========================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎬 الفيديو الأصلي")
    with open(input_path, "rb") as f:
        st.video(f.read())

with col2:
    st.subheader("✅ المعالجة (يتم تجهيزها...)")
    progress = st.progress(0)
    status = st.empty()


# =========================
# Process frames
# =========================
LABEL_TEXT = "Drone"
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    results = model.predict(frame, conf=conf_thres, iou=iou_thres, verbose=False)

    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0]) if b.conf is not None else 0.0

                # box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # label background
                txt = f"{LABEL_TEXT} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                y_top = max(y1 - th - 10, 0)
                cv2.rectangle(frame, (x1, y_top), (x1 + tw + 8, y1), (0, 255, 0), -1)

                # label text
                cv2.putText(
                    frame,
                    txt,
                    (x1 + 4, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

    writer.write(frame)

    # progress
    if total_frames > 0:
        p = min(frame_idx / total_frames, 1.0)
        progress.progress(int(p * 100))
        status.write(f"Processing frame {frame_idx}/{total_frames} ...")
    else:
        if frame_idx % 30 == 0:
            status.write(f"Processing frame {frame_idx} ...")


cap.release()
writer.release()

progress.progress(100)
status.write("✅ Finished!")


# =========================
# Convert to H.264 for browser playback
# =========================
final_output_path = os.path.join(tmp_dir, "output_h264.mp4")

def convert_to_h264(src, dst):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-i", src,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        dst
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

try:
    convert_to_h264(raw_output_path, final_output_path)
    playable_path = final_output_path
except Exception as e:
    st.warning(f"تحويل H.264 فشل، بعرض الملف الأصلي. Error: {e}")
    playable_path = raw_output_path


# =========================
# Show output (bytes)
# =========================
st.subheader("📌 الفيديو الناتج")

with open(playable_path, "rb") as f:
    out_bytes = f.read()

st.video(out_bytes)

st.download_button(
    "⬇️ Download result video",
    data=out_bytes,
    file_name="drone_detection_output.mp4",
    mime="video/mp4"
)

st.caption("ملاحظة: إذا كان الفيديو طويل جدًا ممكن ياخذ وقت بالمعالجة على Streamlit Cloud.")
