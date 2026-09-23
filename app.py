import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision # เพิ่มการ Import Tasks API
import av
import threading
import tempfile
import os
from PIL import Image
import io

# ==========================================
# 1. การตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(page_title="BG Replacer Web App", layout="wide")
st.title("🎥 AI เปลี่ยนพื้นหลัง Real-time (Web App)")
st.markdown("รองรับการเปลี่ยนพื้นหลังเป็น ภาพนิ่ง, วิดีโอ, หรือสีพื้น พร้อมฟังก์ชันถ่ายรูปและอัดวิดีโอ")

# ==========================================
# 2. คลาสประมวลผลวิดีโอ (ทำงานเบื้องหลังด้วย Tasks API)
# ==========================================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # โหลดโมเดล MediaPipe รุ่นใหม่ (Tasks API) เหมือนใน Tkinter
        model_path = 'selfie_segmenter.tflite'
        if not os.path.exists(model_path):
            print(f"⚠️ คำเตือน: ไม่พบไฟล์ {model_path} กรุณานำมาวางในโฟลเดอร์เดียวกับโค้ด")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
        
        # ตัวแปรเก็บการตั้งค่าจาก UI
        self.bg_type = "สีพื้น"
        self.bg_color = (0, 255, 0)
        self.bg_image = None
        self.bg_cap = None
        self.blur_face = False
        
        # ตัวแปรระบบบันทึกและถ่ายรูป
        self.lock = threading.Lock()
        self.latest_frame = None
        self.is_recording = False
        self.video_writer = None

    def recv(self, frame):
        # รับภาพจากกล้อง
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # พลิกภาพเหมือนกระจก

        # ประมวลผลแยกคนกับพื้นหลัง (แปลงรูปแบบสำหรับ Tasks API)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        segmentation_result = self.segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask.numpy_view()
        category_mask = np.squeeze(category_mask)
        
        # สร้าง Mask (ทำให้ขอบเนียนขึ้น)
        mask = (category_mask < 0.1).astype(np.float32)
        mask_blurred = cv2.GaussianBlur(mask, (7, 7), 0)
        mask_3d = np.stack((mask_blurred,) * 3, axis=-1)

        # อ่านค่าการตั้งค่าจาก UI อย่างปลอดภัย
        with self.lock:
            bg_type = self.bg_type
            bg_color = self.bg_color
            bg_image = self.bg_image
            bg_cap = self.bg_cap
            blur_face = self.blur_face
            is_recording = self.is_recording
            video_writer = self.video_writer

        bg_frame = np.zeros(img.shape, dtype=np.uint8)

        # จัดการพื้นหลังตามที่เลือก
        if bg_type == "ภาพนิ่ง" and bg_image is not None:
            bg_frame = cv2.resize(bg_image, (img.shape[1], img.shape[0]))
        
        elif bg_type == "วิดีโอ" and bg_cap is not None:
            ret, b_frame = bg_cap.read()
            if not ret: # ถ้าย้อนวิดีโอจบ ให้วนกลับไปเฟรมแรกใหม่
                bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, b_frame = bg_cap.read()
            if ret:
                bg_frame = cv2.resize(b_frame, (img.shape[1], img.shape[0]))
        
        else: # สีพื้น
            bg_frame[:] = bg_color

        # เอฟเฟกต์ทำหน้าเบลอ (ถ้าเปิด)
        if blur_face:
            img = cv2.GaussianBlur(img, (25, 25), 0)

        # ผสานภาพคน เข้ากับพื้นหลัง (ใช้ทฤษฎี Alpha Blending)
        output_image = (img * mask_3d + bg_frame * (1 - mask_3d)).astype(np.uint8)

        # อัปเดตเฟรมล่าสุด และบันทึกวิดีโอ
        with self.lock:
            self.latest_frame = output_image.copy()
            if is_recording and video_writer is not None:
                video_writer.write(output_image)

        # ส่งภาพกลับไปแสดงบนเว็บ
        return av.VideoFrame.from_ndarray(output_image, format="bgr24")
# ==========================================
# ==========================================
# 3. จัดการ State ของปุ่มกดต่างๆ
# ==========================================
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

# ==========================================
# 4. ส่วนของ UI (หน้าเว็บ)
# ==========================================
col_settings, col_video = st.columns([1, 2.5])

# ---- เมนูด้านซ้าย (ตั้งค่า) ----
with col_settings:
    st.header("⚙️ ตั้งค่าพื้นหลัง")
    bg_type = st.radio("รูปแบบ:", ["สีพื้น", "ภาพนิ่ง", "วิดีโอ"])
    
    bg_image = None
    bg_cap = None
    bg_color = (0, 255, 0)
    
    if bg_type == "สีพื้น":
        hex_color = st.color_picker("เลือกสี", "#00FF00")
        # แปลง HEX เป็น BGR สำหรับ OpenCV
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bg_color = (rgb[2], rgb[1], rgb[0])
        
    elif bg_type == "ภาพนิ่ง":
        uploaded_img = st.file_uploader("อัปโหลดรูปภาพ", type=['jpg', 'jpeg', 'png'])
        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            bg_image = cv2.imdecode(file_bytes, 1)
            
    elif bg_type == "วิดีโอ":
        uploaded_vid = st.file_uploader("อัปโหลดวิดีโอ", type=['mp4', 'mov'])
        if uploaded_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_vid.read())
            bg_cap = cv2.VideoCapture(tfile.name)

    st.markdown("---")
    st.header("✨ ฟังก์ชันเสริม")
    blur_face = st.checkbox("ทำหน้าเบลอ (ปิดบังใบหน้า)")

# ---- เมนูด้านขวา (วิดีโอ & ปุ่มกด) ----
with col_video:
    # แสดงกล้อง (WebRTC)
    webrtc_ctx = webrtc_streamer(
        key="bg-replacer",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False}, # ปิดเสียงไว้ก่อนเพื่อความเสถียรบนเว็บ
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # อัปเดตข้อมูลจาก UI ส่งเข้าไปใน VideoProcessor Thread
    if webrtc_ctx.video_processor:
        with webrtc_ctx.video_processor.lock:
            webrtc_ctx.video_processor.bg_type = bg_type
            webrtc_ctx.video_processor.bg_color = bg_color
            webrtc_ctx.video_processor.bg_image = bg_image
            webrtc_ctx.video_processor.bg_cap = bg_cap
            webrtc_ctx.video_processor.blur_face = blur_face

    st.markdown("---")
    
    # กลุ่มปุ่มกด
    col_btn1, col_btn2 = st.columns(2)
    
    # -- 📸 ปุ่มถ่ายรูป --
    with col_btn1:
        st.subheader("📸 ถ่ายรูปภาพ")
        if st.button("กดเพื่อถ่ายรูป (Snapshot)", use_container_width=True):
            if webrtc_ctx.video_processor and webrtc_ctx.video_processor.latest_frame is not None:
                img_bgr = webrtc_ctx.video_processor.latest_frame
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.session_state.snapshot = Image.fromarray(img_rgb)
            else:
                st.warning("กรุณาเปิดกล้องก่อนถ่ายรูป!")
                
        # ถ้ามีรูปที่ถ่ายไว้ ให้แสดงพร้อมปุ่มดาวน์โหลด
        if st.session_state.snapshot:
            st.image(st.session_state.snapshot, caption="รูปที่ถ่ายล่าสุด", use_container_width=True)

            
            buf = io.BytesIO()
            st.session_state.snapshot.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="⬇️ ดาวน์โหลดรูปภาพ",
                data=byte_im,
                file_name="snapshot.png",
                mime="image/png",
                use_container_width=True
            )

    # -- 🔴 ปุ่มอัดวิดีโอ --
    with col_btn2:
        st.subheader("🔴 บันทึกวิดีโอ")
        
        btn_text = "⏹ หยุดบันทึก" if st.session_state.is_recording else "🔴 เริ่มบันทึกวิดีโอ"
        
        if st.button(btn_text, use_container_width=True):
            if webrtc_ctx.video_processor and webrtc_ctx.video_processor.latest_frame is not None:
                st.session_state.is_recording = not st.session_state.is_recording
                
                with webrtc_ctx.video_processor.lock:
                    if st.session_state.is_recording:
                        # เริ่มอัด
                        h, w = webrtc_ctx.video_processor.latest_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec สำหรับ mp4
                        webrtc_ctx.video_processor.video_writer = cv2.VideoWriter("output_web.mp4", fourcc, 20.0, (w, h))
                        webrtc_ctx.video_processor.is_recording = True
                    else:
                        # หยุดอัด
                        webrtc_ctx.video_processor.is_recording = False
                        if webrtc_ctx.video_processor.video_writer:
                            webrtc_ctx.video_processor.video_writer.release()
                            webrtc_ctx.video_processor.video_writer = None
            else:
                st.warning("กรุณาเปิดกล้องก่อนบันทึกวิดีโอ!")

        # แจ้งสถานะ
        if st.session_state.is_recording:
            st.error("🎥 กำลังบันทึกวิดีโอ... กดปุ่ม 'หยุดบันทึก' เมื่อเสร็จสิ้น")
            
        # ถ้าไม่ได้อัดอยู่ และมีไฟล์วิดีโอเดิมอยู่ ให้โชว์ปุ่มดาวน์โหลด
        if not st.session_state.is_recording and os.path.exists("output_web.mp4"):
            with open("output_web.mp4", "rb") as video_file:
                st.download_button(
                    label="⬇️ ดาวน์โหลดวิดีโอที่บันทึกไว้",
                    data=video_file,
                    file_name="recorded_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
