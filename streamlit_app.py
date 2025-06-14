import streamlit as st
from app.scripts.compare_keypoints_visual import load_keypoints, draw_keypoints
import cv2
import os
import numpy as np
import tempfile

st.set_page_config(page_title="Análise Técnica de Tênis", layout="wide")

st.title("🎾 Análise Técnica de Tênis – MVP")

# Upload de vídeos
st.header("1. Envie seus vídeos")
col1, col2 = st.columns(2)

with col1:
    user_video = st.file_uploader("🎥 Vídeo do Usuário", type=["mp4"], key="user")
with col2:
    ref_video = st.file_uploader("📹 Vídeo de Referência", type=["mp4"], key="ref")

if user_video and ref_video:
    temp_dir = tempfile.mkdtemp()
    user_path = os.path.join(temp_dir, "user.mp4")
    ref_path = os.path.join(temp_dir, "ref.mp4")

    with open(user_path, "wb") as f:
        f.write(user_video.read())
    with open(ref_path, "wb") as f:
        f.write(ref_video.read())

    st.success("Vídeos enviados com sucesso.")

    # Processamento simplificado dos keypoints
    from app.scripts.extract_keypoints import extract_and_save_keypoints
    extract_and_save_keypoints(user_path, "app/results/user_keypoints.json")
    extract_and_save_keypoints(ref_path, "app/results/reference_keypoints.json")

    st.header("2. Visualização dos esqueletos lado a lado")

    cap_u = cv2.VideoCapture(user_path)
    cap_r = cv2.VideoCapture(ref_path)

    width_u = int(cap_u.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_u = int(cap_u.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_u.release()
    cap_r.release()

    user_kps = load_keypoints("app/results/user_keypoints.json", width_u, height_u)
    ref_kps = load_keypoints("app/results/reference_keypoints.json", width_r, height_r)

    stframe = st.empty()
    cap_u = cv2.VideoCapture(user_path)
    cap_r = cv2.VideoCapture(ref_path)

    total_frames = min(len(user_kps), len(ref_kps),
                       int(cap_u.get(cv2.CAP_PROP_FRAME_COUNT)),
                       int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))

    while cap_u.isOpened() and cap_r.isOpened():
        ret_u, frame_u = cap_u.read()
        ret_r, frame_r = cap_r.read()

        if not ret_u or not ret_r:
            break

        idx = int(cap_u.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if idx >= total_frames:
            break

        user_frame = draw_keypoints(user_kps[idx], frame_u.copy())
        ref_frame = draw_keypoints(ref_kps[idx], frame_r.copy())

        combined = np.hstack((ref_frame, user_frame))
        combined = cv2.resize(combined, (1000, 400))
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        stframe.image(combined, channels="RGB", use_container_width=True)

    cap_u.release()
    cap_r.release()
