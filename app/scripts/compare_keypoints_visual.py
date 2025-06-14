import cv2
import json
import numpy as np
import time
import os
import sys
from app.core.constants import POINT_COLOR, LINE_COLOR, RADIUS, THICKNESS, POSE_CONNECTIONS


# Caminhos dos arquivos
user_json_path = "app/results/user_keypoints.json"
ref_json_path = "app/results/reference_keypoints.json"
user_video_path = "app/videos/user.mp4"
ref_video_path = "app/videos/ref.mp4"


def draw_keypoints(keypoints, frame):
    for idx, kp in enumerate(keypoints):
        if kp[2] > 0.3:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), RADIUS, POINT_COLOR, -1)

    for a, b in POSE_CONNECTIONS:
        if keypoints[a][2] > 0.3 and keypoints[b][2] > 0.3:
            x1, y1 = int(keypoints[a][0]), int(keypoints[a][1])
            x2, y2 = int(keypoints[b][0]), int(keypoints[b][1])
            cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, THICKNESS)
    return frame

def load_keypoints(path, frame_width, frame_height):
    with open(path, 'r') as f:
        raw_data = json.load(f)

    converted = []
    for frame in raw_data:
        converted_frame = []

        if isinstance(frame, dict) and "landmarks" in frame:
            for kp in frame["landmarks"]:
                try:
                    x = float(kp["x"]) * frame_width
                    y = float(kp["y"]) * frame_height
                    score = float(kp.get("visibility", 1.0))
                    converted_frame.append([x, y, score])
                except:
                    continue
        elif isinstance(frame, list):
            for kp in frame:
                try:
                    x, y, score = map(float, kp[:3])
                    converted_frame.append([x, y, score])
                except:
                    continue

        if converted_frame:
            converted.append(converted_frame)

    return converted

def visualize_overlay(user_kps, ref_kps, user_video_path, ref_video_path):
    print("🎥 Iniciando visualização sobre o vídeo...")

    user_cap = cv2.VideoCapture(user_video_path)
    ref_cap = cv2.VideoCapture(ref_video_path)

    total_frames = min(len(user_kps), len(ref_kps),
                       int(user_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                       int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    while user_cap.isOpened() and ref_cap.isOpened():
        ret_u, frame_u = user_cap.read()
        ret_r, frame_r = ref_cap.read()

        if not ret_u or not ret_r:
            break

        idx = int(user_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if idx >= total_frames:
            break

        user_frame = draw_keypoints(user_kps[idx], frame_u.copy())
        ref_frame = draw_keypoints(ref_kps[idx], frame_r.copy())

        combined = np.hstack((ref_frame, user_frame))
        cv2.putText(combined, "Referência", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(combined, "Usuário", (ref_frame.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Esqueleto sobre vídeo - Referência vs Usuário", combined)

        key = cv2.waitKey(30)
        if key == 27:
            break

    user_cap.release()
    ref_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(user_json_path) or not os.path.exists(ref_json_path):
        print("❌ Arquivos JSON não encontrados.")
        exit()

    # Captura dimensões dos vídeos
    cap_u = cv2.VideoCapture(user_video_path)
    cap_r = cv2.VideoCapture(ref_video_path)
    height_u = int(cap_u.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_u = int(cap_u.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_u.release()
    cap_r.release()

    user_kps = load_keypoints(user_json_path, width_u, height_u)
    ref_kps = load_keypoints(ref_json_path, width_r, height_r)

    print(f"Usuário: {len(user_kps)} frames")
    print(f"Referência: {len(ref_kps)} frames")

    visualize_overlay(user_kps, ref_kps, user_video_path, ref_video_path)
