# app/extract_pose.py
import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            frame_kp = []
            for lm in results.pose_landmarks.landmark:
                frame_kp.extend([lm.x, lm.y, lm.z, lm.visibility])
            keypoints.append(frame_kp)

    cap.release()
    return np.array(keypoints)

if __name__ == "__main__":
    video_file = "app/reference_videos/curry_side.mp4"  # Altere aqui
    keypoints = extract_keypoints_from_video(video_file)

    os.makedirs("app/keypoints_data", exist_ok=True)
    np.save("app/keypoints_data/curry_side_pose.npy", keypoints)

    print(f"✅ Keypoints salvos: {keypoints.shape}")
