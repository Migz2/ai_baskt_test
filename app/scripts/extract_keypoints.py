import mediapipe as mp
import cv2
import json

def extract_and_save_keypoints(video_path, output_json):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    keypoints_all = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            frame_data = []
            for lm in results.pose_landmarks.landmark:
                frame_data.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })
            keypoints_all.append({"landmarks": frame_data})
    cap.release()

    with open(output_json, "w") as f:
        json.dump(keypoints_all, f)
