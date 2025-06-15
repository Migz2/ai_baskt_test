import cv2
import mediapipe as mp
import json

mp_pose = mp.solutions.pose

def extract_pose_from_video(video_path, output_json_path):
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    keypoints_data = []
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        frame_keypoints = {}

        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                frame_keypoints[str(i)] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                }

        keypoints_data.append({
            "frame": frame_idx,
            "keypoints": frame_keypoints
        })
        frame_idx += 1

    cap.release()

    with open(output_json_path, 'w') as f:
        json.dump(keypoints_data, f)

    return keypoints_data

def extract_keypoints(user_path, ref_path):
    user_json = extract_pose_from_video(user_path, "app/data/user_keypoints.json")
    ref_json = extract_pose_from_video(ref_path, "app/data/ref_keypoints.json")
    return {"keypoints": user_json}, {"keypoints": ref_json}
