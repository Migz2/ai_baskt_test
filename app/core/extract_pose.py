import cv2
import mediapipe as mp
import tempfile

def extrair_keypoints(video_file):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    keypoints_all = []

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            frame_keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            keypoints_all.append(frame_keypoints)

    cap.release()
    return keypoints_all

