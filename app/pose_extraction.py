def extract_keypoints(video_path):
    import cv2
    import numpy as np
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    all_keypoints = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = []
            for i in range(33):
                keypoints.append([landmarks[i].x, landmarks[i].y])
        else:
            # Se não detectar pose, adiciona zeros
            keypoints = [[0.0, 0.0] for _ in range(33)]

        all_keypoints.append(keypoints)

    cap.release()
    pose.close()

    if len(all_keypoints) == 0:
        return None

    keypoints_array = np.array(all_keypoints).astype(np.float32)

    # Verifica se a forma está correta
    if keypoints_array.ndim != 3 or keypoints_array.shape[1:] != (33, 2):
        return None

    return keypoints_array 