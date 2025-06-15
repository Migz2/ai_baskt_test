import cv2
import mediapipe as mp
import tempfile
import os

def gerar_visualizacao_com_sobreposicao(user_video, ref_video):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    drawing = mp.solutions.drawing_utils

    def draw_keypoints_from_file(video_file):
        frames = []
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(video_file.read())
            path = tmp.name
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frames.append(frame)
        cap.release()
        return frames

    user_frames = draw_keypoints_from_file(user_video)
    ref_frames = draw_keypoints_from_file(ref_video)
    h = max(user_frames[0].shape[0], ref_frames[0].shape[0])
    w = user_frames[0].shape[1] + ref_frames[0].shape[1]

    output_path = os.path.join(tempfile.gettempdir(), "output_side_by_side.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))

    for uf, rf in zip(user_frames, ref_frames):
        combined = cv2.hconcat([uf, rf])
        out.write(combined)
    out.release()
    return output_path