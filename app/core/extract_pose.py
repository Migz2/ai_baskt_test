import cv2
import mediapipe as mp
import json
import os

mp_pose = mp.solutions.pose

def extract_keypoints_from_video(video_path, output_json_path):
    # Garante que o diretório do JSON exista
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Inicializa o modelo de pose
    pose = mp_pose.Pose(
        static_image_mode=True,  # útil para garantir detecção em cada frame
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    keypoints_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Exibe o frame na tela para debug
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                })
            keypoints_data.append({
                'frame': frame_count,
                'landmarks': landmarks
            })
            print(f"[DEBUG] Keypoints detectados no frame {frame_count}")
        else:
            print(f"[DEBUG] Nenhum keypoint detectado no frame {frame_count}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    with open(output_json_path, 'w') as f:
        json.dump(keypoints_data, f, indent=4)

    print(f"[INFO] Keypoints salvos em: {output_json_path}")
