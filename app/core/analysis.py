import numpy as np

def calculate_similarity(user_keypoints, ref_keypoints):
    """
    Compara os keypoints do usuário com os da referência e retorna um score por parte do corpo.
    """
    if not user_keypoints or not ref_keypoints:
        return {}

    partes_corpo = {
        "cotovelo": ["left_shoulder", "left_elbow", "left_wrist"],
        "tronco": ["left_hip", "left_shoulder"],
        "sequencia": ["left_knee", "left_hip", "left_elbow"]
    }

    def get_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    scores = {}
    for parte, pontos in partes_corpo.items():
        user_angles, ref_angles = [], []

        for u_frame, r_frame in zip(user_keypoints, ref_keypoints):
            try:
                ua = get_angle(u_frame[pontos[0]], u_frame[pontos[1]], u_frame[pontos[2]])
                ra = get_angle(r_frame[pontos[0]], r_frame[pontos[1]], r_frame[pontos[2]])
                user_angles.append(ua)
                ref_angles.append(ra)
            except Exception:
                continue

        if user_angles and ref_angles:
            diff = np.abs(np.array(user_angles) - np.array(ref_angles))
            score = 1 - np.mean(diff) / 90  # 0 a 1
            scores[parte] = np.clip(score, 0, 1)
        else:
            scores[parte] = 0

    return scores
