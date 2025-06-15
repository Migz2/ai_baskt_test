def compare_keypoints(user_kps, ref_kps):
    differences = []
    for u_frame, r_frame in zip(user_kps, ref_kps):
        frame_diff = sum(
            ((ux - rx)**2 + (uy - ry)**2)**0.5
            for (ux, uy, _), (rx, ry, _) in zip(u_frame, r_frame)
        ) / len(u_frame)
        differences.append(frame_diff)

    avg_diff = sum(differences) / len(differences)
    score = max(0, int(100 - avg_diff * 100))  # Score final
    return differences, score
