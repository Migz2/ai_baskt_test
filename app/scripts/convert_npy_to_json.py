import numpy as np
import json
import os

def convert_npy_to_json(npy_path, output_path):
    data = np.load(npy_path)

    json_data = []
    for frame_idx, flat_frame in enumerate(data):
        frame_data = {
            "frame": int(frame_idx),
            "landmarks": []
        }

        # Reestrutura o frame achatado (132,) para (33, 4)
        keypoints = np.reshape(flat_frame, (-1, 4))

        for kp in keypoints:
            x, y, z, visibility = kp.tolist()
            frame_data["landmarks"].append({
                "x": x,
                "y": y,
                "z": z,
                "visibility": visibility
            })

        json_data.append(frame_data)

    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"✅ Convertido com sucesso: {output_path}")

if __name__ == "__main__":
    npy_path = "app/data/keypoints_data/curry_side_pose.npy"
    output_path = "app/results/reference_keypoints.json"

    if not os.path.exists(npy_path):
        print("❌ Arquivo .npy não encontrado.")
    else:
        convert_npy_to_json(npy_path, output_path)
