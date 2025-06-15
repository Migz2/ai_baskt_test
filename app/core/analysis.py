import numpy as np
from app.core.compare_pose import compare_keypoints

def convert_json_to_array(json_data):
    converted = []
    for frame in json_data:
        frame_data = []
        for i in sorted(frame["keypoints"].keys(), key=int):
            kp = frame["keypoints"][i]
            frame_data.append((kp["x"], kp["y"], kp["z"]))
        converted.append(frame_data)
    return converted

def calculate_similarity(user_json, ref_json):
    user_keypoints = convert_json_to_array(user_json["keypoints"])
    ref_keypoints = convert_json_to_array(ref_json["keypoints"])
    differences, score = compare_keypoints(user_keypoints, ref_keypoints)
    return differences
