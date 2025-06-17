import cv2
import numpy as np
from shot_detection import ShotDetector
from pathlib import Path

def test_shot_detection():
    # Initialize paths
    video_dir = Path("app/videos")
    ref_video = video_dir / "ref.mp4"
    user_video = video_dir / "user.mp4"
    
    # Initialize shot detector
    detector = ShotDetector()
    
    # Test reference video
    print("\nTesting reference video...")
    ref_shots = detector.process_video(str(ref_video))
    print(f"Number of shots detected in reference video: {len(ref_shots)}")
    for i, (frame_number, ball_bbox) in enumerate(ref_shots):
        print(f"Shot {i+1}: Frame {frame_number}, Ball position: {ball_bbox}")
    
    # Test user video
    print("\nTesting user video...")
    user_shots = detector.process_video(str(user_video))
    print(f"Number of shots detected in user video: {len(user_shots)}")
    for i, (frame_number, ball_bbox) in enumerate(user_shots):
        print(f"Shot {i+1}: Frame {frame_number}, Ball position: {ball_bbox}")
    
    return ref_shots, user_shots

if __name__ == "__main__":
    ref_shots, user_shots = test_shot_detection() 