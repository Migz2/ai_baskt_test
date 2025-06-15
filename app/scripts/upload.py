import os
import shutil

def handle_upload(user_file, ref_file):
    upload_folder = "videos"
    os.makedirs(upload_folder, exist_ok=True)

    user_path = os.path.join(upload_folder, "user.mp4")
    ref_path = os.path.join(upload_folder, "ref.mp4")

    with open(user_path, "wb") as f:
        f.write(user_file.getbuffer())

    with open(ref_path, "wb") as f:
        f.write(ref_file.getbuffer())

    return user_path, ref_path
