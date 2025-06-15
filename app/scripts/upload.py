import os
import shutil
from datetime import datetime

UPLOAD_DIR = "app/data/uploads"
FIXED_REF_PATH = "app/data/reference/reference.mp4"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def handle_upload(user_file, ref_file=None, fixed_ref=False):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    upload_folder = os.path.join(UPLOAD_DIR, timestamp)
    ensure_dir(upload_folder)

    # Salvar vídeo do usuário
    user_path = os.path.join(upload_folder, "user.mp4")
    with open(user_path, "wb") as f:
        f.write(user_file.read())

    # Se estiver usando uma referência fixa, copia o arquivo interno
    if fixed_ref:
        ref_path = os.path.join(upload_folder, "ref.mp4")
        shutil.copy(FIXED_REF_PATH, ref_path)
    else:
        if ref_file is None:
            raise ValueError("Arquivo de referência não fornecido.")
        ref_path = os.path.join(upload_folder, "ref.mp4")
        with open(ref_path, "wb") as f:
            f.write(ref_file.read())

    return user_path, ref_path
