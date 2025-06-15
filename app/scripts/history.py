import json
import os
from datetime import datetime

HISTORY_FILE = "app/data/history.json"

def save_analysis_result(score, user_video, ref_video):
    result = {
        "score": int(round(score)),
        "user_video": os.path.basename(user_video),
        "ref_video": os.path.basename(ref_video),
        "timestamp": datetime.now().isoformat()
    }

    history = load_history()
    history.append(result)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)
