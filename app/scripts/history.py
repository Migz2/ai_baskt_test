import json
import os

HIST_PATH = "history.json"

def save_analysis_result(result):
    history = load_history()
    history.append(result)
    with open(HIST_PATH, "w") as f:
        json.dump(history, f)

def load_history():
    if not os.path.exists(HIST_PATH):
        return []
    with open(HIST_PATH, "r") as f:
        return json.load(f)
