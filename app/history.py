import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

def save_analysis_to_history(score: float, video_paths: list[str]) -> None:
    """
    Save analysis data to a history JSON file.
    
    Args:
        score (float): The analysis score
        video_paths (list[str]): List of paths to the analyzed videos
    """
    history_file = Path("history.json")
    
    # Create empty list if file doesn't exist
    if not history_file.exists():
        history_data = []
    else:
        # Read existing data
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
    
    # Create new entry
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "score": score,
        "video_paths": video_paths
    }
    
    # Add new entry to history
    history_data.append(new_entry)
    
    # Save updated history
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=4, ensure_ascii=False)

def load_history() -> List[Dict]:
    """
    Load the analysis history from the JSON file.
    
    Returns:
        List[Dict]: A list of dictionaries containing the analysis history.
                   Each dictionary contains 'timestamp', 'score', and 'video_paths'.
                   Returns an empty list if the file doesn't exist.
    """
    history_file = Path("history.json")
    
    if not history_file.exists():
        return []
    
    with open(history_file, 'r', encoding='utf-8') as f:
        return json.load(f) 