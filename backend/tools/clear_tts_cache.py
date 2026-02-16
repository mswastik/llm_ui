#!/usr/bin/env python3
"""
Clear TTS cache files
"""
import os
import glob

# Get the uploads directory
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads")

def clear_tts_cache():
    """Clear all TTS cache files"""
    if not os.path.exists(UPLOAD_DIR):
        print(f"Uploads directory not found at {UPLOAD_DIR}")
        return
    
    # Find all TTS files
    tts_files = glob.glob(os.path.join(UPLOAD_DIR, "tts_*"))
    
    if not tts_files:
        print("No TTS cache files found")
        return
    
    print(f"Found {len(tts_files)} TTS cache files to delete")
    
    for file_path in tts_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    print(f"Successfully cleared {len(tts_files)} TTS cache files")

if __name__ == "__main__":
    clear_tts_cache()
