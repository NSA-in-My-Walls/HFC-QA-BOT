# Step 1: download_transcripts.py
# --------------------------------
# Fetch all video IDs from a channel and download their YouTube transcripts.

import os
import json
import subprocess

from concurrent.futures import ThreadPoolExecutor
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import xml.parsers.expat
from xml.etree.ElementTree import ParseError

# ─── Configuration ───────────────────────────────────────
CHANNEL_URL  = "https://www.youtube.com/@ChasandJoni/videos"
VIDEOS_JSON  = "videos.json"
OUT_DIR      = "sermon_transcripts"
NUM_THREADS  = 8

def fetch_video_list():
    if not os.path.exists(VIDEOS_JSON):
        print("Fetching video list from yt-dlp...")
        result = subprocess.run(
            ["python", "-m", "yt_dlp", "--flat-playlist", "-J", CHANNEL_URL],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("yt-dlp failed. Here's the error:")
            print(result.stderr)
            exit(1)
        with open(VIDEOS_JSON, "w", encoding="utf-8") as f:
            f.write(result.stdout)

    with open(VIDEOS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [entry["id"] for entry in data.get("entries", [])]

def download_transcript(video_id):
    out_path = os.path.join(OUT_DIR, f"{video_id}.json")
    if os.path.exists(out_path):
        return True

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # transcript is a list of {text, start, duration}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f)
        return True

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"• Skipping {video_id} — no transcript: {e}")
        return False

    except (xml.parsers.expat.ExpatError, ParseError) as e:
        print(f"• Skipping {video_id} — parse error: {e}")
        return False

    except Exception as e:
        print(f"• Error on {video_id}: {e}")
        return False

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    vids = fetch_video_list()
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as exe:
        results = list(tqdm(exe.map(download_transcript, vids), total=len(vids)))
    print(f"Downloaded {sum(results)} / {len(vids)} transcripts.")