import os
import json
import librosa
import soundfile as sf
from yt_dlp import YoutubeDL

def download_youtube(url, path="./downloads"):
    if not os.path.exists(path):
        os.makedirs(path)
    outtmpl = os.path.join(path, '%(id)s.mp4')
    ydl_opts = {
        'format': 'mp4/bestvideo/best',
        # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        #'postprocessors': [{  # Extract audio using ffmpeg
        #    'key': 'FFmpegExtractAudio',
        #    'preferredcodec': 'mp4',
        #}],
        'outtmpl': outtmpl
    }
    data = {}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        data = json.dumps(ydl.sanitize_info(info))
        data = json.loads(data)
        #error_code = ydl.download(url)
    return data

def extract_audio_from_video(video_path, audio_path):
   audio , sr = librosa.load(video_path)
   sf.write(audio_path, audio, sr)