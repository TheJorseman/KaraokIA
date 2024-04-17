import os
import json
import librosa
import soundfile as sf
from yt_dlp import YoutubeDL

def download_youtube(url: str, path: str = "./downloads"):
    """
    Download a youtube video in the best mp4 format and return all the metadata of the video.

    Args:
        url (str): URL of the video
        path (str, optional): Path to download the video. Defaults to "./downloads".

    Returns:
        dict: Metadata of the video
    """
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

def extract_audio_from_video(video_path: str, audio_path: str):
    """
    Extract the audio form a video.

    Args:
        video_path (str): Video path
        audio_path (str): Path to save the extracted audio
    """
    audio , sr = librosa.load(video_path)
    sf.write(audio_path, audio, sr)