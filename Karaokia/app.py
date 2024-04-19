import os
import torch
from typing import Union
import whisperx
from torchaudio.transforms import Resample
from Karaokia.download.youtube_download import download_youtube, extract_audio_from_video
from Karaokia.video.editor import Editor
from .transforms import Mono, Compose
from Karaokia.audio_separation.audio_separation import AudioSeparation
from Karaokia.database.database_karaokia import DBManager
from urllib.parse import urlparse, parse_qs
import numpy as np
import json
import gc
from timeit import default_timer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = "cuda" if torch.cuda.is_available() else "cpu"

class KaraokIA:
    def __init__(
            self, 
            download_folder:str = "./downloads", 
            compute_type: str = "float16",
            whisper_path: str = "./weights/faster-whisper-large-v3",
            db_name: str = "karaokia.db",
            torch_hub_dir: str = "./weights",
            HF_TOKEN:str = "",
            generated_folder:str = "./generated",
        ) -> None:
        torch.hub.set_dir(torch_hub_dir)
        self.HF_TOKEN = HF_TOKEN
        self.download_folder = download_folder
        self.compute_type = compute_type
        self.db_manager = DBManager(name=db_name)
        self.model = whisperx.load_model(whisper_path, device, compute_type=compute_type)
        self.audio_separation = AudioSeparation(two_stems=True)
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.HF_TOKEN, device=device)
        
        self.resample = Resample(self.audio_separation.model.samplerate, 16000)
        self.to_mono = Mono()
        self.demucs_to_whisper = Compose([
            self.resample,
            self.to_mono 
        ])
        self.generated_folder = generated_folder
        self.editor = Editor()


    def video_id(self, value: str):
        """
        Reference:
            https://stackoverflow.com/questions/4356538/how-can-i-extract-video-id-from-youtubes-link-in-python
        Examples:
        - http://youtu.be/SA2iWivDJiE
        - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
        - http://www.youtube.com/embed/SA2iWivDJiE
        - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
        """
        query = urlparse(value)
        if query.hostname == 'youtu.be':
            return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                p = parse_qs(query.query)
                return p['v'][0]
            if query.path[:7] == '/embed/':
                return query.path.split('/')[2]
            if query.path[:3] == '/v/':
                return query.path.split('/')[2]
        # fail?
        return None


    def create_outputs_paths(self, _id: str):
        video_output = os.path.join(self.download_folder, f"{_id}.mp4")
        audio_output = os.path.join(self.download_folder, f"{_id}.wav")
        return video_output, audio_output

    def download_process(self, url: str):
        metadata = download_youtube(url, path=self.download_folder)
        video_output, audio_output = self.create_outputs_paths(metadata['id'])
        extract_audio_from_video(video_output, audio_output)
        metadata.update({
            'video_output': video_output,
            'audio_output': audio_output
        })
        return metadata
    
    def audio_separation_process(self, metadata: dict) -> dict[str: any]:
        separated = self.audio_separation.from_file(metadata['audio_output'])
        base_files = self.audio_separation.save_audio({'base': separated['base']}, self.download_folder, _id=metadata['id'])
        vocals = self.demucs_to_whisper(separated['vocals'])
        self.audio_separation.deallocate()
        self.free()
        return {
            'base': base_files['base'],
            'vocals': vocals.numpy(),
        }

    def speech_recognition(self, source: Union[str, np.ndarray], batch_size:int = 16, language: Union[str, None] = 'es'):
        audio = source
        if isinstance(source, str):
            audio = whisperx.load_audio(source)
        result = self.model.transcribe(audio, batch_size=batch_size, language=language, print_progress=True)
        self.free()
        return audio, result 

    def alignt_output(self, audio: np.ndarray, result: dict, return_char_alignments: bool = False):
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=return_char_alignments)
        self.free()
        return result

    def diarization(self, audio: np.ndarray, result: dict):
        diarize_segments = self.diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        self.free()
        return result

    def save_metadata(self, metadata: dict, _id: int) -> None:
        with open(os.path.join(self.download_folder, f"{_id}.json"), "w") as outfile: 
            json.dump(metadata, outfile)

    def load_metadata(self, _id):
        metadata = {}
        with open(os.path.join(self.download_folder, f"{_id}.json"), "r") as file: 
            metadata = json.load(file)
        return metadata

    def free(self) -> None:
        gc.collect(); torch.cuda.empty_cache();

    def run(self, url: str, batch_size:int = 16, language: Union[str, None] = 'es', diarization: bool = False) -> dict:
        _id = self.video_id(url)
        if _id is None:
            raise ValueError("The URL is not a youtube valid link")
        metadata = self.download_process(url)
        #
        files = {
            'vocals': os.path.join(self.download_folder, f"{_id}_vocals.wav"), 
            'base': os.path.join(self.download_folder, f"{_id}_base.wav")
        }
        if not os.path.exists(files['vocals']) or not os.path.exists(files['base']):
            files = self.audio_separation_process(metadata)
        audio, result = self.speech_recognition(files['vocals'], batch_size=batch_size, language=language)
        result_aligned = self.alignt_output(audio, result)
        if diarization:
            init = default_timer()
            result_aligned = self.diarization(audio, result_aligned)
            print("Diarization time ", default_timer()-init)
        self.save_metadata(result_aligned, _id)
        os.makedirs(self.generated_folder, exist_ok=True)
        output_file = f"{_id}.mp4"
        output_file = os.path.join(self.generated_folder, output_file)
        output_mix = self.editor.generate(metadata['video_output'], files['base'], result_aligned, metadata['title'], output_file)
        return output_mix