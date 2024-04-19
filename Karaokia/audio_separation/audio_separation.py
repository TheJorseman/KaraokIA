from .base import AudioModelBase
from typing import Optional, AnyStr, Any, List, Union, Dict
import warnings
from tqdm import tqdm
import os
import torch as th
import torch
import torchaudio as ta
from demucs.apply import apply_model, BagOfModels
#from demucs.audio import AudioFile, convert_audio, save_audio
from demucs.audio import save_audio as save
from demucs.pretrained import ModelLoadingError, get_model, DEFAULT_MODEL
from demucs.separate import load_track

class AudioSeparation(AudioModelBase):
    """
    Python Interface for Audio Separation (Demucs V4)

    https://github.com/facebookresearch/demucs

    """
    def __init__(
        self,
        model_path: Optional[str] = DEFAULT_MODEL,
        sampling_rate: Optional[int] = 16000, 
        chunk_length: Optional[int] = 5000,
        FP16: Optional[bool] = False,
        torchscript: Optional[bool] = True,
        shifts: Optional[int] = 1,
        overlap: Optional[float] = 0.25, 
        no_split: Optional[bool] = True,
        segment: Optional[int] = 8,
        two_stems: Optional[bool] = False,
        int24: Optional[bool] = False,
        float32: Optional[bool] = False,
        clip_mode: Optional[str] = 'rescale', # clamp
        mp3: Optional[bool] = False,
        mp3_bitrate: Optional[int] = 320,
        jobs: Optional[int] = 0,
        source_names: Optional[str] = ['drums', 'bass', 'other', 'vocals'],
        *args,
        **kwargs
        ) -> None:
        super().__init__(model_path=model_path, sampling_rate=sampling_rate, chunk_length=chunk_length, FP16=FP16, torchscript=torchscript, *args, **kwargs)
        self.shifts = shifts
        self.overlap = overlap
        self.no_split = no_split
        self.segment = segment
        self.two_stems = two_stems
        self.int24 = int24
        self.float32 = float32
        self.clip_mode = clip_mode
        self.mp3 = mp3
        self.mp3_bitrate = mp3_bitrate
        self.jobs = jobs
        self.source_names = source_names
        if self.two_stems:
            self.source_names = ['vocals', 'base']
        self.load_model()
        self.validate_segment(self.segment)
        self.model.cpu()
        self.model.eval()

    def deallocate(self) -> None:
        self.model.cpu()
        self.model.eval()

    def validate_segment(self, segment):
        if segment is not None and segment < 8:
            raise Exception("Segment must greater than 8. ")
        if isinstance(self.model, BagOfModels):
            print(f"Selected model is a bag of {len(self.model.models)} models. "
                "You will see that many progress bars per track.")
            if segment is not None:
                for sub in self.model.models:
                    sub.segment = segment
        else:
            if segment is not None:
                self.model.segment = segment

    def load_model(self):
        try:
            self.model = get_model(name=self.model_path)
        except ModelLoadingError as error:
            raise Exception(error.args[0])

    def read_audio(self, file):
        return load_track(file, self.model.audio_channels, self.model.samplerate)

    def from_file(self, path:str, **kwargs: Any) -> Any:
        self.filename = os.path.basename(path)
        audio = self.read_audio(path)
        return self(audio, **kwargs)

    def preprocess_inputs(self, audio: Union[torch.TensorType, List[torch.TensorType]], *args):
        self.ref = audio.mean(0)
        return (audio - self.ref.mean()) / self.ref.std()

    def forward(self, audio: torch.Tensor) -> List[float]:    
        return apply_model(self.model, audio[None], device=self.device, shifts=self.shifts,
                              split=self.no_split, overlap=self.overlap, progress=True,
                              num_workers=self.jobs)[0]

    def set_stem(self, sources):
        new_sources = []
        sources = list(sources)
        new_sources.append(sources[-1])
        other_stem = torch.zeros_like(sources[-1])
        sources.pop(-1)
        for i in sources:
            other_stem += i
        new_sources.append(other_stem)
        return new_sources

    def save_audio(self, sources: Dict[str, torch.Tensor], folder:str, _format="{_id}_{source_name}.wav", _id:str=""):
        os.makedirs(folder, exist_ok=True)
        kwargs = {
            'samplerate': self.model.samplerate,
            'bitrate': self.mp3_bitrate,
            'clip': self.clip_mode,
            'as_float': self.float32,
            'bits_per_sample': 24 if self.int24 else 16,
        }
        paths = {}
        for name, source in sources.items():
            path = os.path.join(folder, _format.format(_id=_id, source_name=name))
            paths[name] = path
            save(source, path, **kwargs)
        return paths

    def postprocess_output(self, sources, return_dict=True):
        sources = sources * self.ref.std() + self.ref.mean()
        if self.two_stems:
            sources = self.set_stem(sources)
        if not return_dict:
            return sources
        source_dict = {}
        for i, source in enumerate(sources):
            source_dict[self.source_names[i]] = source
        return source_dict

    def __call__(self, audio: torch.Tensor) -> Any:
        audio = self.preprocess_inputs(audio)
        sources = self.forward(audio)
        return self.postprocess_output(sources)

