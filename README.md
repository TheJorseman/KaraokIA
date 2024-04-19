# KaraokIA

KaraokIAis a Python app to create karaoke videos using Deep Learning Models. The demo was built in gradio.

The pipeline is 

* Download Youtube Video using [yt-dlp](https://github.com/yt-dlp/yt-dlp)

* Audio Separation using [demucs](https://github.com/facebookresearch/demucs)

* Speech Recognition using [whisperx](https://github.com/m-bain/whisperX)

* Speaker Diarization (optional) [pyannote](https://huggingface.co/pyannote/speaker-diarization-3.1)

* Video with lyrics

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt
```
## Docker

Use docker. Under construction

```bash
docker 
```



## Usage

```bash
python demo.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[GPLv3](https://choosealicense.com/licenses/gpl-3.0/)