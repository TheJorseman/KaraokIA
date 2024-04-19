import gradio as gr
from Karaokia.app import KaraokIA
from dotenv import load_dotenv
import os
import json
load_dotenv()

HF_TOKEN = os.environ.get('HF_TOKEN')

languages = ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "'no'", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue"]
choices = [None] + languages

app = KaraokIA(HF_TOKEN=HF_TOKEN)


def generate_karaoke(yt_url, language, batch_size, diarization):
    video = app.run(yt_url, language=language, batch_size=batch_size, diarization=diarization)
    return video

demo = gr.Interface(
    fn=generate_karaoke, 
    inputs=[
            gr.Text(label="URL Video"), 
            gr.Dropdown(label="Language", choices=choices, value="es"),
            gr.Slider(label="batch_size", minimum=1, maximum=16),
            gr.Checkbox(label="Diarization", info="Warning! This option is too GPU expensive, takes over 500 seconds"),
            #gr.Checkbox(label="Upload to YT"),
            ],
    outputs=["playable_video"],
    title="KaraokIA",
    description="A AI tool to Generate Karaoke Videos",
    article="",
    flagging_dir="./downloads",
    flagging_options=["Upload to yt"],
    #flagging_callback=
    )

if __name__ == "__main__":
    demo.launch()
