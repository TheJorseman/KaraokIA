from PIL import Image, ImageFilter, ImageFont, ImageDraw
import textwrap
import imageio.v3 as iio
import json
import math
import numpy as np
from tqdm import tqdm
from timeit import default_timer
from typing import Union
from unicodedata import normalize
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
class Editor():

    def __init__(
            self,
            gaussian_kernel: int = 13,
            height_perc: int = 10,
            width_perc: int = 7,
            height: int = 720, 
            width: int = 1280,
            background_color: tuple = (0,0,0),

        ) -> None:
        self.gaussian_kernel = gaussian_kernel
        self.height_perc = height_perc
        self.width_perc = width_perc
        self.height = height
        self.width = width
        self.background_color = background_color


    def blur_frame(self, img: Image) -> Image:
        return img.filter(ImageFilter.GaussianBlur(self.gaussian_kernel))
    
    def get_init_video_cords(self, img, height_perc=10, width_perc=7) -> tuple[int, int]:
        return int(img.width * (width_perc/100)), int(img.height * (height_perc/100))

    def pad_image(self, img: Image) -> Image:
        x = (self.width//2) - (img.width//2)
        y = (self.height//2) - (img.height//2)
        canva = Image.new(img.mode, (self.width, self.height), self.background_color) 
        canva.paste(img, (x, y))
        return canva

    def relu_max(self, value, max=80):
        return max if value >= max else value

    def put_text(
            self,
            img: Image, 
            text: Union[str, list], 
            color_default= (255, 255, 255),
            words_resalt = [],
            color_resalt = (255, 255, 255),
            chars_per_line=32,
            size=80, 
            space_pad=20, 
            pad=50, 
            drop_whitespace=False
        ) -> None:
        # ty https://stackoverflow.com/questions/63436667/justify-text-using-pillow-python


        num_lines = self.relu_max(round(len(text) / 3), 40)
        paragraph = text
        if isinstance(text, str):
            paragraph = textwrap.wrap(text, width=num_lines, drop_whitespace=drop_whitespace)
        
        size = self.relu_max(round((3 / len(paragraph))*80))
        
        pad = round(size * 0.625)
        _, img_width = img.height, img.width
        draw = ImageDraw.Draw(img)
        #font = ImageFont.load_default(size=size)
        font = ImageFont.truetype("arial.ttf", size=size) 
        cords = self.get_init_video_cords(img)
        current_h = cords[1]

        count_word = 0
        for line in paragraph:
            _, _, w, h = draw.textbbox((0, 0), line, font=font)
            cords_init = ((img_width - w) // 2, current_h)
            words = line.strip().split(' ')
            space = space_pad
            width_sum = 0
            for word in words:
                _, _, w_w, _ = draw.textbbox((0, 0), word, font=font)
                word_cords = (cords_init[0] + width_sum, cords_init[1])
                color = color_default
                if count_word in words_resalt:
                    color = color_resalt
                #word = normalize('NFKD', word).encode('ascii','ignore').decode()
                draw.text(word_cords, word, font=font, fill=color)
                width_sum += w_w + space
                count_word +=1
            current_h += h + pad
        

    def get_num_speakers(self, data: dict) -> int:
        speakers = []
        for segment in data['segments']:
            speaker = segment.get('speaker', False)
            if not speaker:
                return 1
            if speaker not in speakers:
                speakers.append(speaker)
        return len(speakers)


    def get_speakers_colors(self, num_speakers: int) -> list:
        colors = [
            (255, 255, 0),   # Amarillo
            (255, 0, 0),     # Rojo
            (0, 255, 0),     # Verde
            (0, 0, 255),     # Azul
            (128, 0, 128),   # Violeta
            (255, 165, 0),   # Naranja
            (0, 255, 255),   # Cian
            (255, 0, 255),   # Magenta
            (128, 128, 128), # Gris
            (0, 0, 0)        # Negro
        ]

        return colors[:num_speakers]


    def calculate_frames(self, segment: dict, fps: float) -> tuple[int, int]:
        init_frame = segment['start'] * fps
        end_frame = segment['end'] * fps

        start = round(init_frame)
        end = round(end_frame)
        if start == end:
            end += 1
        return start, end

    def get_speaker_id(self, segment: dict) -> int:
        speaker_str = segment.get('speaker', 'SPEAKER_00')
        _, number = speaker_str.split('_')
        return int(number)
    
    def get_next_state(self, stack: list, fps: float) -> tuple[dict, dict, tuple[int, int], tuple[int, int]]:
        if len(stack) <= 0:
            return {}, {}, (math.inf, math.inf), (math.inf, math.inf)
        current_segment = stack.pop(0)
        #words = current_segment['words'].copy()
        current_word = current_segment['words'][0]
        init_frame, end_frame = self.calculate_frames(current_segment, fps)
        init_frame_word , end_frame_word = self.calculate_frames(current_word, fps)
        return current_segment, current_word, (init_frame, end_frame), (init_frame_word , end_frame_word)

    def preprocess_frame(self, frame: np.ndarray) -> Image:
        img = Image.fromarray(frame).convert("RGB")
        if self.height !=frame.shape[0] or self.width != frame.shape[1]:
            print("padding")
            img = self.pad_image(img)
        img = self.blur_frame(img)
        return img

    def wrap_to_line(wrap):
        output = {}
        i = 0
        for n, line in enumerate(wrap):
            for _ in line.strip().split(' '):
                output[i] = n
                i += 1
        return output

    def mix_video(self, audio:str, output_name: str, output_mix: str):
        os.system(f"ffmpeg -y -i {audio} -r 30 -i {output_name} -filter:a aresample=async=1 -c:a mp3 -c:v copy {output_mix}")

    def generate(self, source: str, audio_base:str, asr_data: str, title:str, output_file: str, text_color: tuple = (255, 255, 255)):
        num_speakers = self.get_num_speakers(asr_data)
        colors = self.get_speakers_colors(num_speakers)
        metadata = iio.immeta(source, plugin="pyav")
        #metadata['fps'] = 30
        watermark = "Generated by KaraokIA"
        text_intro = title + "\n" + watermark

        intro = {
            'start': 0.0,
            'end': min([3.0, asr_data['segments'][0].get('start', 3.0)]),
            'text': text_intro,
            "words": [{"word": text_intro, "start": 0.0, "end": 3.0, "score": 1.0}],
            "is_intro": True
        }

        stack = [intro] + [segment for segment in asr_data['segments']]

        current_segment, current_word, frame_range, word_range = self.get_next_state(stack, metadata['fps'])

        total = round(metadata["fps"] * metadata["duration"])

        word_poped_id = 0

        with iio.imopen(output_file, "w", plugin="pyav") as out_file:
            
            out_file.init_video_stream("vp9", fps=round(metadata["fps"]))
            
            for i, frame in tqdm(enumerate(iio.imiter(source, plugin="pyav")), total=total):
                # Preprocessing for all frames
                img = self.preprocess_frame(frame)
                # Text Base
                if i >= frame_range[0] and i < frame_range[1]:
                    speaker_id = self.get_speaker_id(current_word)
                    resalt = []
                    if not current_segment.get('is_intro', False):
                        resalt = [i for i in range(word_poped_id + 1)]
                    self.put_text(img, current_segment['text'].strip(), color_default=text_color, words_resalt=resalt, color_resalt=colors[speaker_id])
                    if i == frame_range[1] - 1:
                        current_segment, current_word, frame_range, word_range = self.get_next_state(stack, metadata['fps'])
                        word_poped_id = 0
                # Word Level
                if i >= word_range[0] and i < word_range[1] and not current_segment.get('is_intro', False):
                    speaker_id = self.get_speaker_id(current_word)
                    resalt = [i for i in range(word_poped_id + 1)]
                    self.put_text(img, current_segment['text'].strip(),  words_resalt = resalt, color_resalt = colors[speaker_id], color_default=text_color)
                    if i == word_range[1] - 1:
                        current_word = current_segment['words'][word_poped_id + 1]
                        if not current_word.get('start', False) or not current_word.get('end', False):
                            if word_poped_id + 2 > len(current_segment['words']):
                                word_range = (word_range[0], word_range[1] + 1)
                            else:
                                for n_word_i in range(word_poped_id + 2, len(current_segment['words'])):
                                    if current_segment['words'][n_word_i].get('start', False) and current_segment['words'][n_word_i].get('end', False):
                                        temp_word_range = self.calculate_frames(current_segment['words'][n_word_i], metadata['fps'])
                                        word_range = (word_range[1], temp_word_range[0])
                                        break
                        else:
                            word_range= self.calculate_frames(current_segment['words'][word_poped_id + 1], metadata['fps'])
                                    #self.calculate_frames(current_segment['words'][word_poped_id], metadata['fps'])
                        word_poped_id += 1
                #print("Frame ", i)
                #print("word_poped_id ", word_poped_id)
                out_file.write_frame(np.array(img))

        root, ext = os.path.splitext(output_file)
        output_mix = f"{root}_final{ext}"
        self.mix_video(audio_base, output_file, output_mix)
        return output_mix