from faster_whisper import WhisperModel
from easygoogletranslate import EasyGoogleTranslate
from pydub import AudioSegment

import torch
import torchaudio

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from pathlib import Path

from modeldownloader import download_model

import ffmpeg
import time
import os
import argparse

def local_generation(xtts,text,speaker_wav,language,output_file):
        # Log time
        generate_start_time = time.time()  # Record the start time of loading the model

        gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(speaker_wav)

        out = xtts.inference(
            text,
            language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            # temperature=0.85,
            # length_penalty=1.0,
            # repetition_penalty=7.0,
            # top_k=50,
            # top_p=0.85,
            enable_text_splitting=False
        )

        torchaudio.save(output_file, torch.tensor(out["wav"]).unsqueeze(0), 24000)

        generate_end_time = time.time()  # Record the time to generate TTS
        generate_elapsed_time = generate_end_time - generate_start_time

        print(f"TTS generated in {generate_elapsed_time:.2f} seconds")

def combine_wav_files(file_list, output_filename):
    combined = AudioSegment.empty()

    for file in file_list:
        sound = AudioSegment.from_wav(file)
        combined += sound

    combined.export(output_filename, format='wav')

def removeTempFiles(file_list):
    for file in file_list:
        os.remove(file)

def segment_audio(start_time, end_time, input_file, output_file):
    # Cut the segment using ffmpeg and convert it to required format and specifications
    (
        ffmpeg
        .input(input_file)
        .output(output_file,
                format='wav',
                acodec='pcm_s16le',
                ac=1,
                ar='22050',
                ss=start_time,
                to=end_time)
        .run(capture_stdout=True, capture_stderr=True)
    )


def get_suitable_segment(i, segments):
    min_duration = 5
    if i == 0 or (segments[i].end - segments[i].start) >= min_duration:
        print(f"Segment used N-{i}")
        return segments[i]
    else:
        # Iteration through previous segments starting from the current segment
        for prev_i in range(i - 1, -1, -1):
            if (segments[prev_i].end - segments[prev_i].start) >= min_duration:
                print(f"Segment used N-{prev_i}")
                return segments[prev_i]
        # If no suitable previous one is found,
        # return the current one regardless of its duration.
        return segments[i]



def clean_temporary_files(file_list,directory):
    """
    Remove temporary files from the specified directory
    """
    for temp_file in file_list:
        try:
            temp_filepath=os.path.join(directory,temp_file)
            os.remove(temp_filepath)
        except FileNotFoundError:
            pass # If file was already removed or doesn't exist


def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it does not exist
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Folder '{directory_path}' has been successfully established.")
    except OSError as e:
        print(f"An error occurred while creating folder '{directory_path}': {e}")

# Assuming translator, segment_audio, get_suitable_segment,
# local_generation, combine_wav_files are defined elsewhere.

def translate_and_get_voice(filename,xtts,segments, mode, source_lang, target_lang,
                            speaker_lang, output_folder="out",
                            output_filename="result.mp3", speaker_wavs=None):

    create_directory_if_not_exists(output_folder)
    translator = EasyGoogleTranslate()

    original_segment_files = []
    translated_segment_files = []

    for i, segment in enumerate(segments):
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        translated_text = translator.translate(segment.text,target_lang,source_lang)

        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {translated_text}")

        original_audio_segment_file = f"original_segment_{i}.wav"
        original_segment_files.append(original_audio_segment_file)

        # Segment audio according to mode
        if mode == 1 or mode == 2:
            start_time = segment.start if mode == 1 else get_suitable_segment(i, segments).start
            end_time = segment.end if mode == 1 else get_suitable_segment(i, segments).end

            # Exporting segmented audio; assuming 'filename' is available here
            segment_audio(start_time,end_time,filename,output_folder + "/" + original_audio_segment_file)

        # Prepare TTS input based on provided `mode`
        tts_input_wav = speaker_wavs if (mode == 3 and speaker_wavs) else output_folder + "/" + original_audio_segment_file

        synthesized_audio_file = f"synthesized_segment_{i}.wav"

        # Start transformation using TTS system; assuming all required fields are correct
        local_generation(
            xtts = xtts,
            text=translated_text,
            speaker_wav=tts_input_wav,
            language=speaker_lang,
            output_file=output_folder + "/" + synthesized_audio_file
        )

        translated_segment_files.append(synthesized_audio_file)

    combined_output_filepath = os.path.join(output_folder,output_filename)
    combine_wav_files([os.path.join(output_folder,f) for f in translated_segment_files], combined_output_filepath)
    
    # Optionally remove temporary files after combining them into final output.
    clean_temporary_files(translated_segment_files + (original_segment_files if mode != 3 else []), output_folder)

# source_lang = info.language
# target_lang = "en"
# speaker_lang = "ru"

# segments = list(segments)
# translate_and_get_voice(segments,1,source_lang,target_lang,speaker_lang,"output","result.mp3")

def main():
    parser = argparse.ArgumentParser(description='Translate audio segments and synthesize speech.')

    parser.add_argument('--input',"-i", required=True, help='Path to the file containing segment data')
    parser.add_argument('--whsper_model',"-wm", default='large-v3', help='Model for Whisper')
    parser.add_argument('--xtts_version',"-xv", default='2.0.2', help='Version of XTTS')
    parser.add_argument('--mode',"-m", type=int, choices=[1, 2, 3], default=1,
                        help='Mode of operation for audio processing')
    parser.add_argument('--source_lang',"-sol", default='auto', help='Source language code')
    parser.add_argument('--target_lang',"-tl", default='en', help='Target language code')
    parser.add_argument('--speaker_lang',"-spl", default='ru', help='Speaker language code for synthesis')
    parser.add_argument('--output_folder',"-ofo", default="output", help='Output folder path')
    parser.add_argument('--output_filename',"-ofi", default="result.mp3", help='Name for the output audio file')
    parser.add_argument('--speaker_wav_path',"-spw",
                        type=lambda p: Path(p).absolute(),
                        required=False,
                        help="Path to original speaker's WAV file")

    args = parser.parse_args()

    #  LOAD WHISPER MODEL
    model_size = args.whsper_model
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    filename = args.input # Make sure this is a string
    
    # Load XTTS model
    xtts_model_version = args.xtts_version
    
    this_dir = Path(__file__).parent.resolve()
    
    download_model(this_dir,xtts_model_version)
    
    config = XttsConfig()
    config_path = this_dir / 'models' / f'v{xtts_model_version}' / 'config.json'
    checkpoint_dir = this_dir / 'models' / f'v{xtts_model_version}'
    
    config.load_json(str(config_path))
            
    xtts = Xtts.init_from_config(config)
    xtts.load_checkpoint(config, checkpoint_dir=str(checkpoint_dir))
    xtts.to("cuda")
    
    # Transcribe the audio file using WhisperModel
    segments, info = model.transcribe(filename, beam_size=5)
    
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    translate_and_get_voice(filename =filename,
                            xtts = xtts,
                            segments=segments,
                            mode=args.mode,
                            source_lang=args.source_lang,
                            target_lang=args.target_lang,
                            speaker_lang=args.speaker_lang,
                            output_folder=args.output_folder,
                            output_filename=args.output_filename,
                            speaker_wavs=str(args.speaker_wav_path))

if __name__ == "__main__":
   main()
   print("Done")


