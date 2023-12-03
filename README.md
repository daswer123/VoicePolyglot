# VoicePolyglot
The VoicePolyglot Project harnesses the power of Whisper, Google Translate, and XTTS to bring a novel idea to life: Speaking with your voice in another language.

## DEMO (Turn sound on)

https://github.com/daswer123/VoicePolyglot/assets/22278673/40692374-0f6d-4c37-8c86-5536ffe22101

## Overview
The VoicePolyglot Project harnesses the power of Whisper, Google Translate, and XTTS to bring a novel idea to life: Speaking with your voice in another language.
For that we taking a voice recording, transcribing it into text, breaking it down into sentences, translating those sentences into a specified language, and finally using XTTS technology to vocalize the content in the new language.
This project offers three unique modes of operation that cater to different requirements for voice transformation fidelity and emotional preservation.

#### Mode 1: Sentence-by-Sentence Transformation
In this mode, each sentence from the original recording is independently processed. This approach ensures that the original intonation and emotion of every sentence are captured and reflected in the transformed voice output. However, one potential downside is that shorter sentences might result in compromised voice cloning quality due to insufficient audio data.

#### Mode 2: Selective Sentence Integration
The second mode addresses the issue of short sentence length by selecting only those audio segments that meet a minimum duration criterion while also being as close as possible to the current segment under consideration. This method balances between preserving emotional content and maintaining high-quality voice reproduction.

#### Mode 3: Pre-recorded Voice Source Utilization
Mode three deviates from processing individual inputs and instead relies on pre-recorded voice sources. This method offers consistency across translations by using a standard set of high-quality voice recordings as a foundation for transformation.

## Install
This project uses the CUDA version and runs on a graphics card

To install and try follow these steps:
1) `git clone https://github.com/daswer123/VoicePolyglot`
2) `cd VoicePolyglot`
3) `python -m venv venv`
4) `venv/scripts/activate`
5) `pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118` 
6) `pip install -r requerments.txt`

## How to use
Use the following command structure to provide input parameters for processing your audio file:

```bash
python main.py --input <path_to_audio_file> --target_lang <lang_code> --speaker_lang <lang_code> [options]
```

#### Command-Line Arguments

- `--input`, `-i`: **(Required)** Path to the file containing segment data for processing.

- `--whisper_model`, `-wm`: Model specification for Whisper (default: `'large-v3'`).

- `--xtts_version`, `-xv`: Specifies the version of XTTS to be used (default: `'2.0.2'`).

- `--mode`, `-m`: Selects mode of operation for audio processing with choices `[1, 2, 3]` (default: `1`).

- `--source_lang`, `-sol`: Source language code (default: `'auto'`). The program will attempt automatic language detection if set to 'auto'.

- `--target_lang`, `-tl`: Target language code into which sentences will be translated (default: `'en'` for English).

- `--speaker_lang`, `-spl`: Speaker language code used for synthesis matching speaker's original language nuances (default: `'ru'`).

- `--output_folder`, `-ofo`: Output folder path where processed files will be saved (default: `"output"`).

- `--output_filename`, `-ofi`: Name for saving output audio file after processing is complete (default: `"result.mp3"`).

- `--speaker_wav_path`, `-spw`: Optional path to original speaker’s WAV file which can be provided when using Mode 3.

Example command:
```bash
python main.py --input "./data/segment.wav" --mode 2 --target_lang "en" --speaker_lang "en"
```
