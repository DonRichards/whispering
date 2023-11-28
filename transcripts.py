#!/bin/env python3
# python transcripts.py --num_speakers 2 --language English --model_size medium

# Ignore deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import shutil
import whisper
import datetime
import subprocess
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from pydub import AudioSegment
import os
import time


audio_processor = Audio()

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# Then, use logging.debug(), logging.info(), logging.warning(), etc., to log messages.
logging.debug('This message will help diagnose the issue.')

# Argument parsing
print("Setting up argument parser...")
parser = argparse.ArgumentParser(description='Transcribe and process audio files.')
parser.add_argument('--num_speakers', type=int, required=True, help='Number of speakers in the audio file')
parser.add_argument('--language', type=str, required=True, choices=['English', 'any'], help='Language of the transcription')
parser.add_argument('--model_size', type=str, required=True, choices=['tiny', 'base', 'small', 'medium', 'large'], help='Size of the Whisper model to use')

args = parser.parse_args()

num_speakers = args.num_speakers
language = args.language
model_size = args.model_size

print(f"Arguments received: num_speakers={num_speakers}, language={language}, model_size={model_size}")

embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu"))
print("Loaded embedding model.")

audio = Audio()

# Find the first .mp3 file in the current directory
print("Searching for MP3 files...")
for filename in os.listdir('.'):
    if filename.endswith('.mp3'):
        path = filename
        print(f"Found MP3 file: {filename}")
        break

if not path:
    raise Exception("No MP3 file found in the current directory.")

# Track the original filename
original_path = path

if path[-3:] != 'wav':
    print(f"Converting {path} to WAV format...")
    audio = AudioSegment.from_file(path)
    # Convert stereo to mono if the audio has 2 channels
    if audio.channels > 1:
        audio = audio.set_channels(1)
    audio.export('mono_audio.wav', format='wav')
    path = 'mono_audio.wav'

# Copy Whisper model to a desired location
whisper_model_cache_pt_file = os.path.expanduser(f"/root/.cache/whisper/{model_size}.pt")
destination_pt_file = f"/transcribe/models/{model_size}.pt"  # Corrected the use of f-string

# If the model cache file exists and the destination file doesn't, copy the model to the destination
if os.path.exists(whisper_model_cache_pt_file) and not os.path.exists(destination_pt_file):
    shutil.copyfile(whisper_model_cache_pt_file, destination_pt_file)
    print(f"Copied Whisper model from {whisper_model_cache_pt_file} to {destination_pt_file}")
elif not os.path.exists(whisper_model_cache_pt_file) and os.path.exists(destination_pt_file):
    # Make sure the target directory exists
    os.makedirs(os.path.dirname(whisper_model_cache_pt_file), exist_ok=True)
    shutil.copyfile(destination_pt_file, whisper_model_cache_pt_file)
    print(f"Copied Whisper model from {destination_pt_file} to {whisper_model_cache_pt_file}")
else:
    print(f"Either both or neither file paths contain the Whisper model; no action taken.")

path = os.path.abspath(path)

print("Starting...")
try:
    model = whisper.load_model(model_size)
    result = model.transcribe(path)
except Exception as e:
    logging.error(f"An exception occurred: {e}", exc_info=True)

segments = result["segments"]

with contextlib.closing(wave.open(path,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

print("Processing audio segments...")
def segment_embedding(segment):
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio_processor.crop(path, clip)
    return embedding_model(waveform[None])

embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

print("Clustering speakers...")
clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
    segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

print("Writing transcript to file...")
f = open("transcript.txt", "w")

for (i, segment) in enumerate(segments):
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(segment["start"]))
    if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
        f.write("\n" + segment["speaker"] + ' ' + str(formatted_time(segment["start"])) + '\n')
    f.write(segment["text"][1:] + ' ')
f.close()

# Copy Whisper model to the local location for caching.
if not os.path.exists(destination_pt_file) and os.path.exists(whisper_model_cache_pt_file):
    print(f"Copied Whisper model from {whisper_model_cache_pt_file} to {destination_pt_file}")

# Clean up
if path != original_path:
    print("Cleaning up temporary files...")
    os.remove('mono_audio.wav')  # Remove the temporary mono audio file
    os.rename(original_path, original_path + '.complete')  # Rename the original file with a ".complete" extension

# Print the transcript
print("Transcription complete. Here's the transcript:")
print(open('transcript.txt').read())
