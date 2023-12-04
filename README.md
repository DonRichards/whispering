# Whispering
This repository contains a Python script designed for transcribing multi-speaker audio files. It leverages advanced machine learning models and audio processing techniques to accurately identify and transcribe speech from multiple speakers in an audio file.

```
python transcripts.py --num_speakers 2 --language English --model_size medium
```


## Key Features

- **Automated Speaker Identification**: Utilizes machine learning models to distinguish between different speakers in an audio file.
- **Audio File Conversion**: Converts MP3 files to WAV format, ensuring compatibility with the transcription models.
- **Customizable Transcription Settings**: Allows users to specify the number of speakers, language, and model size through command-line arguments.
- **Speaker-Tagged Transcription**: Outputs a transcript with each speaker's speech clearly tagged and timestamped.
- **Efficient Model Management**: Manages Whisper model files for efficient caching and reuse.

## Technologies Used

- **[Whisper](https://github.com/openai/whisper)**: For transcribing the audio content.
- **[Pyannote.audio](https://github.com/pyannote/pyannote-audio)**: For speaker verification and audio processing.
- **[Pydub](https://github.com/jiaaro/pydub)**: For audio file format conversion.
- **[Sklearn](https://github.com/scikit-learn/scikit-learn)**: For clustering speaker embeddings to identify distinct speakers.
- **[Argparse](https://docs.python.org/3/library/argparse.html)**: For parsing command-line arguments.

## Usage

The script is executed via the command line, where users can specify the number of speakers, the language of the audio, and the size of the Whisper model to be used. It automatically searches for an MP3 file in the current directory, processes it, and outputs a transcript in a text file.

## Getting Started

To use this script, clone the repository, install the required dependencies, and run the script with the appropriate command-line arguments. Detailed instructions are provided in the repository.

## Contribution
Contributions to enhance the script, fix bugs, or improve documentation are welcome. Please refer to the contribution guidelines for more information.

## A more deatiled breakdown of its main functionalities:
1. __Ignoring Deprecated Warnings__: The script starts by attempting to suppress deprecated warnings using the warnings module. This is to avoid cluttering the output with warnings that are not critical to the operation of the script.
1. __Argument Parsing__: It uses argparse to parse command-line arguments. The script expects three arguments: the number of speakers (--num_speakers), the language of the audio (--language), and the size of the Whisper model to use (--model_size).
1. __Loading the Speaker Embedding Model__: The script loads a pretrained speaker embedding model from pyannote.audio, which is used for speaker verification tasks.
1. __Finding and Processing the Audio File__: The script searches for an MP3 file in the current directory, converts it to WAV format if necessary (using pydub), and handles stereo to mono conversion.
1. __Handling Whisper Model Files__: It checks for the existence of the Whisper model files in specific directories and copies them if necessary. This is part of managing the Whisper model's cache.
1. __Transcribing Audio__: The script uses the Whisper model (from the whisper library) to transcribe the audio file. The transcription includes time-stamped segments of the audio.
1. __Processing Audio Segments for Speaker Identification__: For each segment in the transcription, it extracts audio embeddings using the pyannote.audio processor and the speaker embedding model. These embeddings represent the characteristics of the speaker's voice in each segment.
1. __Speaker Clustering__: The script uses AgglomerativeClustering from sklearn.cluster to cluster these embeddings into groups corresponding to different speakers. The number of clusters is set to the number of speakers specified in the command-line arguments.
1. __Writing the Transcript__: It writes the transcribed text to a file, transcript.txt, annotating each segment with the identified speaker and the time of the segment.
1. __Cleanup and Output__: Finally, the script cleans up any temporary files created during its execution and prints the contents of the transcript file to the console.

Overall, the script automates the process of transcribing multi-speaker audio files, identifying different speakers, and outputting a formatted transcript. The use of libraries like [whisper](https://github.com/openai/whisper), pyannote.audio, and pydub indicates that it's designed to handle complex audio processing tasks, including transcription, speaker identification, and audio format conversion.

## How to use this Script
### num_speakers
Number opf speakers

### language
Specify the language of the audio. This is used to determine the alphabet used for the language. Currently supported languages are:

This stores the models locally so that next time you run the script it will be faster.

### Models
The bigger the more accurate.
- tiny
- base
- small
- medium
- large (3.09G)
- large-v1
- large-v2 

## TODO
- [] Get Dockerfile working