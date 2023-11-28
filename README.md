# Whispering

```
python transcripts.py --num_speakers 2 --language English --model_size medium
```

Here's a breakdown of its main functionalities:

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

Overall, the script automates the process of transcribing multi-speaker audio files, identifying different speakers, and outputting a formatted transcript. The use of libraries like whisper, pyannote.audio, and pydub indicates that it's designed to handle complex audio processing tasks, including transcription, speaker identification, and audio format conversion.

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
- large

