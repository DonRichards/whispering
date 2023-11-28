FROM python:3.11-slim
# docker pull python:3.11-slim
# docker build --no-cache -t transcribe_with_whisper .
# docker run -it --rm -v /Users/donr/github/whisper:/transcribe transcribe_with_whisper python transcripts.py --num_speakers 2 --language English --model_size medium

# Set the maintainer label
LABEL maintainer="donrichards@jhu.edu"

# Set environment variables to ensure that output is sent to the terminal without buffering
ENV PYTHONUNBUFFERED 1

# Update and install some basic utilities
RUN apt-get update && apt-get install -y --fix-missing git libsndfile1 ffmpeg

# Install whisper and pyannote packages. MOVED TO requirements.txt
# RUN pip install -q git+https://github.com/pyannote/pyannote-audio@3.0.0
# RUN pip install -q git+https://github.com/openai/whisper.git

# Create a directory for the app
WORKDIR /transcribe

# Copy the current directory contents into the container at /whisper
COPY . /transcribe

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run transcribe.py when the container launches
# CMD ["python", "./transcripts.py"]