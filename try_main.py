 
import torch
import queue
import sys
import pyaudio
import numpy as np
import wave
from faster_whisper import WhisperModel
import openai
import time
from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import requests
from rapidfuzz import process, fuzz 
from dotenv import load_dotenv
import json
import aiofiles
import os
load_dotenv()

client = OpenAI(api_key=os.getenv("api_key"))


def extract_keywords(transcribed_text):
    """Extracts relevant keywords from transcribed text using fuzzy matching."""
    if "issues" in json_template and isinstance(json_template["issues"], list) and json_template["issues"]:
        json_keys = [key.lower() for key in json_template["issues"][0].keys()]
        transcribed_text_lower = transcribed_text.lower()

        matched_keywords = []
        for key in json_keys:
            match_score = fuzz.partial_ratio(key, transcribed_text_lower)  # Compare similarity
            if match_score > 70:  # Set a threshold for similarity
                matched_keywords.append(key)

        return ", ".join(matched_keywords) if matched_keywords else "No relevant keyword found."
    else:
        return "No issues found in JSON."

load_dotenv()

# Load Faster-Whisper Model
model = WhisperModel("large-v2", device="cpu")

# Load JSON file synchronously
with open("issue_data.json", "r", encoding="utf-8") as file:
    json_template = json.load(file)


# Audio Streaming Settings
FORMAT = pyaudio.paInt16  
CHANNELS = 1
RATE = 16000
CHUNK = 1024

audio_queue = queue.Queue()
silence_threshold = 3  # Stop if silence for 3 seconds
last_speech_time = time.time()
recorded_frames = []

def callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    recorded_frames.append(in_data)
    return (in_data, pyaudio.paContinue)

def transcribe_audio(audio_path):
    """Transcribes audio using Whisper and processes text with GPT-4."""
    if os.path.exists(audio_path):
        print(f"Processing audio file: {audio_path}")
        segments, _ = model.transcribe(audio_path, beam_size=5, vad_filter=True)
        transcribed_text = " ".join([segment.text for segment in segments])
        print(f"Recognized: {transcribed_text}")

        # Extract similar keywords using fuzzy matching
        extracted_keywords = extract_keywords(transcribed_text)
        print(f"Identified keyword(s): {extracted_keywords}")

        if extracted_keywords != "No relevant keyword found" and extracted_keywords != "No issues found in JSON.":
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are an AI that extracts relevant keywords. Based on the provided reference, return only the most relevant keyword from this list: {extracted_keywords}."},
                    {"role": "user", "content": transcribed_text}
                ]
            )
            print(response.choices[0].message.content.strip())
    else:
        print(f"File not found: {audio_path}")

# Start live audio recording
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                frames_per_buffer=CHUNK, stream_callback=callback)

print("Listening... Speak into the microphone!")

try:
    while True:
        audio_data = audio_queue.get()
        recorded_frames.append(audio_data)
        if time.time() - last_speech_time > silence_threshold:
            print("No speech detected. Stopping...")
            break
except KeyboardInterrupt:
    print("Stopped by user.")

stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded audio to a WAV file
recorded_audio_file = "streamed_audio.wav"
with wave.open(recorded_audio_file, "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(recorded_frames))

print(f"Audio saved as {recorded_audio_file}")

# Transcribe and process recorded audio
transcribe_audio(recorded_audio_file)

# Process pre-recorded audio
pre_recorded_audio = r"D:\\speetch_to_text\\streamedd_audio.wav"
transcribe_audio(pre_recorded_audio)

