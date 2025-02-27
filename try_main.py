 
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
# Load Faster-Whisper Model
model = WhisperModel("large-v2", device="cpu")
 
import json
import aiofiles
import os
load_dotenv()

# Load JSON file synchronously
with open("issue_data.json", "r", encoding="utf-8") as file:
    json_template = json.load(file)  # Directly parse JSON from file


 
client = OpenAI(api_key=os.getenv("api_key"))
 
# Audio Streaming Settings
FORMAT = pyaudio.paInt16  
CHANNELS = 1
RATE = 16000
CHUNK = 1024
 
audio_queue = queue.Queue()
silence_threshold = 3  # Stop if silence for 3 seconds
last_speech_time = time.time()
recorded_frames = []  # Buffer to store collected audio
 
# Capture Microphone Audio
def callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    recorded_frames.append(in_data)  # Store collected audio
    return (in_data, pyaudio.paContinue)
 
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                frames_per_buffer=CHUNK, stream_callback=callback)
 
print("Listening... Speak into the microphone!")
 
try:
    while True:
        audio_data = audio_queue.get()
        recorded_frames.append(audio_data)  # Store chunks of audio
 
        # Stop if no speech detected for `silence_threshold` seconds
        if time.time() - last_speech_time > silence_threshold:
            print("No speech detected. Stopping...")
            break
 
except KeyboardInterrupt:
    print("Stopped by user.")
 
# Stop and close stream
stream.stop_stream()
stream.close()
p.terminate()
 
# Save the recorded audio to a WAV file
with wave.open("streamed_audio.wav", "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(recorded_frames))
 
print(f"Audio saved as streamed_audio.wav")
 
# Transcribe the saved file instead of real-time chunks
segments, _ = model.transcribe("streamed_audio.wav", beam_size=5, vad_filter=True)
text = " ".join([segment.text for segment in segments])
 
print(f"Recognized: {text}")
 
# Process text with GPT-4 to extract form values
# if text.strip():
#     response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         # {"role": "system", "content": "Extract relevant fields from the user's speech to automatically fill the form."},
#         # {"role": "user", "content": text}
        
#         # {"role": "system", "content": "You are a helpful AI assistant. Answer the user's question clearly and concisely."},
#         # {"role": "user", "content": text}
        
#          {"role": "system", "content": "You are an expert in analyzing JSON data. Your task is to review the provided JSON structure `{json_template}` and identify relevant keywords based on the transcribed text from audio input. Extract the key fields that match the user's intent and provide structured output."},
#          {"role": "user", "content": text}
        
        
        
#     ]
# )
 
# extracted_values = response.choices[0].message.content
# print(f"Form Data: {extracted_values}")
 
# Process text with GPT-4 to extract form values
if text.strip():
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract only the most relevant keyword from the transcribed text and return it in the format: 'Identified keyword: x'."},
            {"role": "user", "content": text}
        ]
    )

    # Print only the extracted keyword
    print(response.choices[0].message.content.strip())

