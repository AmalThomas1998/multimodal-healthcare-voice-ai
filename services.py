import os
import base64
from groq import Groq
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
# from elevenlabs.client import ElevenLabs
from playsound import playsound
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='app.log', filemode='w')

class AudioRecorder:
    """
    Records audio from the default microphone and saves as MP3.
    """
    def __init__(self, timeout: int = 20, phrase_time_limit: int = None):
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        

    def record(self, file_path: str):
        """
        Capture audio from mic, export to MP3.

        Args:
            file_path: Path where the MP3 will be saved.
        """
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                logging.info("Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                logging.info("Listening...")
                audio_data = recognizer.listen(source, timeout=self.timeout, phrase_time_limit=self.phrase_time_limit)
                logging.info("Recording complete.")
                wav_data = audio_data.get_wav_data()
                audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
                audio_segment.export(file_path, format="mp3", bitrate="128k")
                logging.info(f"Audio saved to {file_path}")
                playsound(file_path)  # Play the recorded audio
        except Exception as e:
            logging.error(f"Recording failed: {e}")



if __name__ == "__main__":
    # Example usage
    audio_recorder = AudioRecorder(timeout=10, phrase_time_limit=5)
    audio_recorder.record("test_audio.mp3")

    