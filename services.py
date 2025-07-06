import os
import base64
from groq import Groq
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
# from elevenlabs.client import ElevenLabs
from playsound import playsound
import logging
from dotenv import load_dotenv
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', mode='w'),
        logging.StreamHandler()
    ]
)

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

class STTService:
    """
    Speech-to-Text service using Groq Whisper API.
    """
    def __init__(self, api_key: str, model: str = "whisper-large-v3"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to the audio file.
        Returns:
            The transcription text.
        """
        try:
            logging.info(f"Starting transcription for {audio_path} using model {self.model}.")
            with open(audio_path, "rb") as f:
                transcription = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=f,
                    language="en"
                )
            logging.info("Transcription successful.")
            return transcription.text
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return ""

class VisionService:
    """
    Vision-enabled LLM service for image analysis using Groq API.
    """
    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def analyze(self, prompt: str, image_path: str) -> str:
        """
        Analyze an image along with a text prompt.

        Args:
            prompt: The textual instruction or question.
            image_path: Path to the image file.
        Returns:
            The model's textual response.
        """
        try:
            logging.info(f"Starting image analysis for {image_path} with prompt: {prompt}")
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ]
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model
            )
            logging.info("Image analysis successful.")            
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Image analysis failed: {e}")
            return ""
        
class TTSService:
    """
    Text-to-Speech service with optional ElevenLabs engine.

    Args:
        elevenlabs_api_key: API key for ElevenLabs. If provided, the ElevenLabs engine can be used for TTS.
    """
    def __init__(self, elevenlabs_api_key: str = None):
        self.elevenlabs_api_key = elevenlabs_api_key

    def speak(self, text: str, outfile: str, engine: str = "gtts") -> str:
        """
        Convert text to speech and save/play it.

        Args:
            text: The text to speak.
            outfile: Output file path for the audio.
            engine: Which TTS engine to use ('gtts' or 'eleven').
        Returns:
            The path to the output audio file.
        """
        try:
            logging.info(f"Starting TTS with engine '{engine}' for text: {text}")
            
            if engine == "eleven" and self.elevenlabs_api_key:
               
                client = ElevenLabs(api_key=self.elevenlabs_api_key)
                audio = client.generate(
                    text=text,
                    voice="Arnold",
                    output_format="mp3_22050_32",
                    model="eleven_turbo_v2"
                )
                elevenlabs.save(audio, outfile)
                logging.info(f"Audio generated and saved to {outfile} using ElevenLabs.")
            else:
                tts = gTTS(text=text, lang="en", slow=False)
                tts.save(outfile)
                logging.info(f"Audio generated and saved to {outfile} using gTTS.")

            playsound(outfile)
            logging.info(f"Audio playback completed for {outfile}.")
            return outfile
        except Exception as e:
            logging.error(f"TTS failed: {e}")
            return ""

class DoctorBot:
    def __init__(self, stt: STTService, vision: VisionService, tts: TTSService):
        self.stt = stt
        self.vision = vision
        self.tts = tts

    def handle(self, audio_fp, image_fp=None):
        text = self.stt.transcribe(audio_fp)
        if image_fp:
            reply = self.vision.analyze(text, image_fp)
        else:
            reply = "No image provided."
        self.tts.speak(reply, "final.mp3")
        return text, reply, "final.mp3"

if __name__ == "__main__":
    # Example usage  

    # recorder = AudioRecorder(timeout=20, phrase_time_limit=10)
    # print("Recording audio for 20 seconds...")
    # recorder.record("test_audio.mp3")
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    
    stt_service = STTService(api_key=groq_api_key)
    # transcription = stt_service.transcribe(r"C:\Users\athul\OneDrive\Desktop\Medical Chatbot 2.0\patient_voice.mp3")
    # print("Transcription:", transcription)

    vision_service = VisionService(api_key=groq_api_key)
    # image_analysis = vision_service.analyze("Analyze this image", "test_image.jpg")

    tts_service = TTSService(elevenlabs_api_key=elevenlabs_api_key)
    # # tts_service.speak("Hello, this is a test.", "output.mp3", engine="eleven")
    # tts_service.speak("Hello, this is a test.", "output.mp3", engine="gtts")

    doctor_bot = DoctorBot(stt=stt_service, vision=vision_service, tts=tts_service)
    text, reply, audio_fp = doctor_bot.handle("test.mp3", "test_image.jpg")
    print("Transcription:", text)
    print("Reply:", reply)  
    print("Audio file path:", audio_fp)
    