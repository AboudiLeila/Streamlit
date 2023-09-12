import streamlit as st
from google.cloud import speech_v1p1beta1 as speech
import keyboard
import io
import os

# Set your Google Cloud credentials environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your-credentials.json"

# Create a client for the Google Cloud Speech-to-Text API
client = speech.SpeechClient()

# Function to transcribe speech
def transcribe_speech(audio_data, language_code):
    audio = speech.RecognitionAudio(content=audio_data)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        return result.alternatives[0].transcript

# Streamlit UI
st.title("Real-time Speech Recognition with Google Cloud Speech-to-Text")

# Language selection
language_code = st.selectbox("Select Language", ["en-US", "fr-FR"])

# Open the microphone to capture audio
with st.spinner("Initializing microphone..."):
    st.success("Microphone is ready. Press the 's' key to start/stop recording.")

    recording = False
    audio_data = io.BytesIO()

    while True:
        try:
            e = keyboard.read_event()
            if e.event_type == keyboard.KEY_DOWN:
                if e.name == 's':
                    if not recording:
                        st.info("Recording started...")
                        recording = True
                    else:
                        st.info("Recording stopped...")
                        recording = False
                        keyboard.unhook_all()
                        break
            elif e.event_type == keyboard.KEY_UP and recording:
                audio_data.write(e.scan_code.to_bytes(1, byteorder='big'))

        except Exception as ex:
            st.error(f"Error: {str(ex)}")

# Perform the transcription
if audio_data.getvalue():
    st.subheader("Transcription:")
    text = transcribe_speech(audio_data.getvalue(), language_code)
    st.write("You said: " + text)
