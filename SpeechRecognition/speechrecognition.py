import streamlit as st
import speech_recognition as sr
import keyboard

# Initialize the recognizer
recognizer = sr.Recognizer()


# Function to stop recording when a specific key is pressed
def stop_recording(e):
    if e.name == 's':
        st.write("Recording stopped by key press.")
        keyboard.unhook_all()
        return


# Streamlit UI
st.title("Real-time Speech Recognition")

# Open the microphone to capture audio
with st.spinner("Initializing microphone..."):
    with sr.Microphone() as source:
        st.success("Microphone is ready. Press the 's' key to start/stop recording.")

        # Adjust the energy threshold if needed (this value worked for me, but you may need to fine-tune it)
        recognizer.adjust_for_ambient_noise(source)

        recording = False
        audio_data = None

        while True:
            try:
                e = keyboard.read_event()
                if e.event_type == keyboard.KEY_DOWN:
                    if e.name == 's':
                        if not recording:
                            st.info("Recording started...")
                            recording = True
                            audio_data = recognizer.listen(source, timeout=None)
                        else:
                            st.info("Recording stopped...")
                            recording = False
                            keyboard.unhook_all()
                            break
            except Exception as ex:
                st.error(f"Error: {str(ex)}")

# Perform the transcription
if audio_data is not None:
    st.subheader("Transcription:")
    try:
        text = recognizer.recognize_google(audio_data)  # You can choose a different recognizer if needed
        st.write("You said: " + text)

    except sr.UnknownValueError:
        st.warning("Sorry, I couldn't understand what you said.")
    except sr.RequestError as e:
        st.error(f"Could not request results; {str(e)}")

if st.button("Reset"):
    audio_data = None
    recording = False
