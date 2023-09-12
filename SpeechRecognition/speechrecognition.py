import streamlit as st
import speech_recognition as sr
import keyboard

recognizer = sr.Recognizer()

def transcribe_speech(audio_data, language):
    with sr.AudioFile(audio_data) as source:
        recognizer.adjust_for_ambient_noise(source)
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        try:
            text = recognizer.recognize_sphinx(source, language=language)
            return text
        except sr.UnknownValueError:
            return "Désolé, je n'ai pas compris ce que vous avez dit."
        except sr.RequestError as e:
            return f"Impossible de récupérer les résultats ; {str(e)}"
        except Exception as e:
            return f"Une erreur s'est produite : {str(e)}"

st.title("Reconnaissance vocale en temps réel")

language = st.selectbox("Sélectionnez la langue", ["en-US", "fr-FR"])

with st.spinner("Initialisation du microphone..."):
    with sr.Microphone() as source:
        st.success("Le microphone est prêt. Appuyez sur la touche 's' pour démarrer/arrêter l'enregistrement.")

        recording = False
        audio_data = None

        while True:
            try:
                e = keyboard.read_event()
                if e.event_type == keyboard.KEY_DOWN:
                    if e.name == 's':
                        if not recording:
                            st.info("Enregistrement démarré...")
                            recording = True
                            audio_data = recognizer.listen(source, timeout=None)
                        else:
                            st.info("Enregistrement arrêté...")
                            recording = False
                            keyboard.unhook_all()
                            break
            except Exception as ex:
                st.error(f"Erreur : {str(ex)}")

if audio_data is not None:
    st.subheader("Transcription :")
    text = transcribe_speech(audio_data, language)
    st.write("Vous avez dit : " + text)

if st.button("Enregistrer dans un fichier"):
    if audio_data is not None:
        save_to_file(text, "transcription.txt")
        st.success("Transcription enregistrée dans 'transcription.txt'.")

def save_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

if st.button("Pause"):
    st.write("Enregistrement en pause.")

if st.button("Reprendre"):
    st.write("Enregistrement repris.")
