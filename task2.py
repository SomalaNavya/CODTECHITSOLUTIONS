import speech_recognition as sr

def transcribe_audio_from_mic():
    """
    Transcribes audio captured from the microphone using the SpeechRecognition library.
    """
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Listening... Say something!")
        # Adjust for ambient noise before listening
        recognizer.adjust_for_ambient_noise(source)
        try:
            # Listen to the audio and store it in a variable
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Processing audio...")

            # Use Google's Web Speech API to recognize the audio
            # You can change this to another recognizer if needed
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")

        except sr.WaitTimeoutError:
            print("No speech detected within the time limit.")
        except sr.UnknownValueError:
            print("Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

if _name_ == "_main_":
    transcribe_audio_from_mic()