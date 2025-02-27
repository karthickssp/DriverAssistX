import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Get available voices
voices = engine.getProperty('voices')

def set_voice(voice_index=0, text="Hello! This is the selected voice.", rate=140, volume=1.0):
    """Sets the voice, speech rate, and volume, then speaks the given text."""
    if 0 <= voice_index < len(voices):
        selected_voice = voices[voice_index]
        engine.setProperty('voice', selected_voice.id)
        print(f"Selected Voice: {selected_voice.name} | Language: {selected_voice.languages}")
    else:
        print("Invalid voice index! Defaulting to voice 0.")
        selected_voice = voices[0]
        engine.setProperty('voice', selected_voice.id)

    # Set rate and volume
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)

    # Speak the text
    engine.say(text)
    engine.runAndWait()

# Example Usage:
set_voice(1, text="Testing this voice! Sets the voice, speech rate, and volume, then speaks the given text.")  # Change index for different voices
