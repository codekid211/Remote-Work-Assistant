from flask import Flask, render_template, request
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    target_language = request.form['target_language']
    user_text = request.form['user_text']
    
    # Handling file upload
    if 'file_input' in request.files:
        uploaded_file = request.files['file_input']
        if uploaded_file.filename != '':
            text_from_file = uploaded_file.read().decode('utf-8')
            translated_text_from_file = translate_text(text_from_file, target_language)
            tts_output_from_file = tts(translated_text_from_file, target_language)
            return render_template('result.html', original_text=text_from_file, translated_text=translated_text_from_file, audio_file=tts_output_from_file)
    
    # Handling text input
    if user_text:
        text = user_text
    else:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print('Audio analyzed')
            r.adjust_for_ambient_noise(source, duration=5)
            print("Speak")
            audio = r.listen(source)
        
        try:
            text = r.recognize_google(audio)
            print("Text from speech: ", text)
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
            return "Error: Speech recognition could not understand audio"
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return "Error: Could not request results from Google Speech Recognition service"
    
    translated_text = translate_text(text, target_language)
    print(f"\nOriginal Text: {text}")
    print(f"Translated Text: {translated_text}")
    
    tts_output = tts(translated_text, target_language)
    
    return render_template('result.html', original_text=text, translated_text=translated_text, audio_file=tts_output)

def translate_text(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

def tts(text, target_language):
    tts = gTTS(text=text, lang=target_language)
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    audio_data = audio_file.read()
    encoded_audio = base64.b64encode(audio_data).decode('utf-8')
    return f"data:audio/mpeg;base64,{encoded_audio}"

if __name__ == '__main__':
    app.run(debug=True)
