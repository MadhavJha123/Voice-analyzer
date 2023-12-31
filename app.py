import streamlit as st
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
from langdetect import detect
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from googletrans import Translator
from collections import Counter
from datetime import datetime
import numpy as np
from sqlmodel import Field, SQLModel
import os
from passlib.hash import pbkdf2_sha256
from pydub import AudioSegment

# Set up database
Base = declarative_base()
engine = create_engine('sqlite:///transcriptions.db')
SQLModel.metadata = Base.metadata
Session = sessionmaker(bind=engine)
session = Session()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String)

# Define the Transcription model
class Transcription(Base):
    __tablename__ = 'transcriptions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey(User.id), nullable=False)
    text = Column(String)
    language = Column(String)
    timestamp = Column(DateTime, default=datetime.now)

SQLModel.metadata.create_all(engine)

#data={"username":"abc"}

def convert_audio_to_wav(audio_file):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file)

    # Ensure the sample width is 2 (16-bit)
    audio = audio.set_sample_width(2)

    # Convert to mono if it's stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Export as WAV
    wav_file_path = audio_file.name.replace(os.path.splitext(audio_file.name)[1], ".wav")
    audio.export(wav_file_path, format="wav")

    return wav_file_path

def audio_to_text_converter():
    st.header("Audio to Text converter")

    # Initialize session state
    if 'user' not in st.session_state:
        show1 = st.empty()
        choice = show1.radio("Select an option", ("Register", "Login"))

        if choice == "Register":
            st.header("Register")
            new_username = st.text_input("Username:")
            
            # data["username"]=new_username
            new_password = st.text_input("Password:", type="password")
            confirm_password = st.text_input("Confirm Password:", type="password")

            if st.button("Register"):
                if new_password == confirm_password:
                    register_user(new_username, new_password)
                    
                else:
                    st.error("Passwords do not match. Please try again.")

        elif choice == "Login":
            show2 = st.empty()
            show2.header("Login")
            show3 = st.empty()
            show4 = st.empty()
            show5 = st.empty()
            username = show3.text_input("Username:")
            password = show4.text_input("Password:", type="password")

            if show5.button("Login"):
                if login_user(username, password):
                    st.session_state.user = {'username': username}
                    show1.empty()
                    show2.empty()
                    show3.empty()
                    show4.empty()
                    show5.empty()
                    dashboard(username)
                    
    else:
        # data['username'] = show.text_input("Enter your name to continue")
        # show.empty()
        user=get_or_create_user(st.session_state.user['username'])
        username= user.username
        dashboard(username)
    

def get_or_create_user(username):
    user = session.query(User).filter(User.username == username).first()
    if user is None:
        user = User(username=username)
        session.add(user)
        session.commit()
    return user

# @st.cache(allow_output_mutation=True)
def get_transcriptions(user_id):
    return session.query(Transcription).filter(Transcription.user_id == user_id).all()


def dashboard(username):
    st.header(f'Welcome {username} 👋')
    useri=session.query(User).filter(User.username==username).first()
    st.sidebar.header("Transcription History:")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg",'aac'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav", start_time=0)

        if st.button("Convert to Text"):
            with st.spinner("Transcribing..."):
                wav_file_path = convert_audio_to_wav(uploaded_file)
                text = convert_audio_to_text(wav_file_path)
            try:
                st.subheader("Transcription:")
                st.write(text)
                os.remove(wav_file_path)

                # Language detection and translation
                detected_language = detect(text)

                if detected_language != 'en':
                    translated_text = translate_text(text)
                    st.subheader("Translated to English:")
                    st.write(translated_text)
                # Save the history in the sidebar
                save_transcription(useri.id, {'text': text, 'language': detected_language})
                display_top_phrases(useri.id,text)

            except:
                pass

    st.header("Try speaking something:")
    recording_button = st.button("Start Recording")
    info=st.empty()
    if recording_button:
        info.warning("Recording... Speak something!")
        try:
            audio_data = record_audio()
            #st.audio(audio_data, format="audio/wav", start_time=0)
            #audio_data=convert_audio_to_wav(audio_data)
        except:
            st.error("couldn't hear")
        info.success("Recording complete!")

        with st.spinner("Transcribing..."):
            #wav_file_path = convert_audio_to_wav(audio_data)
            text = recorded_audio_to_text(audio_data)
        info.empty()
        try:
            st.subheader("Transcription:")
            st.write(text)

            # Language detection and translation
            detected_language = detect(text)

            if detected_language != 'en':
                translated_text = translate_text(text)
                st.subheader("Translated to English:")
                st.write(translated_text)

            # Save the history in the sidebar
            # Display user-specific information and analytics
            save_transcription(useri.id, {'text': text, 'language': detected_language})
            display_top_phrases(useri.id,text)

        except:
            pass
    
    history = get_transcriptions(useri.id)
    if history:
        for item in history:
            st.sidebar.text(item.text)
        display_frequent_words(useri.id)
    else:
        st.sidebar.text("Transcription history will be shown here")

def record_audio():
    duration = 10  # seconds
    sample_rate = 44100

    # Record audio using sounddevice
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
        sd.wait()
        return audio_data
    except Exception as e:
        st.error(f"Error during audio recording: {e}")
        return None


def recorded_audio_to_text(audio_data):
    recognizer = sr.Recognizer()
    temp_wav_file = "temp_audio.wav"
    sf.write(temp_wav_file, audio_data, 44100)

    with sr.AudioFile(temp_wav_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        #return "Speech Recognition could not understand audio."
        st.warning("Speech Recognition could not understand audio. Try again.")
    except sr.RequestError as e:
        #return f"Could not request results from Google Speech Recognition service; {e}"
        st.warning(f"Could not request results from Google Speech Recognition service")
    finally:
        # Clean up: remove the temporary WAV file
        os.remove(temp_wav_file)
    

def convert_audio_to_text(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        #return "Speech Recognition could not understand audio."
        st.warning("Speech Recognition could not understand audio.")
    except sr.RequestError as e:
        #return f"Could not request results from Google Speech Recognition service; {e}"
        st.warning(f"Could not request results from Google Speech Recognition service; {e}")

def translate_text(text):
    translator = Translator()
    translated_text = translator.translate(text, dest='en').text
    return translated_text

# Function to save transcription to the database
def save_transcription(user_id, transcription):
    entry = Transcription(user_id=user_id, text=transcription['text'], language=transcription['language'])
    session.add(entry)
    session.commit()

# Function to display most frequent words
def display_frequent_words(user_id):
    transcriptions = session.query(Transcription).filter(Transcription.user_id == user_id).all()
    user = session.query(User).filter(User.id == user_id).first()
    username = user.username
    all_transcriptions = session.query(Transcription).all()

    user_text = ' '.join([t.text for t in transcriptions])
    all_text = ' '.join([t.text for t in all_transcriptions])

    user_word_freq = Counter(user_text.split())
    all_word_freq = Counter(all_text.split())

    # Get the top 10 words
    top_user_words = dict(user_word_freq.most_common(10))
    top_all_words = dict(all_word_freq.most_common(10))

    # Display user-specific word frequency
    st.subheader(f"Top 10 Most Frequent Words for {username}:")
    st.bar_chart(top_user_words, color='#ffaa00', use_container_width=True)

    # Display overall word frequency
    st.subheader("Top 10 Most Frequent Words Across All Users:")
    st.bar_chart(top_all_words, color='#ffaa00', use_container_width=True)


# Function to display top 3 unique phrases
def display_top_phrases(user_id,text):
    transcriptions = session.query(Transcription).filter(Transcription.user_id == user_id,Transcription.text==text).all()
    user_text = ' '.join([t.text for t in transcriptions])
    user_words = user_text.split()
    # Count the frequency of each word
    word_counts = Counter(user_words)

    # Sort words by frequency (least spoken words first)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1])

    # Get the top 3 least spoken words
    least_spoken_words = [word[0] for word in sorted_words[:3]]

    st.subheader("Top 3 unique Words:")
    st.write(least_spoken_words)

# Function to display similar users (bonus)
def display_similar_users(user_id):
    # Implement similarity detection logic here (e.g., using embeddings)
    pass

def register_user(username, password):
    # Hash the password before storing it
    hashed_password = pbkdf2_sha256.hash(password)

    # Insert the user into the database
    user = User(username=username, password=hashed_password)
    try:
        session.add(user)
        session.commit()
        st.success("Registration successful. You can now log in.")
    except:
        st.warning("This user already exists. Try another username.")

def login_user(username, password):
    # Retrieve the hashed password from the database
    result = session.query(User).filter(User.username == username).first()

    # If the user exists and the password is correct, return True
    if result and pbkdf2_sha256.verify(password, result.password):
        return True
    else:
        st.error("Wrong credentials.")

if __name__ == '__main__':
    audio_to_text_converter()
