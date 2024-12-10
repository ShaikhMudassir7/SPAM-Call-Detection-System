import streamlit as st
import os
import joblib
from transformers import pipeline
import tempfile
import nltk

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in transcription.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Streamlit UI Setup
st.title("AI-Powered Message Classification")
st.subheader("Upload an audio file to transcribe and classify the message")

# Audio File Upload
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
whisper = pipeline('automatic-speech-recognition', model='openai/whisper-medium', device=0)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav", start_time=0)
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
        temp_file.write(uploaded_file.read())
        audio_path = temp_file.name

    st.success(f"Uploaded audio saved as {audio_path}")
    
    # Transcription Step
    try:
        st.write("Transcribing audio...")
        text = whisper(
            audio_path,
            chunk_length_s=30,  # Optional: adjust for longer files
            generate_kwargs={"language": "<|en|>"}  # Enforce English transcription
        )
        transcription = text['text']
        st.text_area("Transcript of the audio", value=transcription, height=200)
        
        # Classification Step
        st.write("Classifying the transcribed message...")
        
        # Load the pre-trained model
        model_path = "spam_detection_pipeline.pkl"  # Replace with your trained model path
        if os.path.exists(model_path):
            pipeline = joblib.load(model_path)
            
            # Predict Message Category
            prediction = pipeline.predict([transcription])
            st.write(f"**Predicted Category:** {prediction}")
        else:
            st.error("Model file not found. Please ensure 'spam_detection_pipeline.pkl' exists in the directory.")
    
    except Exception as e:
        st.error(f"An error occurred during transcription or classification: {e}")
