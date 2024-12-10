import streamlit as st
import os
import joblib
from transformers import pipeline


# Streamlit UI Setup
st.title("AI-Powered Message Classification")
st.subheader("Upload an audio file to transcribe and classify the message")

# Audio File Upload
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
whisper = pipeline('automatic-speech-recognition', model = 'openai/whisper-medium', device = 0)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav", start_time=0)
    audio_path = f"temp_audio.{uploaded_file.type}"
    
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded audio saved as {audio_path}")
    
    # Transcription Step
    try:
        st.write("Transcribing audio...")
        transcript = whisper('audio_path')
        
        
        st.text_area("Transcript of the audio", value=transcript, height=200)
        
        # Classification Step
        st.write("Classifying the transcribed message...")
        
        # Load the pre-trained model
        model_path = "spam_detection_pipeline.pkl"  # Replace with your trained model path
        if os.path.exists(model_path):
            pipeline = joblib.load(model_path)
            
            # Predict Message Category
            prediction = pipeline.predict([transcript])[0]
            category_mapping = {0: "Not Spam", 1: "Spam"}  # Update categories if needed
            predicted_category = category_mapping.get(prediction, "Unknown")
            
            st.write(f"**Predicted Category:** {predicted_category}")
        else:
            st.error("Model file not found. Please ensure 'spam_detection_pipeline.pkl' exists in the directory.")
    
    except Exception as e:
        st.error(f"An error occurred during transcription or classification: {e}")

# Note: You can implement additional logic for dataset reloading and visualization as needed.
