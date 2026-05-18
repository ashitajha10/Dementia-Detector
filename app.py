import streamlit as st
import joblib
import numpy as np
from extract_combined_features_single import extract_features
import tempfile
import os

model_path = "outputs/models/ensemble_lasso_rf_svm.pkl"
model = joblib.load(model_path)

st.set_page_config(page_title="Dementia Detection (Voting Model)", layout="centered")

# Custom styling for premium UI feel
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F8FAFC;
        border-radius: 8px 8px 0px 0px;
        padding-left: 16px;
        padding-right: 16px;
        font-weight: 600;
        border: 1px solid #E2E8F0;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #2563EB !important;
        border: 1px solid #93C5FD !important;
        border-bottom: none !important;
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Outfit', sans-serif;
    }
    h2, h3 {
        color: #2C3E50;
    }
</style>
""", unsafe_allow_html=True)

st.title("Dementia Detection App")
st.subheader("Using Voting Ensemble on Combined Features")

st.markdown("""
This application detects markers of dementia from a patient's spoken transcript using a Voting Ensemble Classifier trained on a rich set of linguistic, acoustic hesitation, and embedding-based features.
""")

def run_prediction(text):
    if not text.strip():
        st.warning("Please enter or transcribe some text input first.")
        return
        
    with st.spinner("Extracting features and predicting..."):
        # Extract features
        features = extract_features(text)

        # Predict
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        # Display result
        label_map = {0: "Control", 1: "Dementia"}
        
        st.write("---")
        st.markdown("### 📊 Prediction Analysis")
        
        if prediction == 1:
            st.error(f"⚠️ **Result:** Patient shows high indicators of **{label_map[prediction]}**")
        else:
            st.success(f"✅ **Result:** Patient is classified as **{label_map[prediction]}**")
            
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Healthy Control Probability", value=f"{proba[0]*100:.1f}%")
        with col2:
            st.metric(label="Dementia Probability", value=f"{proba[1]*100:.1f}%")
            
        st.markdown("**Dementia Indicator Level:**")
        st.progress(float(proba[1]))

tab1, tab2 = st.tabs(["✍️ Manual Transcript Input", "🎤 Voice Recording & Transcription"])

with tab1:
    st.markdown("### ✍️ Manual Input")
    st.info("Type or paste a pre-recorded patient transcript below.")
    user_input_manual = st.text_area(
        "Patient Speech Transcript:",
        height=200,
        placeholder="Type patient transcript here...",
        key="manual_text"
    )
    if st.button("Predict from Manual Transcript", key="btn_manual"):
        run_prediction(user_input_manual)

with tab2:
    st.markdown("### 🎤 Voice Recording & Transcription")
    st.info("Click the microphone button to start recording. Speak clearly, and click stop when you are done. The browser will transcribe your speech automatically.")
    
    from streamlit_mic_recorder import speech_to_text
    
    # Render the speech-to-text recording widget
    transcribed_text = speech_to_text(
        language='en',
        start_prompt="🎤 Start Recording Voice",
        stop_prompt="🛑 Stop Recording (Transcribe)",
        just_once=False,
        key="stt_recorder"
    )
    
    if transcribed_text:
        # Save transcription to session state
        st.session_state.transcription = transcribed_text
        st.success("Speech successfully transcribed!")
            
    # Always display the transcription in an editable text area so the clinician can verify/edit it
    user_input_voice = st.text_area(
        "Transcribed Text (Review and edit before predicting):",
        value=st.session_state.get("transcription", ""),
        height=200,
        placeholder="Awaiting transcription... Click 'Start Recording Voice' above to begin.",
        key="voice_text"
    )
    # Sync edited text to session state
    st.session_state.transcription = user_input_voice
    
    if st.button("Predict from Transcribed Voice", key="btn_voice"):
        run_prediction(user_input_voice)
