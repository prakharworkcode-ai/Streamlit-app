import streamlit as st
import torch
import librosa
import numpy as np
import tempfile

from model import ClassifierModelV1

SAMPLING_FREQUENCY = 16000
MEL_BANDS = 128

device = "cpu"
@st.cache_resource
def load_model():

    model = ClassifierModelV1(input_shape=1,output_shape=1)

    model.load_state_dict(
        torch.load("models/MODEL_DFC.pth",map_location=device)
    )

    model.eval()
    return model

model = load_model()

def gen_meldb(audio_path):

    y, sr = librosa.load(audio_path,sr=SAMPLING_FREQUENCY)

    target_len = 2 * SAMPLING_FREQUENCY

    if len(y) < target_len:
        y = np.pad(y,(0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=MEL_BANDS,
        fmin=20,
        fmax=8000
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db

def predict(audio_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(audio_file.read())

        mel = gen_meldb(tmp.name)

    mel = (
        mel - mel.mean()
    ) / (
        mel.std() + 1e-8
    )

    mel = torch.tensor(
        mel,
        dtype=torch.float32
    )

    mel = mel.unsqueeze(0)
    mel = mel.unsqueeze(0)

    with torch.no_grad():

        logits = model(mel)

        prob = torch.sigmoid(
            logits
        ).item()

    return prob


st.title("Deepfake Audio Detector")

uploaded_file = st.file_uploader(
    "Upload WAV file",
    type=["wav"]
)

if uploaded_file:

    st.audio(uploaded_file)

    if st.button("Predict"):

        prob = predict(uploaded_file)

        if prob > 0.5:

            st.error(
                f"FAKE ({prob:.2%})"
            )

        else:

            st.success(
                f"REAL ({1-prob:.2%})"
            )