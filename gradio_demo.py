# pydantic-1.10.7, gradio 3.34 are required

import gradio as gr
import torch
import librosa
import numpy as np
from model import Model

# Initialize constants and model
CKPT = '../models/model_DF_WCE_100_16_1e-06/epoch_36.pth'
EER = 0.2097  # EER for thresholding
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Model(None, device)
sd = torch.load(CKPT)
sd = {k.replace('module.', ''): v for k, v in sd.items()}  # model was saved in DataParallel wrap
model.load_state_dict(sd)
model.eval()

model = model.to(device)

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # Padding logic if needed
    num_repeats = (max_len // x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x

def classify_audio(filepath: str) -> str:

    audio, _ = librosa.load(filepath, sr=16000)
    audio = pad(audio)
    audio_tensor = torch.Tensor(audio).to(device).unsqueeze(0)  # add batch dimension

    print('Running model inference...')
    with torch.no_grad():
        out = model(audio_tensor)
    out_score = out[0, 1].item()

    result = "Real" if out_score >= EER else "Spoof"
    return result + f"\n(Raw CM score from model: {out_score:.4f})"


# Interface
demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.components.Audio(type="filepath"),
    outputs='text'
)

demo.launch()
