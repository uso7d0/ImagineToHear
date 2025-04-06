import torch
import librosa
import torch.nn.functional as F
import pandas as pd

def single_text_multi_audio_similarity(
    text: str,
    audio_paths: list,  
    model,
    processor,
    device,
) -> torch.Tensor:
    try:
        text_inputs = processor(
            text=text, 
            return_tensors="pt",
        ).to(device)
    except:
        print(text)
        text_inputs = processor(
            text='sound', 
            return_tensors="pt",
        ).to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)  


    waveforms = []
    target_sr = 48000
    for path in audio_paths:
        waveform, sr = librosa.load(path, sr=None)
        waveforms.append(waveform)

    audio_inputs = processor(
        audios=waveforms,
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        audio_emb = model.get_audio_features(**audio_inputs) 

    similarity_matrix = F.cosine_similarity(
        text_emb.unsqueeze(1),    
        audio_emb.unsqueeze(0),   
        dim=-1
    )

    return similarity_matrix
