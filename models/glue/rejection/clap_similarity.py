import torch
import librosa
import torch.nn.functional as F

def batch_text_multi_audio_similarity(
    spans: list[str],
    all_audio_paths: list[list[str]],
    model,
    processor,
    device,
    target_sr=48000,
):
   
    text_inputs = processor(
        text=spans,
        return_tensors="pt",
        padding=True 
    ).to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs) 


    flattened_paths = [p for paths_per_span in all_audio_paths for p in paths_per_span]
    
    waveforms = []
    for path in flattened_paths:
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

    sim_matrix = F.cosine_similarity(
        text_emb.unsqueeze(1),
        audio_emb.unsqueeze(0),
        dim=-1
    )


    chunked_similarities = []
    start = 0
    for i in range(len(spans)):
        chunk_size = len(all_audio_paths[i])
        row_i = sim_matrix[i, start : start + chunk_size]  
        chunked_similarities.append(row_i.unsqueeze(0))
        start += chunk_size

    return chunked_similarities

