import argparse
import ast
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import ClapProcessor, ClapModel
from clap_similarity import batch_text_multi_audio_similarity 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--device", type=str, default="0", required=False)
    parser.add_argument("--data", type=str, default="stsb", required=False)
    parser.add_argument("--set", type=str, default="train", required=False)
    return parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def rejection(args): 
    input_csv = f"./"
    output_csv = f"./"

    model_id = "laion/clap-htsat-fused"
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id).to(device)

    df = pd.read_csv(input_csv)
  
    df['audio_paths'] = df['audio_paths'].apply(ast.literal_eval)
    df['spans_audio'] = df['spans_audio'].apply(ast.literal_eval)

    thresholds = [0.0, 0.2, 0.3, 0.4, 0.6, 0.8]
    default_path = f".//sound_0.wav"


    for idx, row in tqdm(df.iterrows(), total=len(df)):
        spans = row['spans_audio']
        audio_paths = row['audio_paths']
        if spans == ['sound']:
            for thresh in thresholds:
                df.at[idx, f"audio_paths_{thresh}"] = [default_path]
            continue

    
        chunked_sims = batch_text_multi_audio_similarity(
            spans=spans,
            all_audio_paths=audio_paths,
            model=model,
            processor=processor,
            device=device,
            target_sr=48000  
            )
     
        for thresh in thresholds:
            selected_paths_for_this_thresh = []

            for i, span_text in enumerate(spans):
                candidate_audios = audio_paths[i]
                similarity_row = chunked_sims[i] 

                selected_path = None
                for j in range(similarity_row.size(1)):
                    if similarity_row[0, j].item() > thresh:
                        selected_path = candidate_audios[j]
                        break

                if selected_path is None:
                    selected_path = default_path
                
                selected_paths_for_this_thresh.append(selected_path)

            df.at[idx, f"audio_paths_{thresh}"] = selected_paths_for_this_thresh

    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    args = parse_args()
    print("----args_info----")
    print(args)
    seed_everything(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    rejection(args)
