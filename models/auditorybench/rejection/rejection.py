import ast
import pandas as pd
import torch
from clap_similarity import single_text_multi_audio_similarity
import argparse
import random
import os
import numpy as np
from tqdm import tqdm
from transformers import ClapProcessor, ClapModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--device", type=str, default="0", required=False)
    parser.add_argument("--data", type=str, default="animal_sounds", required=False) 
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
    processor = ClapProcessor.from_pretrained(model_id, cache_dir='./models')
    model = ClapModel.from_pretrained(model_id, cache_dir='./models').to(device)

 
    df = pd.read_csv(input_csv)
    df['audio_paths'] = df['audio_paths'].apply(ast.literal_eval)

    thresholds = [00, 0.2, 0.4, 0.6, 0.8]

    default_path = f"./sound_0.wav"

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        description = row['description']
        audio_paths = row['audio_paths']
        
        similarity = single_text_multi_audio_similarity(description, audio_paths, model=model, processor=processor, device=device)
    
        for thresh in thresholds:
            selected_path = None
            
            for i in range(similarity.size(1)):
                if similarity[0, i].item() >= thresh:
                    selected_path = audio_paths[i]
                    break
         
            if selected_path is None:
                selected_path = default_path
            

            df.at[idx, f'audio_path_{thresh}'] = selected_path

    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    args = parse_args()
    print("----args_info----")
    print(args)
    seed_everything(args.seed)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    rejection(args)
