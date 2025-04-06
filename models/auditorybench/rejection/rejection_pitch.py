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
    parser.add_argument("--data", type=str, default="height_of_sounds", required=False) 
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

    df['audio_paths1'] = df['audio_paths1'].apply(ast.literal_eval)
    df['audio_paths2'] = df['audio_paths2'].apply(ast.literal_eval)
    
    thresholds = [0.0, 0.2, 0.4, 0.6, 0.8]

    default_path = f"./sound_0.wav"

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        span1 = row['span1']
        span2 = row['span2']
        audio_paths1 = row['audio_paths1']
        audio_paths2 = row['audio_paths2']
    
        similarity1 = single_text_multi_audio_similarity(text=span1, audio_paths=audio_paths1, model=model, processor=processor, device=device)
        similarity2 = single_text_multi_audio_similarity(text=span2, audio_paths=audio_paths2, model=model, processor=processor, device=device)
    

        for thresh in thresholds:
            selected_path1 = None
            selected_path2 = None
            
            for i in range(similarity1.size(1)):
                if similarity1[0, i].item() >= thresh:
                    selected_path1 = audio_paths1[i]
                    break
            for i in range(similarity2.size(1)):
                if similarity2[0, i].item() >= thresh:
                    selected_path2 = audio_paths2[i]
                    break
         
            if selected_path1 is None:
                selected_path1 = default_path
            if selected_path2 is None:
                selected_path2 = default_path
                
            
            df.at[idx, f'audio_path1_{thresh}'] = selected_path1
            df.at[idx, f'audio_path2_{thresh}'] = selected_path2
            

    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    args = parse_args()
    print("----args_info----")
    print(args)
    seed_everything(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    rejection(args)
