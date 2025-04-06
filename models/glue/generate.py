import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForTokenClassification, BertTokenizer
import os
from diffusers import StableAudioPipeline
from tqdm import tqdm
import argparse
import random
import numpy as np
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--device", type=str, default="0", required=False)
    parser.add_argument("--data", type=str, default="sst2", required=False) 
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

def detect_target_word(sentence, tokenizer, model, device):
    encoding = tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    predictions = predictions.squeeze().tolist()

    detected_texts = []
    current_sentence = ""
    current_word = ""

    for token, label in zip(tokens, predictions):
        if label == 1:
            if token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word:
                    current_sentence += current_word + " "
                current_word = token
        else:
            if current_word:
                current_sentence += current_word + " "
                current_word = ""
            
            if current_sentence:
                detected_texts.append(current_sentence.strip().replace(" - ", "-"))
                current_sentence = ""

    if current_word:
        current_sentence += current_word
        detected_texts.append(current_sentence.strip().replace(" - ", "-"))


    return detected_texts


def generate(args):
    df = pd.read_csv(f"./")

    model = BertForTokenClassification.from_pretrained(f'./', num_labels=2)
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(f'./')
    descriptions = [detect_target_word(sentence, tokenizer, model, device) for sentence in tqdm(df['sentence'])]
    descriptions = [tmp if tmp != [] else ['sound'] for tmp in descriptions]
    model_gen = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to(device)

    output_dir = f'./'

    file_paths = []

    for i, description in enumerate(tqdm(descriptions, desc="Stable")):
        tmp_paths = []
        for desc in description:
            
            safe_desc = desc.replace(" ", "_").replace("/", "_")

            filename0 = f"{safe_desc}_0.wav"
            filepath0 = os.path.join(output_dir, filename0)
            filename1 = f"{safe_desc}_1.wav"
            filepath1 = os.path.join(output_dir, filename1)
            filename2 = f"{safe_desc}_2.wav"
            filepath2= os.path.join(output_dir, filename2)
            filename3 = f"{safe_desc}_3.wav"
            filepath3 = os.path.join(output_dir, filename3)
            filename4 = f"{safe_desc}_4.wav"
            filepath4 = os.path.join(output_dir, filename4)

            
            if os.path.exists(filepath0):
                tmp_paths.append([filepath0 ,filepath1, filepath2, filepath3, filepath4])
                continue
           

            prompt = desc
            negative_prompt = "Low quality"
            generator = torch.Generator("cuda").manual_seed(args.seed)

            audio = model_gen(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                audio_end_in_s=5.0,
                num_waveforms_per_prompt=5,
                generator=generator,
            ).audios
            
            output = audio[0].T.float().cpu().numpy()
            sf.write(filepath0, output, model_gen.vae.sampling_rate)

            output = audio[1].T.float().cpu().numpy()
            sf.write(filepath1, output, model_gen.vae.sampling_rate)

            output = audio[2].T.float().cpu().numpy()
            sf.write(filepath2, output, model_gen.vae.sampling_rate)

            output = audio[3].T.float().cpu().numpy()
            sf.write(filepath3, output, model_gen.vae.sampling_rate)

            output = audio[4].T.float().cpu().numpy()
            sf.write(filepath4, output, model_gen.vae.sampling_rate)
        file_paths.append(tmp_paths)


    df['audio_paths'] = file_paths
    df['spans_audio'] = descriptions
    df.to_csv(f"./", index=False)



if __name__ == "__main__":
    args = parse_args()
    print("----args_info----")
    print(args)
    seed_everything(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    generate(args)