import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from dataloader import create_data_loader
from model import AudioImaginationBERT
from trainer import train_epoch, validate
from transformers import AutoTokenizer, ASTFeatureExtractor
from transformers.optimization import get_cosine_schedule_with_warmup
import torch.nn as nn


    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--lr", type=float, default=3e-4, required=False)
    parser.add_argument("--device", type=str, default="0", required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--data_path", type=str, default="./", required=False)
    parser.add_argument("--output_path", type=str, default="./", required=False)
    parser.add_argument("--language_model_name", type=str, default="google-bert/bert-base-uncased", required=False)
    parser.add_argument("--audio_model_name", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593", required=False)
    parser.add_argument("--save_model_name", type=str, default="audio-bert", required=False)
    parser.add_argument("--clap", type=float, default=0.4, required=False)
    parser.add_argument("--fusion_idx", type=int, default=0, required=False)
    parser.add_argument("--iter", type=int, default=5, required=False)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Train(args):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(f"./")
    dev_df = pd.read_csv(f"./")
    test_df = pd.read_csv(f"./")
    test_df2 = pd.read_csv(f"./")

    train_df['audio_path'] = train_df[f'audio_path_{args.clap}']
    dev_df['audio_path'] = dev_df[f'audio_path_{args.clap}']
    test_df['audio_path'] = test_df[f'audio_path_{args.clap}']
    test_df2['audio_path'] = test_df2[f'audio_path_{args.clap}']

    os.makedirs(args.output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, cache_dir="./models")
    extractor = ASTFeatureExtractor.from_pretrained(args.audio_model_name, cache_dir="./models")

    train_data_loader = create_data_loader(train_df, tokenizer, extractor, args.batch_size, shuffle_=True)
    dev_data_loader = create_data_loader(dev_df, tokenizer, extractor, args.batch_size)
    test_data_loader = create_data_loader(test_df, tokenizer, extractor, args.batch_size)
    test_data_loader2 = create_data_loader(test_df2, tokenizer, extractor, args.batch_size)

    model = AudioImaginationBERT(
        language_model_path=args.language_model_name,
        audio_model_path=args.audio_model_name,
        tokenizer=tokenizer,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_data_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    max_acc = 0
    for epoch in range(args.epochs):
        print("-" * 10)
        print(f"Epoch {epoch}/{args.epochs-1}")
        print("-" * 10)
        train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler, args.fusion_idx-1)
        dev_output, dev_loss, dev_acc = validate(model, dev_data_loader, args.fusion_idx-1, device)

        if dev_acc > max_acc:
            max_acc = dev_acc
            torch.save(model.state_dict(), './model.pt')

        print(f"Train loss {train_loss} accuracy {train_acc}")
        print(f"Dev loss {dev_loss} accuracy {dev_acc}")
        print("")

    
    model.load_state_dict(torch.load('./model.pt'))
    test_output, test_loss, test_acc = validate(model, test_data_loader, args.fusion_idx-1, device)
    wiki_output, test_loss2, test_acc2 = validate(model, test_data_loader2, args.fusion_idx-1, device)

    print(f"Best dev acc {max_acc}")
    print(f"test acc {test_acc}")
    print(f"wiki test acc {test_acc2}")





if __name__ == "__main__":
    args = parse_args()
    print("----args_info----")
    print(args)
    seed_everything(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    Train(args)
