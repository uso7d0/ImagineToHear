import torch
from trainer import train_epoch, validate
from dataloader import create_audio_data_loader
from model import AudioImaginationForGLUE
import pandas as pd
import os
import argparse
import warnings
import random
import numpy as np
from transformers import ASTFeatureExtractor
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=str, default="base", required=False)
    parser.add_argument("--epochs", type=int, default=3, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--lr", type=float, default=2e-5, required=False)
    parser.add_argument("--device", type=str, default="0", required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--data_path", type=str, default="./", required=False)
    parser.add_argument("--output_path", type=str, default="./", required=False)
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased", required=False)
    parser.add_argument("--save_model_name", type=str, default="bert-base", required=False)
    parser.add_argument("--clap", type=float, default=0.3, required=False)
    parser.add_argument("--fusion_idx", type=int, default=0, required=False)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_num_labels = {
        "sst2": 2,
        "qnli": 2,
        "qqp": 2,
        "mnli": 3,
        "mrpc": 2,
        "stsb": 1,
    }

    task_list = ["sst2", "qnli", "qqp", "mnli", "mrpc", "stsb"]
    base_path = "./"

    audio_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    for task in task_list:
        print(f"=== [TASK: {task}] ===")
        train_df = pd.read_csv(f"{base_path}/{task}_train.csv")
        dev_df   = pd.read_csv(f"{base_path}/{task}_valid.csv")

        train_df['audio_paths'] = train_df[f'audio_paths_{args.clap}']
        dev_df['audio_paths'] = dev_df[f'audio_paths_{args.clap}']

        train_dl = create_audio_data_loader(train_df, task, tokenizer, audio_extractor, args.batch_size, shuffle_=True)
        dev_dl = create_audio_data_loader(dev_df,   task, tokenizer, audio_extractor, args.batch_size, shuffle_=False)

        num_labels = task_num_labels[task]
        model = AudioImaginationForGLUE(
            language_model_path=args.model_name,
            audio_model_path="MIT/ast-finetuned-audioset-10-10-0.4593",
            tokenizer=tokenizer,
            num_labels=num_labels
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total_steps = len(train_dl)*args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps*0.1),
            num_training_steps=total_steps
        )

        best_metric = -999
        for epoch in range(args.epochs):
            print(f"--- EPOCH {epoch+1}/{args.epochs} ---")
            train_acc, train_loss = train_epoch(model, train_dl, optimizer, device, scheduler, args.fusion_idx)
            _, dev_loss, dev_metric = validate(model, dev_dl, device, args.fusion_idx)

            print(f"[Train] loss={train_loss:.4f}, acc(only cls)={train_acc:.4f}")
            print(f"[Valid] loss={dev_loss:.4f}, metric={dev_metric:.4f}")

            if dev_metric > best_metric:
                best_metric = dev_metric
                save_path = f"./{task}.pt"
                torch.save(model.state_dict(), save_path)
        
        print(f"Best dev: {best_metric}")
        print("=====================================\n")



if __name__ == "__main__":
    args = parse_args()
    print("----args_info----")
    print(args)
    seed_everything(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    Train(args)
