import argparse
import os
import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
from trainer import encode_data, train_epoch, evaluate
import pandas as pd
import os
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--lr", type=float, default=3e-5, required=False)
    parser.add_argument("--device", type=str, default="0", required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--train_data", type=str, default="sst2", required=False)
    return parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.determinitmpic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    df_train = pd.read_csv(f'./')
    df_dev = pd.read_csv(f'./')
    df_test = pd.read_csv(f'./')


    train_inputs, train_masks, train_labels = encode_data(df_train, tokenizer)

    val_inputs, val_masks, val_labels = encode_data(df_dev, tokenizer)
    test_inputs, test_masks, test_labels = encode_data(df_test, tokenizer)

    

    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)

    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
    

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)
    

    optimizer = AdamW(model.parameters(), lr=args.lr)

    train_loss = 0
    for epoch in range(args.epochs):
        train_loss += train_epoch(model, train_dataloader, optimizer, device)

        _, val_f1 = evaluate(model, val_dataloader, device)
        _, test_f1 = evaluate(model, test_dataloader, device)
        
        print(f"Train loss {train_loss/args.epochs}")
        print(f"{args.train_data} dev: {np.mean(val_f1):.4f}")
        print(f"{args.train_data} test: {np.mean(test_f1):.4f}")

        print('-'*60)

    model.save_pretrained(f"./")
    tokenizer.save_pretrained(f"./")
    
    
    


if __name__ == "__main__":
    args = parse_args()
    print("----args_info----")
    print(args)
    seed_everything(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    train(args)