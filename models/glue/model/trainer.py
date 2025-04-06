import torch.nn as nn
import torch
import numpy as np
from utils import AverageMeter, calc_f1_acc, calc_spearmanr  


def train_epoch(model, data_loader, optimizer, device, scheduler, fusion_idx):
    model.train()
    losses = AverageMeter()

    correct_predictions = 0
    total_steps = 0

    for batch in data_loader:
        batch_size = batch['input_ids'].size(0)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["labels"].to(device)

        audio_inputs = batch["audio_inputs"].to(device)
        spans_token_pos = batch["spans_token_pos"].to(device)
        in_audios = batch["in_audios"].to(device)

        optimizer.zero_grad()
        outputs, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_inputs=audio_inputs,
            spans_token_pos=spans_token_pos,
            in_audios=in_audios,
            labels=label,
            fusion_idx=fusion_idx,
            device=device
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), batch_size)

        if model.num_labels > 1:
          
            correct_predictions += calc_f1_acc(outputs, label, return_sum=True)
            total_steps += batch_size
        else:
        
            pass

    if model.num_labels > 1:
        epoch_acc = correct_predictions / total_steps
    else:
        epoch_acc = 0  

    return epoch_acc, losses.avg


@torch.no_grad()
def validate(model, data_loader, device, fusion_idx):
    model.eval()
    losses = []
    all_outputs = []
    all_labels = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        audio_inputs = batch["audio_inputs"].to(device)
        spans_token_pos = batch["spans_token_pos"].to(device)
        in_audios = batch["in_audios"].to(device)

        outputs, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_inputs=audio_inputs,
            spans_token_pos=spans_token_pos,
            in_audios=in_audios,
            labels=labels,
            fusion_idx=fusion_idx,
            device=device
        )
        losses.append(loss.item())
        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if model.num_labels == 1:
        metric_val = calc_spearmanr(all_outputs, all_labels)
    else:
        metric_val = calc_f1_acc(all_outputs, all_labels)

    return all_outputs, np.mean(losses), metric_val
