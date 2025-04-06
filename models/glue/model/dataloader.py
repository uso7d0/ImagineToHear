import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from functools import partial
import ast

class AudioGLUE_Dataset(Dataset):
    def __init__(
        self,
        texts,
        texts2,
        labels,
        spans_list,
        audio_paths_list,
        tokenizer,
        max_length=512,
        do_lower_case=True,
    ):
        self.texts = texts
        self.texts2 = texts2  
        self.labels = labels
        self.spans_list = spans_list
        self.audio_paths_list = audio_paths_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.do_lower_case = do_lower_case

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text1 = str(self.texts[idx])
        text2 = str(self.texts2[idx]) if self.texts2[idx] is not None else ""

        if self.do_lower_case:
            text1 = text1.lower()
            text2 = text2.lower()

        encoding = self.tokenizer(
            text1,
            text2,
            max_length=self.max_length,
            truncation=True,
            padding="do_not_pad", 
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)         
        attention_mask = encoding["attention_mask"].squeeze(0)  

        offsets = encoding["offset_mapping"].squeeze(0)     

        spans = self.spans_list[idx]         
        audio_paths = self.audio_paths_list[idx] 

        in_audios = []
        audio_arrays = []

        for apath in audio_paths:
            if any(apath.endswith(f'/sound_{i}.wav') for i in range(5)):
                in_audios.append(False)
            else:
                in_audios.append(True)
            wave, sr = librosa.load(apath, sr=16000)
            audio_arrays.append(wave)

        concat_text = self.tokenizer.decode(input_ids, skip_special_tokens=True).lower()

        spans_token_pos = []
        for span in spans:
            try:
                start_char = concat_text.index(span)
                end_char = start_char + len(span)
            except:
                spans_token_pos.append([0, 0])
                continue

            token_start = None
            for i, (off_s, off_e) in enumerate(offsets):
                if off_s <= start_char <= off_e:
                    token_start = i
                    break

            token_end = None
            if token_start is not None:
                for i in range(token_start, len(offsets)):
                    off_s, off_e = offsets[i]
                    if off_e >= end_char:
                        token_end = i + 1  
                        break

            if token_start is None or token_end is None:
                spans_token_pos.append([0, 0])
            else:
                spans_token_pos.append([token_start, token_end])


        label = self.labels[idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "spans_token_pos": spans_token_pos,  
            "audios": audio_arrays,              
            "in_audios": in_audios,               
        }

def audio_dynamic_collate_fn(batch, audio_extractor):
    max_seq_len = max(b["input_ids"].size(0) for b in batch)
    max_num_spans = max(len(b["spans_token_pos"]) for b in batch)

    input_ids_list = []
    attention_mask_list = []
    label_list = []
    spans_token_pos_list = []
    in_audio_list = []
    audio_list_2d = []

    for b in batch:
        seq_len = b["input_ids"].size(0)
        pad_len = max_seq_len - seq_len

        padded_ids = torch.cat(
            [b["input_ids"], torch.zeros(pad_len, dtype=torch.long)]
        )
        padded_mask = torch.cat(
            [b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]
        )

        input_ids_list.append(padded_ids)
        attention_mask_list.append(padded_mask)
        label_list.append(b["label"])

     
        stp = b["spans_token_pos"]
        while len(stp) < max_num_spans:
            stp.append([0,0])
        spans_token_pos_list.append(stp)
       
        ia = b["in_audios"]
        while len(ia) < max_num_spans:
            ia.append(False)
        in_audio_list.append(ia)
        
        audios_for_this_sample = b["audios"]
        while len(audios_for_this_sample) < max_num_spans:
            audios_for_this_sample.append(np.zeros(80000, dtype=np.float32))

        audio_list_2d.append(audios_for_this_sample)

    input_ids_tensor = torch.stack(input_ids_list, dim=0)        
    attention_mask_tensor = torch.stack(attention_mask_list, dim=0)  

    if isinstance(label_list[0], float):
        labels_tensor = torch.tensor(label_list, dtype=torch.float)
    else:
        labels_tensor = torch.tensor(label_list, dtype=torch.long)

    spans_token_pos_tensor = torch.tensor(spans_token_pos_list, dtype=torch.long)
    in_audio_tensor = torch.tensor(in_audio_list, dtype=torch.bool)

    audio_np = np.array(audio_list_2d, dtype=np.float32) 
    audio_torch = torch.from_numpy(audio_np)            
    audio_torch = audio_torch.transpose(0,1)             
   
    audio_features = []
    for i in range(max_num_spans):
        wave_i = audio_torch[i].tolist() 
        out = audio_extractor(
            wave_i,
            sampling_rate=16000,
            return_tensors="pt"
        )
        audio_features.append(out.input_values)

    audio_features = torch.stack(audio_features, dim=1)
  

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "labels": labels_tensor,
        "spans_token_pos": spans_token_pos_tensor,
        "in_audios": in_audio_tensor,
        "audio_inputs": audio_features, 
    }

def create_audio_data_loader(df, task_name, tokenizer, audio_extractor, batch_size, shuffle_=False):

    if task_name != 'sst2':
        texts = [text.split('[SEP]')[0] for text in df['sentence']]
        texts2 = [text.split('[SEP]')[1] for text in df['sentence']]
    else:
        texts = df['sentence'].tolist()
        texts2 = [None] * len(df)

    labels = df["label"].tolist() 

    spans_list = df["spans_audio"].apply(ast.literal_eval).tolist()
    audio_paths_list = df["audio_paths"].apply(ast.literal_eval).tolist()

    dataset = AudioGLUE_Dataset(
        texts=texts,
        texts2=texts2,
        labels=labels,
        spans_list=spans_list,
        audio_paths_list=audio_paths_list,
        tokenizer=tokenizer,
        max_length=512,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_,
        num_workers=1,
        collate_fn=partial(audio_dynamic_collate_fn, audio_extractor=audio_extractor)
    )
    return loader