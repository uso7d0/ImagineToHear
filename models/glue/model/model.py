import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, ASTModel, ASTConfig

class AudioMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FusionGateUnit(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_t = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, H_a: torch.Tensor, H_t: torch.Tensor) -> torch.Tensor:
        gate = self.sigmoid(self.W_a(H_a) + self.W_t(H_t)) 
        fused = gate * H_a + (1.0 - gate) * H_t
        return fused

class FusionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fusion_gate = FusionGateUnit(hidden_dim)

    def forward(self, text_feats: torch.Tensor, audio_feats: torch.Tensor):
        attn_output, _ = self.mha(text_feats, audio_feats, audio_feats)
        out1 = self.norm1(text_feats + attn_output)
        out2 = self.norm2(out1 + self.ffn(out1))
        fused_out = self.fusion_gate(out2, text_feats)
        return fused_out

class AudioImaginationForGLUE(nn.Module):
    def __init__(self, 
                 language_model_path: str, 
                 audio_model_path: str, 
                 tokenizer,
                 num_labels: int=2, 
                 dropout_rate: float=0.1):
        super().__init__()

    
        self.language_config = AutoConfig.from_pretrained(language_model_path)
        self.language_enc = AutoModel.from_pretrained(language_model_path)
        self.language_enc.resize_token_embeddings(len(tokenizer))
       
        self.audio_config = ASTConfig.from_pretrained(audio_model_path)
        self.audio_enc = ASTModel.from_pretrained(audio_model_path)
        

        hidden_dim = self.language_config.hidden_size
        num_heads = self.language_config.num_attention_heads
        ff_dim = self.language_config.intermediate_size
        audio_enc_dim = self.audio_config.hidden_size

        self.fusion_layer = FusionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout_rate
        )
        self.mlp = AudioMLP(audio_enc_dim, hidden_dim)

    
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_rate)
        

    def fuse_spans_with_audio(self,
                         hidden_states: torch.Tensor,
                         audio_inputs: torch.Tensor,
                         spans_token_pos: torch.Tensor,
                         in_audios: torch.Tensor,
                         span_idx: int,
                         device: str = "cuda") -> torch.Tensor:
     
        B, S, hidden_dim = hidden_states.size()


        audio_feats = self.audio_enc(input_values=audio_inputs[:, span_idx])[0]  
        audio_feats = self.mlp(audio_feats)                                    

        active_idx = (in_audios[:, span_idx] == True).nonzero(as_tuple=True)[0]
        if len(active_idx) == 0:
            return hidden_states 

        start_positions = spans_token_pos[active_idx, span_idx, 0]
        end_positions   = spans_token_pos[active_idx, span_idx, 1]
        lengths = end_positions - start_positions
        max_len = int(lengths.max().item())
    
        span_embs = torch.zeros(len(active_idx), max_len, hidden_dim, device=device)
        for j, sample_id in enumerate(active_idx):
            st = start_positions[j].item()
            ed = min(end_positions[j].item(), S)
            L = ed - st
            if L > 0:
                span_embs[j, :L, :] = hidden_states[sample_id, st:ed, :]

        fused_span_embs = self.fusion_layer(span_embs, audio_feats[active_idx])

        for j, sample_id in enumerate(active_idx):
            st = start_positions[j].item()
            ed = min(end_positions[j].item(), S)
            L = ed - st
            if L > 0:
                hidden_states[sample_id, st:ed, :] = fused_span_embs[j, :L, :]

        return hidden_states

    def forward(self, 
                input_ids, 
                attention_mask, 
                audio_inputs, 
                spans_token_pos, 
                in_audios,
                labels=None,
                fusion_idx=None,
                device="cuda"):

        hidden_states = self.language_enc.embeddings(input_ids)

        B = input_ids.size(0)
        S = hidden_states.size(1)

        if fusion_idx == -1:
            for span_idx in range(spans_token_pos.size(1)):
                hidden_states = self.fuse_spans_with_audio(
                    hidden_states,
                    audio_inputs,
                    spans_token_pos,
                    in_audios,
                    span_idx,
                    device
                )

        for i, layer_moduels in enumerate(self.language_enc.encoder.layer):
            hidden_states = layer_moduels(hidden_states)[0]

            if i == fusion_idx:

                for span_idx in range(spans_token_pos.size(1)):
                    hidden_states = self.fuse_spans_with_audio(
                        hidden_states,
                        audio_inputs,
                        spans_token_pos,
                        in_audios,
                        span_idx,
                        device
                    )


        cls_emb = self.dropout(hidden_states[:, 0, :])
        logits = self.classifier(cls_emb) 

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)

        return logits, loss
