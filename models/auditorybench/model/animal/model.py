import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoConfig, ASTModel, ASTConfig
from transformers.modeling_outputs import MaskedLMOutput

def extract_mask_token_embeddings(outputs, mask_token_index):
    predicted_token_ids = []
    

    for i in range(mask_token_index.size(0)):
        mask_indices = mask_token_index[i].nonzero(as_tuple=True)[0] 
        
        if len(mask_indices) > 0:
            predicted_ids = outputs.logits[i, mask_indices].argmax(dim=-1)  
            predicted_token_ids.append(predicted_ids)
        else:
            predicted_token_ids.append(torch.tensor([], device=outputs.logits.device))
    
    return predicted_token_ids

class AudioMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
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
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                         num_heads=num_heads, 
                                         dropout=dropout, 
                                         batch_first=True)
        
        
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm2 = nn.LayerNorm(hidden_dim)

        self.fusion_gate = FusionGateUnit(hidden_dim)

    def forward(self, text_feats: torch.Tensor, audio_feats: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        
        k_v = audio_feats
        
        q = text_feats
        
        attn_output, _ = self.mha(q, k_v, k_v, key_padding_mask=None)  
        
        out1 = self.norm1(q + attn_output)
        
        ffn_output = self.ffn(out1)
        out2 = self.norm2(out1 + ffn_output)

        fused_out = self.fusion_gate(H_a=out2, H_t=text_feats)

        return fused_out

class AudioImaginationBERT(nn.Module):
    def __init__(self, language_model_path: str, audio_model_path: str, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.language_config = AutoConfig.from_pretrained(language_model_path)
        self.language_model = AutoModelForMaskedLM.from_pretrained(language_model_path)
        self.language_model.resize_token_embeddings(len(tokenizer))
        
        self.audio_config = ASTConfig.from_pretrained(audio_model_path)
        self.audio_enc = ASTModel.from_pretrained(audio_model_path)

        hidden_dim = self.language_config.hidden_size
        num_heads = self.language_config.num_attention_heads
        ff_dim = self.language_config.intermediate_size
        audio_enc_dim = self.audio_config.hidden_size

        self.fusion_layer = FusionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim
        )
        self.mlp = AudioMLP(audio_enc_dim, hidden_dim)

    def fuse_spans_with_audio(
    self,
    hidden_states: torch.Tensor,
    token_indices: torch.Tensor,
    in_audio_indices: list[int],
    audio_embs: torch.Tensor,
    fusion_layer: nn.Module,
    device: torch.device
) -> torch.Tensor:
    
        if len(in_audio_indices) > 0:
            span_lengths = token_indices[in_audio_indices, 1] - token_indices[in_audio_indices, 0]
            max_span_len = int(span_lengths.max())
        else:
            max_span_len = 0

        span_embs = torch.zeros(len(in_audio_indices), max_span_len, hidden_states.size(-1)).to(device)

        for i, idx in enumerate(in_audio_indices):
            start_pos, end_pos = token_indices[idx]
            end_pos = min(end_pos, hidden_states.size(1))  
            length = end_pos - start_pos
            span = hidden_states[idx, start_pos:end_pos, :]
            span_embs[i, :length, :] = span
        fused_span_embs = fusion_layer(span_embs, audio_embs[in_audio_indices])

   
        for i, idx in enumerate(in_audio_indices):
            start_pos, end_pos = token_indices[idx]
            end_pos = min(end_pos, hidden_states.size(1))
            length = end_pos - start_pos
            hidden_states[idx, start_pos:end_pos, :] = fused_span_embs[i, :length, :]

        return hidden_states

    def forward(self, audio_features, span_token_pos, input_ids, targets, in_audio, fusion_idx, device):
        audio_embs = self.audio_enc(input_values=audio_features).last_hidden_state
        audio_embs = self.mlp(audio_embs)

        hidden_states = self.language_model.bert.embeddings(input_ids)

        in_audio_indices = [i for i, v in enumerate(in_audio) if v]

        hidden_states = self.language_model.bert.embeddings(input_ids)

        if fusion_idx == -1:
            hidden_states = self.fuse_spans_with_audio(
                hidden_states, span_token_pos, in_audio_indices, audio_embs, self.fusion_layer, device)
           
        for i, layer_moduels in enumerate(self.language_model.bert.encoder.layer):
            hidden_states = layer_moduels(hidden_states)[0]

            if i == fusion_idx:
                hidden_states = self.fuse_spans_with_audio(
                hidden_states, span_token_pos, in_audio_indices, audio_embs, self.fusion_layer, device)

            
        logits = self.language_model.cls(hidden_states)
        mask_token_index = (input_ids == self.tokenizer.mask_token_id)
        targets[~mask_token_index] = -100
        if targets is not None:
            loss_func = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_func(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            outputs = MaskedLMOutput(
                loss=loss,
                logits=logits,
                )
        else:
            outputs = MaskedLMOutput(
                loss=None,
                logits=logits
            )
        predicted_token_id = extract_mask_token_embeddings(outputs, mask_token_index)

        return outputs, predicted_token_id
