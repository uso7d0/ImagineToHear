# 💭🎧 Imagine to Hear: Auditory Knowledge Generation can be an Effective Assistant for Language Models

[**📄 arXiv**](https://arxiv.org/abs/2503.16853) | [**🌐 Project Page**](https://imagine-to-hear.github.io)

This repository contains the official code for our paper:  
**"Imagine to Hear: Auditory Knowledge Generation can be an Effective Assistant for Language Models" (Yoo et al., 2025)**


## Introduction
Language models (LMs) often lack auditory commonsense knowledge due to their text-only pretraining. While retrieval-based approaches (e.g., AudioBERT) augment models with external audio features, they suffer from limitations like database dependency and retrieval failure.  

**Imagine to Hear (ITH)** overcomes these issues by generating **auditory knowledge** using **text-to-audio generative models**.  
It:
- Detects multiple audio-related text spans,
- Generates corresponding audio,
- Injects audio into the LM via a **fusion module**,  
achieving **state-of-the-art performance** on AuditoryBench—**without using external audio databases**.

![image](https://github.com/user-attachments/assets/f18b6a68-1033-438c-8182-44e685e5cf82)

## Method
### 🔍 1. Imagination Module
- Detects **multiple auditory spans** from text (via fine-tuned BERT-base).
- Generates audio for each span using a **text-to-audio diffusion model** (e.g., Stable Audio).
- Uses **CLAP-based rejection sampling** to ensure semantic alignment.

### 🔗 2. Fusion Module
- Employs **cross-attention** between auditory and textual tokens.
- Integrates audio via a **Fusion Gate Unit**, adaptively blending text and sound representations.

### 🧠 3. Language Encoder
- Final language prediction is made by a BERT-base LM, conditioned on fused audio-text representation.

## Training
- Span Detector: 5 epochs, batch size 16, LR 1e-5 (AdamW).
- Language Model: 8 epochs, batch size 32, LR 3e-4 - 4e-5.
- Audio Generator: Frozen text-to-audio diffusion model (e.g., stable-audio-open-1.0).
- All models trained on NVIDIA RTX 4090 / A6000 / L40S.

## Results
ITH achieves **new SOTA results** on AuditoryBench.

### 🐾 Animal Sound Recognition

| Model        | Dev   | Test  | Wiki Test |
|--------------|-------|-------|-----------|
| BERT-base    | 15.51 | 13.46 | 3.05      |
| RoBERTa-base | 14.67 | 14.04 | 2.54      |
| AudioBERT    | 38.28 | 36.63 | 14.32     |
| **ITH (Ours)** | **39.36** | **41.55** | **19.09**     |

### 🎵 Sound Pitch Comparison

| Model        | Dev   | Test  | Wiki Test |
|--------------|-------|-------|-----------|
| BERT-base    | 59.42 | 60.41 | 48.06     |
| RoBERTa-base | 54.50 | 55.84 | 47.45     |
| AudioBERT    | 73.18 | 74.83 | 55.31     |
| **ITH (Ours)** | **79.82** | **78.96** | **76.74**     |


## 🎧 Case Study

Imagine to Hear (ITH) uses generated auditory knowledge to answer questions that traditional language models cannot solve accurately.

**Example 1**  
- **Sentence**: Rattle is the sound a [MASK] makes.  
- **Answer**: Snake  
- **BERT**: *Dog* ❌  
- **ITH**: ✅ *Snake*

**Example 2**  
- **Sentence**: The sound of chirping in the fall is often associated with a [MASK].  
- **Answer**: Cricket  
- **BERT**: *Bird* ❌  
- **ITH**: ✅ *Cricket*

These examples highlight how ITH can generate relevant audio (e.g., rattling, chirping) and use it to guide accurate predictions, where BERT fails due to lack of auditory grounding.

![image](https://github.com/user-attachments/assets/df05de85-b6a5-414a-a6f1-fb069bf09b9b)

👉 Want to hear the generated sounds? Check out more case studies on our [🔗 project page](https://imagine-to-hear.github.io).



## 📚 BibTeX

If you use this work in your research, please cite:

```bibtex
@article{yoo2025imagine,
  title={Imagine to Hear: Auditory Knowledge Generation can be an Effective Assistant for Language Models},
  author={Yoo, Suho and Ok, Hyunjong and Lee, Jaeho},
  journal={arXiv preprint arXiv:2503.16853},
  year={2025}
}
