# 🚀 TinyStories LLM — Building a Small Language Model from Scratch

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-%23007ACC?style=for-the-badge&logo=ai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Transformer%20Architecture-%23FFB6C1?style=for-the-badge"/>
</p>

## 🧠 Overview  
This project implements a **miniature Large Language Model (LLM)** **from scratch**, trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.  
The goal is **not** to build another language model — but to **understand every moving part**: tokenization, attention, optimization, and text generation.

It’s a ground-up implementation inspired by GPT-style architectures, written in PyTorch for the purpose of learning.

---

## ✨ Key Features
- 🧩 **Transformer Architecture** — Implements multi-head self-attention, positional encodings, and layer normalization manually.  
- 🏋️ **Training on TinyStories** — Trained from scratch on a child-friendly corpus for lightweight yet meaningful language generation.  
- 🧪 **Sampling & Text Generation** — Greedy, temperature, and top-k sampling implemented natively.  
- 📈 **Loss Visualization** — Real-time tracking of training and validation losses.  
- 🧰 **Fully Reproducible** — Minimal dependencies, clear code structure, and easily extendable.

---

## 🧬 Model Architecture
| Component | Description |
|------------|-------------|
| Embedding Layer | Token + positional embeddings |
| Transformer Blocks | Multi-Head Attention + MLP + LayerNorm |
| Context Length | Configurable |
| Parameters | ~4.8M (configurable) |
| Framework | PyTorch |

---

## 🧑‍💻 Implementation Highlights
- **No external model libraries** — all attention, normalization, and weight initialization coded manually.  
- **Trains on CPU or single GPU** — making it accessible to anyone with modest hardware.  

---

## 📊 Training Performance
| Metric | Value |
|--------|-------|
| Dataset | TinyStories (≈ 1K sentences) |
| Best Loss | ~1.4 (validation) |

---


## 🧠 Lessons Learned
- Understood **transformer internals** and **scaling laws** through hands-on experimentation.  
- Gained intuition about **context windows**, **attention bottlenecks**, and **training stability**.   



## 🏗️ Project Structure
```
LLM
├── Code/
│   ├── preprocess.ipynb          # Notebook for preparing the dataset
│   ├── train_test.ipynb          # Notebook for training and evaluating the model
│   └── train_test.py             # Script version of training and evaluation pipeline
├── Dataset/
│   └── 0000.parquet              # First part of the raw TinyStories dataset used for training
│   └── text.txt                  # parquet file processed to plain text.
├── SavedModel/
│   └── model.pth                 # Directory containing trained model weights
├── convergence.png               # Loss curves (training and validation)
└── README.md
└── LICENSE
``` 

---

## 📚 References 
- Karpathy, *Let's build GPT: from scratch, in code, spelled out.* 


---

## 👨‍💻 Author
**Saeed Mohseni**  
Graduate Researcher, Institute for Advanced Computing  
Virginia Tech, VA, USA  

🌐 [Website](https://saeedmohseni.netlify.app/) | 📫 saeedmohseni@vt.edu  

---

## 🌟 If you like this project...
⭐ **Star** the repository  
🍴 **Fork** it  
🧠 **Discuss** ideas or improvements  
