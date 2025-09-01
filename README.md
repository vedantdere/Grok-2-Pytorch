# 🚀 Grok-2 – Pure PyTorch Reimplementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15g8I1cXcC1p2wvjQst8GPlWV2bEgD4Ke?usp=sharing)


<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" />
</p>

<p align="center">
    <img src="./assets/img1.png">
    (img credit ChatGPT)
</p>


**Grok-2**, built entirely in **PyTorch** — a beginner-friendly implementation designed to enhance model understanding for future AI researchers. 

Without Inheriting from anything other than `nn.Module` 😁



---

## ✨ Features

* **Pure PyTorch** – No heavy dependencies beyond PyTorch & Hugging Face utilities.
* **RMSNorm** and **Rotary Position Embeddings**.
* **Custom Attention** – supports eager.
* **Easy to Modify** – great for research and teaching.
* In **Pytorch**
---


## 📦 Installation

```bash
git https://github.com/vedantdere/Grok-2-Pytorch.git
cd Grok-2-Pytorch
```

---
## Model Configurations

We provide two predefined model configurations:

1. **Grok2Config** – original config
2. **Grok2ConfigTiny** - Config defined to run on cpu.

---

## 🚀 Quick Start

```python
import torch
from config import Grok2ConfigTiny
from model import Grok1ModelForCausalLM  # your main file

# 1. Load Config
config = Grok2ConfigTiny()

# 2. Load Model
model = Grok1ModelForCausalLM(config)

# 3. Define Inputs
batch_size = 1
seq_len = 5 # for small model testing
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)  # shape: [batch_size, seq_len]

# 4. Run forward pass
logits = model(input_ids=input_ids,positions=position_ids)

print("Input IDs shape:", input_ids.shape)
print("Logits shape:", logits.shape)  # (batch_size, seq_len, vocab_size)
```

---

## 🧠 Key Design Choices

* **Modular** – Every component (attention, MLP, norm, router) is a standalone class.
* **Readable** – Code avoids excessive abstraction for clarity.
* **Custom Attention Backend** – Can integrate with flash attention or other kernels.

---

## 📊 Simple Training Script

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss()

for step in range(100):
    logits = model(input_ids=input_ids)
    loss = loss_fn(logits.view(-1, config.vocab_size), input_ids.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {step}, Loss: {loss.item():.4f}")
```

----

## 🙏 Credits
Huge thanks to the Grok team and Elon Musk for open-sourcing Grok-2 🙌
This reimplementation draws major inspiration from sglang and Hugging Face Transformers 🚀

## 🙌 Inspiration

This work draws significant inspiration from Andrej Karpathy’s minGPT.


---

## 🤝 Contributing

Pull requests, bug reports, and feature suggestions are welcome!
If you improve the architecture or add new routing strategies, please share them with the community.

---

## 🌟 Star the Repo

If you find this project useful, please ⭐ it on GitHub – it helps more people discover it.