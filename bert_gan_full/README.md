# 🧠 BERT-GAN: Malicious URL Detection Using Adversarial Learning

![Python](https://img.shields.io/badge/python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11+-orange)
![License](https://img.shields.io/badge/license-MIT-green)

This repository implements a deep learning architecture that combines **BERT**, **Bidirectional GRUs**, **Generative Adversarial Networks (GAN)**, and **Co-Attention Mechanisms** for robust malicious URL classification.

---

## 🔍 Project Highlights

- 🔡 **Contextual Embedding** via `DistilBERT`
- 🔁 **Sequential Modeling** using `Bi-GRU`
- ⚔️ **Adversarial Learning** with `GAN`
- 🎯 **Attention Fusion** via `Co-Attention Mechanism`
- 📈 **Visualization & Metrics** via Confusion Matrix, ROC, and PR curves

---

## 📁 Folder Structure

```
bert_gan_full/
├── models/               # Generator, Discriminator, Encoder, Co-Attention
├── scripts/              # train.py, evaluate.py for CLI use
├── utils/                # tokenizers, loaders (placeholder)
├── data/                 # CSV input data
├── notebooks/            # Demo and Evaluation notebooks
├── bert_gan_full_model_FINAL.py  # Monolithic model build
├── requirements.txt
└── README.md
```

---

## 📦 Installation

```bash
git clone https://github.com/Milugo/bert-gan-url-classifier.git
cd bert-gan-url-classifier
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### ▶️ Option 1: Run Script

```bash
python bert_gan_full_model_FINAL.py
```

### 💻 Option 2: Jupyter Notebook

Open and run:
```bash
notebooks/bert_gan_demo.ipynb
```

---

## 📊 Evaluation

Use the notebook:
```
notebooks/bert_gan_eval_and_visuals.ipynb
```

Includes:
- Accuracy, F1, AUC
- Confusion Matrix
- ROC & PR Curves

---

## 🧪 Sample Data Format

```
url,label
http://example.com/bad,0
http://example.com/good,1
```

---

## 🤝 Contributing

Want to contribute? Start by opening an issue or submitting a pull request!

---

## 📜 License

This project is licensed under the MIT License.