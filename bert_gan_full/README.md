# BERT-GAN for Malicious URL Detection

This project implements a BERT-GAN architecture to classify URLs as malicious or benign. It combines:

- Contextual Embeddings via DistilBERT
- Sequential Modeling via Bi-GRU
- GAN: Generator + Discriminator for adversarial learning
- Co-Attention mechanism to merge real/synthetic features
- Classification layer for binary prediction

## How to Run

1. Prepare `data.csv` inside the `data/` directory.
2. Run the Python script:

```bash
python bert_gan_full_model_FINAL.py
```