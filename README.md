# Spam Detector API

This repository contains utilities for training spam classification models.

## LSTM Training

The new `src/model/lstm_train.py` script trains an LSTM based model using
TensorFlow/Keras. It reads a CSV file with columns `label` and `text`, converts
texts to sequences using Keras `Tokenizer`, and saves the trained model along
with the tokenizer.

Usage example:

```bash
python src/model/lstm_train.py
```

The script will train separate models for the English and Chinese datasets and
store them as `model_en_lstm.h5` and `model_cn_lstm.h5`.
