import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


def train_lstm_model(
    data_path: str,
    model_path: str,
    tokenizer_path: str,
    num_words: int = 5000,
    max_len: int = 100,
    epochs: int = 5,
    batch_size: int = 32,
) -> None:
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].apply(lambda x: 1 if x in ["spam", "垃圾短信"] else 0).values

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = Sequential(
        [
            Embedding(num_words, 128, input_length=max_len),
            LSTM(64),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", acc)

    model.save(model_path)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)


if __name__ == "__main__":
    train_lstm_model(
        "../../data/en_balanced.csv",
        "model_en_lstm.h5",
        "tokenizer_en.pkl",
    )
    train_lstm_model(
        "../../data/cn_balanced.csv",
        "model_cn_lstm.h5",
        "tokenizer_cn.pkl",
    )
