import os
import json
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from src.preprocess import load_data, prepare_dataframe, save_metadata


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "svm_spam.joblib")
DEFAULT_META_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "metadata.json")


def train(csv_path: str, model_path: str = DEFAULT_MODEL_PATH, meta_path: str = DEFAULT_META_PATH, random_state: int = 42) -> dict:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    df = load_data(csv_path)
    df = prepare_dataframe(df)
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
        ("clf", LinearSVC(class_weight="balanced", random_state=random_state)),
    ])

    pipeline.fit(X_train, y_train)

    # evaluation
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    joblib.dump(pipeline, model_path)
    save_metadata(meta_path, {"metrics": metrics, "random_state": random_state})

    return {"model_path": model_path, "meta_path": meta_path, "metrics": metrics}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="path to sms csv")
    parser.add_argument("--out", help="model out path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    print("Training with:", args.csv)
    res = train(args.csv, model_path=args.out)
    print(json.dumps(res, indent=2))
