import re
import json
from typing import Tuple
import pandas as pd


def load_data(csv_path: str, encoding: str = "utf-8") -> pd.DataFrame:
    # CSV has no header: first column label, second column text
    df = pd.read_csv(csv_path, header=None, encoding=encoding)
    df = df.rename(columns={0: "label", 1: "text"})
    return df


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # remove non-printable
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    # simple lower
    text = text.lower()
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].fillna("").apply(normalize_text)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def save_metadata(path: str, meta: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
