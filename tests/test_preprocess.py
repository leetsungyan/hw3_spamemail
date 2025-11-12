from src.preprocess import normalize_text, prepare_dataframe
import pandas as pd


def test_normalize_text():
    s = "  Hello WORLD!!\n"
    assert normalize_text(s) == "hello world!!"


def test_prepare_dataframe():
    df = pd.DataFrame({"label": ["ham", "spam"], "text": ["Hi there", None]})
    out = prepare_dataframe(df)
    assert out.shape[0] == 2
    assert out["label"].tolist() == [0, 1]
    assert isinstance(out["text"].iloc[1], str)
