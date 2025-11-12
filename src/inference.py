import os
import joblib
import math
from typing import Dict

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "svm_spam.joblib")


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def decision_to_confidence(score: float) -> float:
    # Map SVM decision function to 0..1 via sigmoid
    try:
        return 1.0 / (1.0 + math.exp(-abs(score)))
    except OverflowError:
        return 1.0 if score > 0 else 0.0


def predict_text(model, text: str) -> Dict:
    label_num = model.predict([text])[0]
    # decision function returns distance; for linearSVC it may be a single value
    try:
        score = model.decision_function([text])[0]
        if isinstance(score, (list, tuple)):
            # multiclass, take max abs
            score = float(max(score, key=abs))
    except Exception:
        score = 0.0
    conf = float(decision_to_confidence(score))
    label = "spam" if int(label_num) == 1 else "ham"
    return {"label": label, "confidence": conf}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model path", default=MODEL_PATH)
    parser.add_argument("--text", help="text to classify", required=True)
    args = parser.parse_args()
    m = load_model(args.model)
    print(predict_text(m, args.text))
