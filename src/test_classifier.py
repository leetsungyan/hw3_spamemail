#!/usr/bin/env python
"""Quick test script for spam classifier."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.inference import load_model, predict_text
import os

MODEL_PATH = os.path.join(root, "models", "svm_spam.joblib")

def test_samples():
    model = load_model(MODEL_PATH)
    
    samples = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts",
        "Hi John, can you review the document I sent?",
        "Congratulations! You have won a million dollars",
        "Meeting at 3pm tomorrow, please confirm",
    ]
    
    print("📊 Testing Spam Classifier with sample emails:")
    print("=" * 70)
    
    for i, text in enumerate(samples, 1):
        result = predict_text(model, text)
        emoji = "🚨" if result["label"] == "spam" else "✅"
        print(f"\n{i}. {emoji} Result: {result['label'].upper()} (confidence: {result['confidence']:.2%})")
        display_text = text[:60] + "..." if len(text) > 60 else text
        print(f"   Email: {display_text}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_samples()
