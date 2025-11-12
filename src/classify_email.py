#!/usr/bin/env python
"""
Simple Spam Email Classifier - can be imported and used directly.

Usage as script:
    python classify_email.py

Usage as module:
    from src.classify_email import classify
    result = classify("Your email text here")
    print(result)
"""

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.inference import load_model, predict_text

MODEL_PATH = os.path.join(root, "models", "svm_spam.joblib")
_model = None


def classify(email_text: str):
    """
    Classify an email as spam or ham.
    
    Args:
        email_text (str): The email content to classify
        
    Returns:
        dict: {'label': 'spam' or 'ham', 'confidence': float}
    """
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return predict_text(_model, email_text)


def main():
    """Interactive mode."""
    global _model
    
    print("=" * 70)
    print("🚀 Spam Email Classifier")
    print("=" * 70)
    print()
    
    print("📦 Loading model...", end=" ", flush=True)
    try:
        _model = load_model(MODEL_PATH)
        print("✓")
    except Exception as e:
        print(f"✗\nError: {e}")
        return
    
    print("\n💡 Enter email text (or 'q' to quit):")
    print("-" * 70)
    
    count = 0
    while True:
        count += 1
        print(f"\n📧 Email #{count}:")
        text = input("> ").strip()
        
        if text.lower() in ["q", "quit", "exit"]:
            print("\n👋 Goodbye!")
            break
        
        if not text:
            print("⚠️  Please enter some text.")
            count -= 1
            continue
        
        try:
            result = classify(text)
            label = result["label"].upper()
            confidence = result["confidence"]
            emoji = "🚨" if label == "SPAM" else "✅"
            print(f"\n{emoji} Result: {label} (Confidence: {confidence:.2%})")
        except Exception as e:
            print(f"❌ Error: {e}")
            count -= 1


if __name__ == "__main__":
    main()
