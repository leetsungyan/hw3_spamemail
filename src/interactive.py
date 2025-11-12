#!/usr/bin/env python
"""Interactive spam email classifier - user-friendly interface."""

import os
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.inference import load_model, predict_text

MODEL_PATH = os.path.join(root, "models", "svm_spam.joblib")


def main():
    print("=" * 60)
    print("🚀 Spam Email Classifier (Interactive)")
    print("=" * 60)
    print()
    
    # Load model once
    print("📦 Loading model...", end=" ", flush=True)
    try:
        model = load_model(MODEL_PATH)
        print("✓ Done!")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    print()
    print("💡 Enter email text to classify (type 'exit' or 'quit' to exit)")
    print("-" * 60)
    print()
    
    count = 0
    while True:
        try:
            count += 1
            print(f"\n📧 Email #{count}:")
            text = input("> ").strip()
            
            if text.lower() in ["exit", "quit"]:
                print("\n👋 Goodbye!")
                break
            
            if not text:
                print("⚠️  Please enter some text.")
                count -= 1
                continue
            
            result = predict_text(model, text)
            label = result["label"].upper()
            confidence = result["confidence"]
            
            # Display result with emoji
            emoji = "🚨" if label == "SPAM" else "✅"
            print(f"\n{emoji} Classification: {label}")
            print(f"   Confidence: {confidence:.2%}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
