import os
import requests

RAW_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(OUT_DIR, "sms_spam_no_header.csv")


def download(url: str = RAW_URL, out_path: str = OUT_PATH) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


if __name__ == "__main__":
    print("Downloading dataset to:", OUT_PATH)
    path = download()
    print("Downloaded:", path)
