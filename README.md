# SVM Spam Classification

Minimal project to train an SVM-based spam classifier for SMS-like messages.

Usage (quick):

1. Create a virtual environment and install deps:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt
```

2. Download data:

```powershell
python .\scripts\download_data.py
```

3. Train:

```powershell
python -m src.train .\data\sms_spam_no_header.csv
```

4. Infer:

```powershell
python -m src.inference --text "Free entry in 2 a wkly comp to win FA Cup" --model .\models\svm_spam.joblib
```
