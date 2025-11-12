## 1. Implementation
- [ ] 1.1 撰寫資料下載/檢查腳本 (`scripts/download_data.py`)
- [ ] 1.2 實作前處理模組 (`src/preprocess.py`)：清理、Tokenize、TF-IDF
- [ ] 1.3 實作訓練腳本 (`src/train.py`)：sklearn Pipeline + GridSearchCV 訓練 SVM
- [ ] 1.4 撰寫評估腳本 (`src/evaluate.py`)：計算 Accuracy/Precision/Recall/F1
- [ ] 1.5 序列化模型與 artifact (`models/svm_spam.joblib`)
- [ ] 1.6 建立推論 wrapper (`src/inference.py`)：輸入文字 → 回傳 label 與 confidence
- [ ] 1.7 新增單元測試與集成測試 (`tests/`)
- [ ] 1.8 撰寫 README 與使用說明
- [ ] 1.9 （可選）加入 CI 檔案，執行 pytest 與 lint

## 2. Validation
- [ ] 2.1 使用 `openspec validate add-svm-spam-classifier --strict`（開發者運行）
- [ ] 2.2 PR 審查並取得批准後才開始實作（依 OpenSpec 流程）
