# Proposal: add-svm-spam-classifier

## Why
建立一個 SVM 為核心的垃圾簡訊（spam）分類能力，提供從資料擷取、前處理、訓練、評估到模型匯出的完整可重現流程，方便日後部署與驗證。

## What Changes
- 新增 capability `spam-classification`，包含 SVM-based 訓練與推論流程。
- 新增訓練腳本、前處理模組、推論包裝、與對應測試。
- 新增規格增量與實作任務清單。

**BREAKING:** 無（向後相容）。

## Impact
- Affected specs: `openspec/changes/add-svm-spam-classifier/specs/spam-classification/spec.md`
- Affected code (suggested): `src/preprocess.py`, `src/train.py`, `src/inference.py`, `models/`
- Tests: 新增單元與集成測試於 `tests/`。
- Deployment: 產出可序列化的模型檔（`models/svm_spam.joblib`）。
