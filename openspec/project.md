# Project Context

## Purpose
建立一個以支援向量機（SVM）為核心的垃圾訊息（spam）分類專案，使用公開的 SMS/SMS-like 資料集訓練模型，提供可重現的資料前處理、訓練、評估與模型匯出流程，方便後續部署或延伸為 API/服務。

## Tech Stack
- 語言：Python 3.9+ / 3.10+（首選 3.10）
- 資料處理：pandas, numpy
- 特徵與文字處理：scikit-learn (TfidfVectorizer, pipeline), scipy
- 模型：scikit-learn SVM（SVC 或 LinearSVC）、GridSearchCV
- 模型序列化：joblib
- 測試：pytest
- 程式風格/格式化：black、isort、flake8
- 開發/依賴管理：venv 或 Poetry/requirements.txt

## Project Conventions

### Code Style
[Describe your code style preferences, formatting rules, and naming conventions]
- 使用 88 列或 100 列的 line-length（以 `pyproject.toml` 或 `.flake8` 為準）
- 格式化：`black` 自動格式化（在 pre-commit 中執行）
- 匯入排序：`isort`
- 變數與函式命名：snake_case；類名使用 PascalCase
- commit 信息：採用簡短、描述性的訊息，建議使用 Conventional Commits（type(scope): description）

### Architecture Patterns
[Document your architectural decisions and patterns]
- 小型專案採用簡單 ML pipeline 結構：
	- `data/`：原始與處理過的資料（.gitignore 大型資料）
	- `src/`：核心程式碼（`src/preprocess.py`、`src/train.py`、`src/serve.py` 等）
	- `models/`：匯出的訓練後模型（.joblib）
	- `notebooks/`：分析或實驗性 Notebook
	- `tests/`：pytest 測試
- 以可重用的函式與 sklearn Pipeline 為優先，保持單一責任原則。

### Testing Strategy
[Explain your testing approach and requirements]
- 單元測試：對資料前處理函式、特徵管線（pipeline）與模型介面做單元測試
- 集成測試：快速訓練流水線（使用小型樣本或 fixture）以檢查 end-to-end 流程
- 評估門檻：提供一組基準指標（Accuracy / Precision / Recall / F1）用於 CI 的簡易回歸檢查（可選）

### Git Workflow
[Describe your branching strategy and commit conventions]
- Branching：`main`（穩定）、`develop`（開發）、feature 分支 `feature/<short-desc>` 或 `add-<feature>`
- Pull Request：每個變更透過 PR 進行代碼審查與 CI 檢查
- Commit convention：建議使用 Conventional Commits

## Domain Context
[Add domain-specific knowledge that AI assistants need to understand]
- 使用的資料為短訊（SMS）spam 分類；來源為公開範例資料集（sms_spam_no_header.csv）
- 主要挑戰：文本的變化性、簡短語句導致上下文少、非結構化文字的雜訊

## Important Constraints
[List any technical, business, or regulatory constraints]
- 不儲存或傳送含個資的資料到外部服務（若資料包含 PII，需先進行去識別化）
- 模型大小與推論延遲：目標為輕量可部署（可接受將模型限制在幾 MB）
- 規格檔（spec）必須包含至少一個 `#### Scenario:`

## External Dependencies
[Document key external services, APIs, or systems]
- 訓練資料來源（Raw CSV）：https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- PyPI 套件：scikit-learn, pandas, numpy, joblib, pytest, black, isort

