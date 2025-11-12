## Design: SVM spam-classification (minimal)

### Context
- 資料為短文本 SMS，目標是二元分類（spam vs ham）。

### Goals
- 可重現的訓練流程
- 可序列化的模型 artifact
- 簡單、可解釋的特徵管線（TF-IDF + SVM）

### Decisions
- 特徵化：使用 `TfidfVectorizer`（unigram + bigram，可設定 max_features）
- 分類器：優先使用 `LinearSVC`（速度較快、效果穩定）；必要時改用 `SVC(kernel='rbf')` 並啟用 `probability=True`
- 超參數調整：`GridSearchCV` 或 `RandomizedSearchCV`，以 F1-score 為主要指標
- 訓練/驗證切分：80%/20%（固定 random_state）
- 模型序列化：`joblib.dump` 到 `models/svm_spam.joblib`

### Risks / Mitigations
- 稀有類別（class imbalance）：若資料不平衡，使用 class_weight='balanced' 或重採樣
- 模型體積：優先選擇 `LinearSVC` 以節省空間

### Migration / Reproducibility
- 所有隨機種子需設定（np.random.seed & sklearn random_state）
- 保存訓練配置與分割種子到 `models/metadata.json`
