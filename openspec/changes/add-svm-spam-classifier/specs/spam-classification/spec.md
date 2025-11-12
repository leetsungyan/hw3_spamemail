## ADDED Requirements

### Requirement: SVM-based Spam Classification
The system SHALL provide a training pipeline that produces a SVM-based spam classifier for short-text messages (SMS).

#### Scenario: Training completes successfully
- **WHEN** the CSV dataset (sms_spam_no_header.csv) is available and the training script is executed with default config
- **THEN** the system shall produce a serialized model file `models/svm_spam.joblib` and a `models/metadata.json` containing training config and metrics
- **AND** the script shall output evaluation metrics including Accuracy, Precision, Recall, and F1

#### Scenario: Inference returns label and confidence
- **WHEN** a single message text is provided to the inference wrapper
- **THEN** the system SHALL return a JSON object with keys `label` ("spam" or "ham") and `confidence` (a float 0..1) representing model confidence

#### Scenario: Evaluation meets baseline
- **WHEN** the model is evaluated on a hold-out test set
- **THEN** the model SHALL achieve an F1-score no lower than 0.80 (project baseline) OR the evaluation output SHALL record the achieved metrics for review

