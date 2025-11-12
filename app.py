"""
Streamlit app for SVM Spam Email Classifier
Provides a user-friendly web interface for spam classification
"""

import streamlit as st
import sys
from pathlib import Path
import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Add project root to path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from src.classify_email import classify

# Import helpers (model & preprocess). Wrap in try/except so we can show a friendly
# error message if these imports fail in the deployment environment.
import_exception = None
try:
    from src.inference import load_model
    from src.preprocess import load_data, prepare_dataframe
except Exception as e:
    load_model = None
    load_data = None
    prepare_dataframe = None
    import_exception = e

# Page config
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTextArea > div > div > textarea {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚨 Spam Email Classifier")
st.markdown("---")
st.markdown("""
使用機器學習 (SVM) 快速判斷郵件是否為垃圾訊息
""")

# Sidebar info
with st.sidebar:
    st.header("📊 關於此應用")
    st.info("""
    **技術棧:**
    - 模型: Support Vector Machine (SVM)
    - 特徵化: TF-IDF Vectorizer
    - 訓練資料: SMS Spam Collection Dataset
    
    **性能指標:**
    - Accuracy: 98.83%
    - Precision: 97.89%
    - Recall: 93.29%
    - F1 Score: 95.53%
    """)
    
    st.markdown("---")
    st.markdown("""
    **使用說明:**
    1. 在下方文字框輸入郵件內容
    2. 點擊「分類」按鈕
    3. 查看分類結果與信心度
    """)
    
    st.markdown("---")
    st.markdown("[GitHub 倉庫](https://github.com/leetsungyan/hw3_spamemail)")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📧 輸入郵件內容")
    email_text = st.text_area(
        "郵件文本:",
        placeholder="請輸入郵件內容...",
        height=200,
        label_visibility="collapsed"
    )

with col2:
    st.subheader("分類結果")
    
    # Classification button
    if st.button("🔍 分類", use_container_width=True, type="primary"):
        if not email_text.strip():
            st.error("❌ 請輸入郵件內容")
        else:
            with st.spinner("⏳ 分類中..."):
                try:
                    result = classify(email_text)
                    label = result["label"]
                    confidence = result["confidence"]
                    
                    # Display result
                    if label == "spam":
                        st.error(f"🚨 **垃圾郵件**", icon="🚨")
                        st.metric(
                            label="信心度",
                            value=f"{confidence:.2%}",
                            delta="高風險" if confidence > 0.7 else "中風險" if confidence > 0.5 else "低風險"
                        )
                    else:
                        st.success(f"✅ **正常郵件**", icon="✅")
                        st.metric(
                            label="信心度",
                            value=f"{confidence:.2%}",
                            delta="安全" if confidence > 0.7 else "中等" if confidence > 0.5 else "較低"
                        )
                    
                    # Show more details
                    with st.expander("📋 詳細資訊"):
                        st.write(f"**分類結果:** {label.upper()}")
                        st.write(f"**信心度:** {confidence:.4f}")
                        st.progress(confidence)
                    
                except Exception as e:
                    st.error(f"❌ 發生錯誤: {str(e)}")

# Example section
st.markdown("---")
st.subheader("📝 範例郵件")

example_col1, example_col2 = st.columns(2)

with example_col1:
    st.markdown("#### 🚨 垃圾郵件範例")
    spam_examples = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005",
        "Congratulations! You have won a million dollars. Click here to claim",
        "URGENT: Your account has been compromised. Verify identity NOW!"
    ]
    for i, example in enumerate(spam_examples):
        if st.button(f"範例 {i+1}", key=f"spam_{i}"):
            st.session_state.email_text = example

with example_col2:
    st.markdown("#### ✅ 正常郵件範例")
    ham_examples = [
        "Hi John, can you review the document I sent?",
        "Meeting at 3pm tomorrow, please confirm",
        "Thank you for your order. Your package will arrive tomorrow"
    ]
    for i, example in enumerate(ham_examples):
        if st.button(f"範例 {i+1}", key=f"ham_{i}"):
            st.session_state.email_text = example

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; margin-top: 2rem;">
    <small>Spam Email Classifier v1.0 | Powered by Streamlit</small>
</div>
""", unsafe_allow_html=True)

# --- Analysis section -----------------------------------------------------
@st.cache_data
def ensure_dataset(csv_path: str):
    # If dataset not present, download from raw GitHub URL
    RAW_URL = (
        "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
    )
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        r = requests.get(RAW_URL, timeout=30)
        r.raise_for_status()
        with open(csv_path, "wb") as f:
            f.write(r.content)
    return csv_path


@st.cache_data
def load_dataset_and_model():
    csv_path = os.path.join(root, "data", "sms_spam_no_header.csv")
    ensure_dataset(csv_path)
    if import_exception is not None:
        # Raise a clear error so Streamlit can show it in logs
        raise RuntimeError(f"Failed to import project helpers: {import_exception}")
    df = load_data(csv_path)
    df = prepare_dataframe(df)
    X = df["text"].values
    y = df["label"].values
    model = load_model(os.path.join(root, "models", "svm_spam.joblib"))
    return df, X, y, model


def get_class_distribution(df: pd.DataFrame):
    counts = df["label"].map({0: "ham", 1: "spam"}).value_counts()
    return counts


def get_top_tokens(model, top_n: int = 20):
    # Expect a sklearn Pipeline with 'tfidf' and 'clf'
    try:
        vect = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
    except Exception:
        return None
    feature_names = vect.get_feature_names_out()
    coefs = clf.coef_[0]
    top_positive_idx = np.argsort(coefs)[-top_n:][::-1]
    top_negative_idx = np.argsort(coefs)[:top_n]
    top_pos = [(feature_names[i], float(coefs[i])) for i in top_positive_idx]
    top_neg = [(feature_names[i], float(coefs[i])) for i in top_negative_idx]
    return top_pos, top_neg


def compute_performance(model, X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }
    # decision scores for threshold sweep
    try:
        scores = model.decision_function(X_test)
        if scores.ndim > 1:
            # multiclass: take column for positive class
            scores = scores[:, 1]
    except Exception:
        scores = None
    return metrics, y_test, scores


def threshold_sweep(y_true, scores, num=50):
    thresholds = np.linspace(np.min(scores), np.max(scores), num=num)
    rows = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        rows.append({"threshold": float(t), "precision": p, "recall": r, "f1": f})
    return pd.DataFrame(rows)


with st.expander("🔎 Model analysis & diagnostics", expanded=False):
    df, X, y, model = load_dataset_and_model()
    st.subheader("Class distribution")
    distr = get_class_distribution(df)
    st.bar_chart(distr)

    st.subheader("Top tokens by class (SVM coefficients)")
    top = get_top_tokens(model, top_n=20)
    if top is not None:
        top_pos, top_neg = top
        pos_df = pd.DataFrame(top_pos, columns=["token", "weight"])
        neg_df = pd.DataFrame(top_neg, columns=["token", "weight"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top tokens (spam)**")
            st.table(pos_df)
        with c2:
            st.markdown("**Top tokens (ham)**")
            st.table(neg_df)
    else:
        st.write("Model does not expose coefficients (unable to show tokens).")

    st.subheader("Model performance")
    metrics, y_test, scores = compute_performance(model, X, y)
    st.write(metrics)
    st.table(pd.DataFrame([metrics]).T.rename(columns={0: "value"}))

    if scores is not None:
        st.subheader("Threshold sweep (precision / recall / f1)")
        sweep_df = threshold_sweep(y_test, scores, num=50)
        st.line_chart(sweep_df.set_index("threshold")["precision"])
        st.line_chart(sweep_df.set_index("threshold")["recall"])
        st.line_chart(sweep_df.set_index("threshold")["f1"])
        with st.expander("Threshold table"):
            st.dataframe(sweep_df)
    else:
        st.write("Decision scores not available for threshold sweep.")
