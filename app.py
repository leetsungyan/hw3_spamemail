"""
Streamlit app for SVM Spam Email Classifier
Provides a user-friendly web interface for spam classification
"""

import streamlit as st
import sys
from pathlib import Path
import os

# Add project root to path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from src.classify_email import classify

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
