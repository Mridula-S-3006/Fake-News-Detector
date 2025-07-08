import streamlit as st
from transformers import pipeline

# Set Streamlit page config
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Title and description
st.title("📰 Fake News Detector")
st.markdown("Classify news as **real or fake** using a pretrained transformer model.")

# Load pretrained model from Hugging Face
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news")

model = load_model()

# Text input
user_input = st.text_area("✍️ Enter news headline or article here:", height=200)

# Analyze button
if st.button("🚀 Analyze"):
    if not user_input.strip():
        st.warning("Please enter some news content.")
    else:
        with st.spinner("Thinking... 🤔"):
            prediction = model(user_input)[0]
            label = prediction["label"]
            confidence = prediction["score"]

            # Display result
            if "FAKE" in label.upper():
                st.error(f"🛑 This seems **FAKE** ({confidence:.2%} confidence)")
            else:
                st.success(f"✅ This seems **REAL** ({confidence:.2%} confidence)")
