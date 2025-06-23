import streamlit as st
from transformers import pipeline

# Load model (you can replace this with your fine-tuned model later)
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

classifier = load_model()

# App layout
st.set_page_config(page_title="Mental Distress Detector", layout="centered")

st.title("🧠 Mental Distress Detection")
st.markdown("Enter a message and this app will detect potential emotional distress.")

# Input
user_input = st.text_area("📝 Message", placeholder="Paste a message here...", height=150)

if st.button("🔍 Analyze (Predifined Model)"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            results = classifier(user_input)[0]
            results = sorted(results, key=lambda x: x["score"], reverse=True)

        # Show predictions with bars
        st.subheader("Emotion Scores:")
        for res in results:
            st.write(f"**{res['label']}**")
            st.progress(res["score"])
    else:
        st.warning("Please enter some text.")
