import os
import sys
import json
import datetime
import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── PATH HACK ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ───────────────

from App.model.go_emotions import GoEmotionsClassifier
from App.Streamlit.emotion_chart import emotion_bar_chart

st.set_page_config(page_title="Mental Distress App", layout="wide")

# CSV for alerts
ALERTS_CSV = os.path.join(ROOT, "alerts.csv")
if not os.path.isfile(ALERTS_CSV):
    pd.DataFrame(columns=["timestamp","level","text","preds"]).to_csv(ALERTS_CSV, index=False)

def save_alert(text, preds):
    severity_map = {"no_distress":0,"mild":1,"moderate":2,"severe":3}
    # top = highest-probability label
    top = max(preds, key=lambda p: p["score"])
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "level": top["label"],
        "text": text,
        "preds": json.dumps(preds, ensure_ascii=False)
    }
    pd.DataFrame([record]).to_csv(ALERTS_CSV, mode="a", index=False, header=False)

# Auth stub
if "role" not in st.session_state:
    with st.sidebar:
        st.title("Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u=="admin" and p=="admin":
                st.session_state.role="admin"; st.success("Logged in as admin")
            elif u=="user" and p=="user":
                st.session_state.role="user"; st.success("Logged in as user")
            else:
                st.error("Invalid credentials")
    st.stop()

role = st.session_state.role

@st.cache_resource
def load_models():
    predef = GoEmotionsClassifier()
    ft_path = os.path.join(ROOT, "App", "model", "distress_classifier")
    tok = AutoTokenizer.from_pretrained(ft_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(ft_path)
    return predef, tok, mdl

predef_clf, tokenizer_ft, model_ft = load_models()

distress_label_map = {0:"no_distress",1:"mild",2:"moderate",3:"severe"}

def predict_finetuned(text, top_k=4):
    inputs  = tokenizer_ft(text, return_tensors="pt", truncation=True)
    outputs = model_ft(**inputs)
    probs   = F.softmax(outputs.logits, dim=1)
    k       = min(top_k, probs.size(1))
    top_probs, top_labels = torch.topk(probs, k, dim=1)
    return [
        {"label": distress_label_map[int(i)], "score": float(s)}
        for i, s in zip(top_labels[0], top_probs[0])
    ]

def predict_predefined(text, top_k=5):
    return predef_clf.predict(text, top_k)

page = (
    st.sidebar.selectbox("Admin Panel", ["Model Test","Alerts List","Alerts Dashboard"])
    if role=="admin"
    else st.sidebar.selectbox("Menu", ["Feed"])
)

# ─── Feed (User) ───
if page=="Feed":
    st.title("📢 Social Feed")
    # initialize history
    if "posts" not in st.session_state:
        st.session_state.posts = []

    with st.form("post_form", clear_on_submit=True):
        text = st.text_area("What's on your mind?", height=120)
        if st.form_submit_button("Post") and text:
            preds = predict_finetuned(text)
            # look only at the top label
            top = max(preds, key=lambda p: p["score"])
            flagged = top["label"] in {"moderate","severe"}
            # save post history
            st.session_state.posts.append({
                "text": text,
                "flagged": flagged
            })
            if flagged:
                st.error("🚨 Need help?")
                save_alert(text, preds)

    # show all past posts, newest first
    st.markdown("---")
    for post in reversed(st.session_state.posts):
        if post["flagged"]:
            st.error(post["text"])
        else:
            st.write(post["text"])

# ─── Model Test (Admin) ───
elif page=="Model Test":
    st.title("🛠️ Model Testing")
    txt = st.text_area("Enter text to analyze", height=120)
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Test Predefined"):
            st.plotly_chart(emotion_bar_chart(predict_predefined(txt)), use_container_width=True)
    with c2:
        if st.button("Test Fine-Tuned"):
            st.plotly_chart(emotion_bar_chart(predict_finetuned(txt)), use_container_width=True)
    if st.button("Show Detailed Report"):
        st.table(pd.DataFrame(predict_finetuned(txt)))

# ─── Alerts List (Admin) ───
elif page=="Alerts List":
    st.title("🚨 Alerts List")
    df = pd.read_csv(ALERTS_CSV)
    if df.empty:
        st.info("No alerts yet.")
    else:
        df["preds"] = df["preds"].apply(json.loads)
        # level filter
        levels = sorted(df["level"].unique())
        lv = st.multiselect("Filter by level", levels, default=levels)
        # date filter
        df["dt"] = pd.to_datetime(df["timestamp"])
        mn, mx = df["dt"].dt.date.min(), df["dt"].dt.date.max()
        start, end = st.date_input("Date range", [mn,mx])
        mask = df["level"].isin(lv) & df["dt"].dt.date.between(start,end)
        st.dataframe(df.loc[mask, ["timestamp","level","text"]])

# ─── Alerts Dashboard (Admin) ───
elif page=="Alerts Dashboard":
    st.title("📊 Alerts Dashboard")
    df = pd.read_csv(ALERTS_CSV)
    if df.empty:
        st.info("No alerts.")
    else:
        df["preds"] = df["preds"].apply(json.loads)
        df["dt"] = pd.to_datetime(df["timestamp"])
        sev_map = {"no_distress":0,"mild":1,"moderate":2,"severe":3}
        df["sev"] = df["preds"].apply(lambda ps: max(sev_map[p["label"]] for p in ps))
        st.bar_chart(df["sev"].value_counts().sort_index())
        st.markdown("### All Alerts")
        st.dataframe(df[["timestamp","level","text"]])

