import os
import sys
import json
import datetime
import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── PATH HACK ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ───────────────

from App.model.go_emotions import GoEmotionsClassifier
from App.Streamlit.emotion_chart import emotion_bar_chart

# Page config with custom theme
st.set_page_config(
    page_title="Friendbook - Mental Health Social Network", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤝"
)

# Custom CSS for Facebook-like styling
def load_css():
    st.markdown("""
    <style>
    /* Facebook-like colors for user pages */
    .user-theme {
        --primary-color: #1877f2;
        --secondary-color: #42a5f5;
        --accent-color: #e3f2fd;
        --text-color: #1c1e21;
        --bg-color: #f0f2f5;
    }
    
    /* Dark theme for admin */
    .admin-theme {
        --primary-color: #374151;
        --secondary-color: #6b7280;
        --accent-color: #1f2937;
        --text-color: #f9fafb;
        --bg-color: #111827;
    }
    
    /* User page styling */
    .user-page {
        background: linear-gradient(135deg, #e3f2fd 0%, #f0f2f5 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .post-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1877f2;
    }
    
    .distress-post {
        border-left: 4px solid #f44336;
        background: linear-gradient(to right, #ffebee, #ffffff);
    }
    
    .help-button {
        background: #1877f2;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 600;
    }
    
    /* Admin page styling */
    .admin-page {
        background: #111827;
        color: #f9fafb;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .admin-card {
        background: #1f2937;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #374151;
    }
    
    /* Metric card with top bar and border */
    .metric-card {
        background: #23272f;
        border-radius: 12px;
        border: 0px solid #1e2939;
        padding: 0;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    
    }
    .metric-card-bar {
        background: #1e2939;
        color: #fff;
        font-size: 1.0 em;
        font-weight: bold;
        padding: 10px 0 8px 0;
        letter-spacing: 0.5px;
    }

    .metric-card-value {
        background: #f3f4f6;
        color: #23272f;
        font-size: 2.5em;
        font-weight: bold;
        padding: 18px 0 14px 0;
    }
    
    /* Login page styling */
    .login-container {
        background: linear-gradient(135deg, #1877f2 0%, #42a5f5 100%);
        border-radius: 15px;
        padding: 30px;
        color: white;
    }
    
    .about-section {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,242,245,0.95) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .about-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url('https://images.pexels.com/photos/3760067/pexels-photo-3760067.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1');
        background-size: cover;
        background-position: center;
        opacity: 0.1;
        z-index: -1;
    }
    
    .about-content {
        position: relative;
        z-index: 1;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .feature-card {
        background: rgba(24, 119, 242, 0.1);
        border-radius: 10px;
        padding: 15px;
        border-left: 3px solid #1877f2;
    }
    
    .stats-row {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    
    .stat-item {
        text-align: center;
        margin: 10px;
    }
    
    .stat-number {
        font-size: 2em;
        font-weight: bold;
        color: #1877f2;
    }
    
    .stat-label {
        font-size: 0.9em;
        color: #65676b;
    }
    
    /* Navigation */
    .nav-header {
        background: #1877f2;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #65676b;
        font-size: 14px;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# CSV for alerts and help messages
ALERTS_CSV = os.path.join(ROOT, "alerts.csv")
HELP_MESSAGES_CSV = os.path.join(ROOT, "help_messages.csv")

if not os.path.isfile(ALERTS_CSV):
    pd.DataFrame(columns=["timestamp","level","text","preds"]).to_csv(ALERTS_CSV, index=False)

if not os.path.isfile(HELP_MESSAGES_CSV):
    pd.DataFrame(columns=["timestamp","user_id","message","feelings","help_type"]).to_csv(HELP_MESSAGES_CSV, index=False)

def save_alert(text, preds):
    severity_map = {"no_distress":0,"mild":1,"moderate":2,"severe":3}
    top = max(preds, key=lambda p: p["score"])
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "level": top["label"],
        "text": text,
        "preds": json.dumps(preds, ensure_ascii=False)
    }
    pd.DataFrame([record]).to_csv(ALERTS_CSV, mode="a", index=False, header=False)

def save_help_message(message, feelings, help_type="general"):
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "user_id": st.session_state.get("username", "anonymous"),
        "message": message,
        "feelings": feelings,
        "help_type": help_type
    }
    pd.DataFrame([record]).to_csv(HELP_MESSAGES_CSV, mode="a", index=False, header=False)

def create_about_section():
    return """
    <div class="about-section">
        <div class="about-content">
            <h1 style="text-align: center; color: #1877f2; margin-bottom: 30px;">
                🤝 Welcome to Friendbook
            </h1>
            <p style="text-align: center; font-size: 1.2em; color: #1c1e21; margin-bottom: 30px;">
                <strong>Your AI-Powered Mental Health Companion</strong>
            </p>
            
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-number">94.2%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">93.8%</div>
                    <div class="stat-label">F1-Score</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">24/7</div>
                    <div class="stat-label">Support</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">4</div>
                    <div class="stat-label">Distress Levels</div>
                </div>
            </div>
            
            <h3 style="color: #1877f2; margin-top: 25px;">🎯 Our Mission</h3>
            <p>Friendbook revolutionizes social media by putting mental health first. Our advanced AI analyzes posts in real-time to identify users who may need support, creating a safer and more caring online community.</p>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <h4>🧠 AI-Powered Detection</h4>
                    <p>Advanced NLP model trained on mental health data</p>
                </div>
                <div class="feature-card">
                    <h4>💙 Immediate Support</h4>
                    <p>Instant help when distress is detected</p>
                </div>
                <div class="feature-card">
                    <h4>🔒 Privacy First</h4>
                    <p>Your conversations remain confidential</p>
                </div>
                <div class="feature-card">
                    <h4>🤝 Community Care</h4>
                    <p>Connect with others who understand</p>
                </div>
            </div>
            
            <h3 style="color: #1877f2; margin-top: 25px;">🧠 AI Technology</h3>
            <p>Our model uses fine-tuned DistilBERT to classify emotional distress into four categories:</p>
            <ul style="margin-left: 20px;">
                <li><strong>No Distress:</strong> Normal, positive communication</li>
                <li><strong>Mild:</strong> Minor concerns or everyday stress</li>
                <li><strong>Moderate:</strong> Significant emotional distress</li>
                <li><strong>Severe:</strong> Critical situations requiring immediate help</li>
            </ul>
            
            <h3 style="color: #1877f2; margin-top: 25px;">🚨 Emergency Resources</h3>
            <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;">
                <p><strong>Crisis Hotlines:</strong></p>
                <ul>
                    <li><strong>US:</strong> 988 (Suicide & Crisis Lifeline)</li>
                    <li><strong>Text:</strong> HOME to 741741</li>
                    <li><strong>International:</strong> befrienders.org</li>
                </ul>
            </div>
            
            <p style="text-align: center; margin-top: 25px; font-style: italic; color: #65676b;">
                Remember: You're never alone, and it's okay to ask for help ❤️
            </p>
        </div>
    </div>
    """

# Enhanced Auth System
def show_login_page():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("### 🤝 Welcome to Friendbook")
        st.markdown("**Your Mental Health Social Network**")
        
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Login", use_container_width=True):
                if username == "admin" and password == "admin":
                    st.session_state.role = "admin"
                    st.session_state.username = username
                    st.success("✅ Logged in as admin")
                    st.rerun()
                elif username == "user" and password == "user":
                    st.session_state.role = "user"
                    st.session_state.username = username
                    st.success("✅ Logged in as user")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        with col_btn2:
            if st.button("📝 Sign Up", use_container_width=True):
                st.info("👋 Contact admin for new accounts")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.html(create_about_section()#, unsafe_allow_html=True
                )

# Model loading
@st.cache_resource
def load_models():
    predef = GoEmotionsClassifier()
    ft_path = os.path.join(ROOT, "App", "model", "distress_classifier")
    tok = AutoTokenizer.from_pretrained(ft_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(ft_path)
    return predef, tok, mdl

# Check authentication
if "role" not in st.session_state:
    show_login_page()
    st.stop()

# Initialize username if not set
if "username" not in st.session_state:
    st.session_state.username = "user" if st.session_state.role == "user" else "admin"

# Load models after authentication
predef_clf, tokenizer_ft, model_ft = load_models()

# Header with logout
def show_header():
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        if st.session_state.role == "user":
            st.markdown('<div class="nav-header">🤝 Friendbook - Your Safe Space</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="nav-header admin-theme">⚙️ Friendbook Admin Dashboard</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**👤 {st.session_state.username}**")
    
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

show_header()

role = st.session_state.role
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

def show_help_modal():
    with st.form("help_form"):
        st.markdown("### 💙 We're Here to Help")
        st.markdown("You're not alone. Let us know how we can support you.")
        
        feelings = st.text_area("How are you feeling right now?", 
                              placeholder="Share what's on your mind...", 
                              height=100)
        
        help_message = st.text_area("How can we help you today?", 
                                  placeholder="What kind of support do you need?", 
                                  height=100)
        
        help_type = st.selectbox("What kind of help do you need?", 
                               ["Someone to talk to", "Professional resources", "Emergency support", "Just someone to listen"])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("💙 Send Help Request", use_container_width=True):
                if feelings or help_message:
                    save_help_message(help_message, feelings, help_type)
                    st.success("✅ Your message has been sent. Someone will reach out soon.")
                    st.balloons()
                else:
                    st.warning("Please share your feelings or how we can help.")
        
        with col2:
            if st.form_submit_button("📞 Emergency Help", use_container_width=True):
                st.error("🚨 **EMERGENCY RESOURCES**")
                st.markdown("""
                - **Crisis Hotline:** 988 (US)
                - **Text HOME to 741741** for crisis support
                - **Emergency Services:** 911
                - **International:** befrienders.org
                """)

# Navigation
if role == "admin":
    page = st.sidebar.selectbox("🎛️ Admin Panel", 
                               ["📊 Dashboard", "🧪 Model Test", "🚨 Alerts List", "💬 Help Messages", "📈 Analytics"])
else:
    page = st.sidebar.selectbox("📱 Menu", ["🏠 Feed", "💙 Get Help"])

# ═══════════════════════════════════════════════════════════════════
# USER PAGES
# ═══════════════════════════════════════════════════════════════════

if page == "🏠 Feed":
    st.markdown('<div class="user-page">', unsafe_allow_html=True)
    
    # Initialize posts in session state
    if "posts" not in st.session_state:
        st.session_state.posts = []
    
    # Post creation form
    with st.form("post_form", clear_on_submit=True):
        st.markdown("### ✍️ Share Your Thoughts")
        text = st.text_area("What's on your mind?", 
                          placeholder="Share your thoughts, feelings, or what's happening in your life...", 
                          height=120)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            post_button = st.form_submit_button("📝 Post", use_container_width=True)
        
        if post_button and text:
            with st.spinner("Analyzing your post..."):
                preds = predict_finetuned(text)
                top = max(preds, key=lambda p: p["score"])
                flagged = top["label"] in {"moderate", "severe"}

                # Save post
                post_data = {
                    "text": text,
                    "flagged": flagged,
                    "timestamp": datetime.datetime.now(),
                    "predictions": preds,
                    "username": st.session_state.username
                }
                st.session_state.posts.append(post_data)

                # Store flagged state in session
                st.session_state.last_post_flagged = flagged

                if flagged:
                    save_alert(text, preds)
                    st.success("📝 Post shared successfully!")
                    st.markdown("---")
                    st.error("🚨 **We noticed you might need support**")
                    st.markdown("Your well-being matters to us. Would you like to connect with someone?")
        else:
            # If no post was just made, ensure the variable exists
            if "last_post_flagged" not in st.session_state:
                st.session_state.last_post_flagged = False

    # Place this OUTSIDE the form, after the form block
    if st.session_state.last_post_flagged and st.button("💙 Yes, I'd like help"):
        st.session_state.show_help_modal = True
        st.session_state.last_post_flagged = False  # Reset after showing modal
        st.rerun()
    
    # Show help modal if requested
    if st.session_state.get("show_help_modal", False):
        show_help_modal()
        if st.button("❌ Close"):
            st.session_state.show_help_modal = False
            st.rerun()
    
    # Display posts
    st.markdown("---")
    st.markdown("### 📱 Recent Posts")
    
    if not st.session_state.posts:
        st.markdown("*No posts yet. Be the first to share something!*")
    else:
        for i, post in enumerate(reversed(st.session_state.posts)):
            post_class = "distress-post" if post["flagged"] else "post-card"
            
            st.markdown(f'<div class="{post_class}">', unsafe_allow_html=True)
            
            # Post header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**👤 {post['username']}**")
            with col2:
                st.markdown(f"*{post['timestamp'].strftime('%Y-%m-%d %H:%M')}*")
            
            # Post content
            st.markdown(post["text"])
            
            # Show distress level if flagged
            if post["flagged"]:
                top_pred = max(post["predictions"], key=lambda p: p["score"])
                st.markdown(f"⚠️ **Distress Level: {top_pred['label'].title()}** ({top_pred['score']:.2%})")
                
                if st.button(f"💙 Offer Support", key=f"support_{i}"):
                    st.session_state.show_help_modal = True
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "💙 Get Help":
    st.markdown('<div class="user-page">', unsafe_allow_html=True)
    
    st.markdown("### 💙 Mental Health Resources")
    
    tab1, tab2, tab3 = st.tabs(["🆘 Immediate Help", "📚 Resources", "💬 Talk to Someone"])
    
    with tab1:
        st.markdown("#### 🚨 Crisis Support")
        st.error("""
        **If you're in immediate danger or having thoughts of self-harm:**
        - **Call 988** (Suicide & Crisis Lifeline)
        - **Text HOME to 741741** (Crisis Text Line)
        - **Call 911** for emergency services
        """)
        
        st.markdown("#### 🌐 International Resources")
        st.info("""
        - **International:** [befrienders.org](https://befrienders.org)
        - **UK:** 116 123 (Samaritans)
        - **Canada:** 1-833-456-4566
        - **Australia:** 13 11 14 (Lifeline)
        """)
    
    with tab2:
        st.markdown("#### 📖 Mental Health Resources")
        st.markdown("""
        - **National Alliance on Mental Illness (NAMI):** [nami.org](https://nami.org)
        - **Mental Health America:** [mhanational.org](https://mhanational.org)
        - **Psychology Today:** Find therapists near you
        - **Headspace:** Meditation and mindfulness app
        - **BetterHelp:** Online therapy platform
        """)
    
    with tab3:
        show_help_modal()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# ADMIN PAGES
# ═══════════════════════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.markdown('<div class="admin-page">', unsafe_allow_html=True)
    
    # Load data
    alerts_df = pd.read_csv(ALERTS_CSV) if os.path.exists(ALERTS_CSV) else pd.DataFrame()
    help_df = pd.read_csv(HELP_MESSAGES_CSV) if os.path.exists(HELP_MESSAGES_CSV) else pd.DataFrame()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-card-bar" style="background:#1e2939;">
                ⚠️ Total Alerts
            </div>
            <div class="metric-card-value">{}</div>
        </div>
        '''.format(len(alerts_df)), unsafe_allow_html=True)

    with col2:
        severe_count = len(alerts_df[alerts_df["level"] == "severe"]) if not alerts_df.empty else 0
        st.markdown('''
        <div class="metric-card">
            <div class="metric-card-bar" style="background:#1e2939;">
                🚨 Severe Cases
            </div>
            <div class="metric-card-value">{}</div>
        </div>
        '''.format(severe_count), unsafe_allow_html=True)

    with col3:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-card-bar" style="background:#1e2939;">
                💬 Help Requests
            </div>
            <div class="metric-card-value">{}</div>
        </div>
        '''.format(len(help_df)), unsafe_allow_html=True)

    with col4:
        active_users = len(set(help_df["user_id"])) if not help_df.empty else 0
        st.markdown('''
        <div class="metric-card">
            <div class="metric-card-bar" style="background:#1e2939;">
                👥 Active Users
            </div>
            <div class="metric-card-value">{}</div>
        </div>
        '''.format(active_users), unsafe_allow_html=True)
    
    # Charts
    if not alerts_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Distress Levels Distribution")
            level_counts = alerts_df["level"].value_counts()
            # Define color mapping for the levels
            color_map = {
                "severe": "#155dfc",   # bleuish purple
                "moderate": "#8ec5ff" # orange
            }
            # Get colors for the levels present in the data
            colors = [color_map.get(level, "#90caf9") for level in level_counts.index]
            fig = px.pie(
                values=level_counts.values,
                names=level_counts.index,
                color=level_counts.index,
                color_discrete_sequence=colors,
                title="Distress Levels Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Alerts Over Time")
            alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"])
            daily_counts = alerts_df.groupby(alerts_df["timestamp"].dt.date).size()
            fig = px.line(x=daily_counts.index, y=daily_counts.values)
            fig.update_layout(xaxis_title="Date", yaxis_title="Number of Alerts")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🧪 Model Test":
    st.markdown('<div class="admin-page">', unsafe_allow_html=True)
    
    st.markdown("### 🧪 AI Model Testing Interface")
    
    txt = st.text_area("Enter text to analyze:", 
                      placeholder="Type or paste text here to test the distress detection model...", 
                      height=120)
    
    if txt:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎭 Emotion Analysis (Predefined)")
            if st.button("🎭 Analyze Emotions", use_container_width=True):
                with st.spinner("Analyzing emotions..."):
                    preds = predict_predefined(txt)
                    fig = emotion_bar_chart(preds)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🧠 Distress Analysis (Fine-tuned)")
            if st.button("🧠 Analyze Distress", use_container_width=True):
                with st.spinner("Analyzing distress levels..."):
                    preds = predict_finetuned(txt)
                    fig = emotion_bar_chart(preds)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        if st.button("📋 Show Detailed Analysis", use_container_width=True):
            st.markdown("#### 📋 Detailed Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Emotion Predictions:**")
                emotion_df = pd.DataFrame(predict_predefined(txt))
                st.dataframe(emotion_df, use_container_width=True)
            
            with col2:
                st.markdown("**Distress Predictions:**")
                distress_df = pd.DataFrame(predict_finetuned(txt))
                st.dataframe(distress_df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🚨 Alerts List":
    st.markdown('<div class="admin-page">', unsafe_allow_html=True)
    
    st.markdown("### 🚨 Mental Health Alerts")
    
    df = pd.read_csv(ALERTS_CSV)
    
    if df.empty:
        st.info("📭 No alerts recorded yet.")
    else:
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            levels = sorted(df["level"].unique())
            selected_levels = st.multiselect("🎚️ Filter by Distress Level", levels, default=levels)
        
        with col2:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            date_range = st.date_input("📅 Date Range", 
                                     value=[df["timestamp"].dt.date.min(), df["timestamp"].dt.date.max()])
        
        # Apply filters
        if len(date_range) == 2:
            mask = (df["level"].isin(selected_levels) & 
                   df["timestamp"].dt.date.between(date_range[0], date_range[1]))
            filtered_df = df[mask]
        else:
            filtered_df = df[df["level"].isin(selected_levels)]
        
        # Display results
        st.markdown(f"**📊 Showing {len(filtered_df)} of {len(df)} alerts**")
        
        for idx, row in filtered_df.iterrows():
            # Fix timestamp formatting
            timestamp_str = str(row['timestamp'])[:19] if len(str(row['timestamp'])) > 19 else str(row['timestamp'])
            
            with st.expander(f"⚠️ {row['level'].title()} Alert - {timestamp_str}"):
                st.markdown(f"**📝 Message:** {row['text']}")
                
                # Parse predictions
                try:
                    preds = json.loads(row['preds'])
                    st.markdown("**🧠 AI Confidence:**")
                    for pred in preds:
                        st.progress(pred['score'], text=f"{pred['label']}: {pred['score']:.2%}")
                except:
                    st.markdown("*Prediction data unavailable*")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "💬 Help Messages":
    st.markdown('<div class="admin-page">', unsafe_allow_html=True)
    
    st.markdown("### 💬 Help Requests from Users")
    
    if not os.path.exists(HELP_MESSAGES_CSV):
        st.info("📭 No help messages yet.")
    else:
        help_df = pd.read_csv(HELP_MESSAGES_CSV)
        
        if help_df.empty:
            st.info("📭 No help messages recorded yet.")
        else:
            st.markdown(f"**📊 Total Help Requests: {len(help_df)}**")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                help_types = help_df["help_type"].unique()
                selected_types = st.multiselect("🎯 Filter by Help Type", help_types, default=help_types)
            
            with col2:
                help_df["timestamp"] = pd.to_datetime(help_df["timestamp"])
                date_range = st.date_input("📅 Date Range", 
                                         value=[help_df["timestamp"].dt.date.min(), 
                                               help_df["timestamp"].dt.date.max()],
                                         key="help_date_range")
            
            # Apply filters
            if len(date_range) == 2:
                mask = (help_df["help_type"].isin(selected_types) & 
                       help_df["timestamp"].dt.date.between(date_range[0], date_range[1]))
                filtered_df = help_df[mask]
            else:
                filtered_df = help_df[help_df["help_type"].isin(selected_types)]
            
            # Display messages
            for idx, row in filtered_df.iterrows():
                # Fix timestamp formatting
                timestamp_str = str(row['timestamp'])[:19] if len(str(row['timestamp'])) > 19 else str(row['timestamp'])
                
                with st.expander(f"💙 {row['help_type']} - {row['user_id']} - {timestamp_str}"):
                    st.markdown(f"**😔 How they're feeling:**")
                    st.markdown(f"> {row['feelings']}")
                    
                    st.markdown(f"**🤝 How we can help:**")
                    st.markdown(f"> {row['message']}")
                    
                    st.markdown(f"**📞 Support Type:** {row['help_type']}")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"✅ Mark Resolved", key=f"resolve_{idx}"):
                            st.success("✅ Marked as resolved")
                    with col2:
                        if st.button(f"📞 Priority Follow-up", key=f"priority_{idx}"):
                            st.warning("📞 Added to priority follow-up list")
                    with col3:
                        if st.button(f"📋 Add Notes", key=f"notes_{idx}"):
                            st.text_area("Notes:", key=f"note_text_{idx}")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📈 Analytics":
    st.markdown('<div class="admin-page">', unsafe_allow_html=True)
    st.markdown("### 📈 Advanced Analytics Dashboard")
    
    # Load training results if available
    training_results_path = os.path.join(ROOT, "App/model/training_results.json")
    if os.path.exists(training_results_path):
        with open(training_results_path, 'r') as f:
            training_results = json.load(f)
        
        # --- Performance Bar Chart with tags ---
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1-Score', 'Precision', 'Recall'],
            'Score': [
                training_results.get('final_accuracy', 0.0),
                training_results.get('final_f1', 0.0),
                training_results.get('final_precision', 0.0),
                training_results.get('final_recall', 0.0)
            ]
        })
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎯 Model Performance")
            fig = px.bar(
                metrics_df, x='Metric', y='Score', 
                title="Model Performance Metrics",
                text=metrics_df['Score'].apply(lambda x: f"{x:.3f}")
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                yaxis=dict(range=[0.0, 1.0]),
                showlegend=False,
                margin=dict(t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Training History Line Chart ---
        with col2:
            st.markdown("#### 📊 Training History")
            if 'training_history' in training_results:
                history_df = pd.DataFrame(training_results['training_history'])
                if not history_df.empty:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    if 'loss' in history_df.columns:
                        fig.add_trace(
                            go.Scatter(x=history_df['epoch'], y=history_df['loss'], 
                                      name="Training Loss", mode="lines+markers"),
                            secondary_y=False,
                        )
                    
                    if 'eval_accuracy' in history_df.columns:
                        fig.add_trace(
                            go.Scatter(x=history_df['epoch'], y=history_df['eval_accuracy'], 
                                      name="Validation Accuracy", mode="lines+markers"),
                            secondary_y=True,
                        )
                    
                    fig.update_xaxes(title_text="Epoch")
                    fig.update_yaxes(title_text="Loss", secondary_y=False)
                    fig.update_yaxes(title_text="Accuracy", secondary_y=True)
                    fig.update_layout(
                        margin=dict(t=40, b=40),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("🔄 Training results not available. Run the model training script to generate analytics.")
    
    # Usage statistics
    alerts_df = pd.read_csv(ALERTS_CSV) if os.path.exists(ALERTS_CSV) else pd.DataFrame()
    help_df = pd.read_csv(HELP_MESSAGES_CSV) if os.path.exists(HELP_MESSAGES_CSV) else pd.DataFrame()
    
    if not alerts_df.empty:
        st.markdown("#### 📊 Usage Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Weekly Alert Trends**")
            level_option = st.selectbox(
                "Filter by distress level",
                options=["all", "severe", "moderate"],
                index=0,
                key="distress_level_filter"
            )

            alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"], format='mixed', errors='coerce')
            alerts_df = alerts_df.dropna(subset=["timestamp"])
            alerts_df["year"] = alerts_df["timestamp"].dt.isocalendar().year
            alerts_df["week"] = alerts_df["timestamp"].dt.isocalendar().week

            color_map = {
                "all": "blue",
                "severe": "red",
                "moderate": "orange"
            }

            # Prepare weekly counts
            weekly_total = alerts_df.groupby(["year", "week"]).size().reset_index(name="total")
            weekly_severe = alerts_df[alerts_df["level"] == "severe"].groupby(["year", "week"]).size().reset_index(name="severe")
            weekly_moderate = alerts_df[alerts_df["level"] == "moderate"].groupby(["year", "week"]).size().reset_index(name="moderate")

            # Merge for plotting
            weekly = pd.merge(weekly_total, weekly_severe, on=["year", "week"], how="left")
            weekly = pd.merge(weekly, weekly_moderate, on=["year", "week"], how="left")
            weekly = weekly.fillna(0)
            weekly["label"] = weekly.apply(lambda row: f"{row['year']}-W{row['week']}", axis=1)

            import plotly.graph_objects as go

            if level_option == "all":
                fig = go.Figure()
                # Bar for total
                fig.add_trace(go.Bar(
                    x=weekly["label"], y=weekly["total"], name="Total Alerts",
                    marker_color=color_map["all"],
                    text=weekly["total"], textposition="outside"
                ))
                # Line for severe
                fig.add_trace(go.Scatter(
                    x=weekly["label"], y=weekly["severe"], name="Severe",
                    mode="lines+markers+text", line=dict(color=color_map["severe"], width=3),
                    text=weekly["severe"], textposition="top center"
                ))
                # Line for moderate
                fig.add_trace(go.Scatter(
                    x=weekly["label"], y=weekly["moderate"], name="Moderate",
                    mode="lines+markers+text", line=dict(color=color_map["moderate"], width=3, dash="dash"),
                    text=weekly["moderate"], textposition="top center"
                ))
                fig.update_layout(
                    title="Weekly Alert Volume (All)",
                    xaxis_title="Year-Week", yaxis_title="Number of Alerts",
                    barmode="overlay"
                )
            else:
                # Filter for selected level
                weekly_alerts = alerts_df[alerts_df["level"] == level_option].groupby(["year", "week"]).size()
                labels = [f"{y}-W{w}" for y, w in weekly_alerts.index]
                if len(weekly_alerts) > 1:
                    fig = go.Figure(go.Scatter(
                        x=labels, y=weekly_alerts.values, mode="lines+markers+text",
                        name=level_option.title(), line=dict(color=color_map[level_option], width=3),
                        text=weekly_alerts.values, textposition="top center"
                    ))
                else:
                    fig = go.Figure(go.Bar(
                        x=labels, y=weekly_alerts.values, name=level_option.title(),
                        marker_color=color_map[level_option],
                        text=weekly_alerts.values, textposition="outside"
                    ))
                    st.info("Only one week of data available. Add more alerts over time to see trends.")

                fig.update_layout(
                    title=f"Weekly Alert Volume ({level_option.title()})",
                    xaxis_title="Year-Week", yaxis_title="Number of Alerts"
                )

            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Help Request Types**")
            if not help_df.empty:
                help_type_counts = help_df["help_type"].value_counts()
                fig = px.pie(values=help_type_counts.values, names=help_type_counts.index,
                           title="Distribution of Help Requests")
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
🤝 <strong>Friendbook</strong> - Mental Health Social Network<br>
Made with ❤️ for mental health awareness and support<br>
<em>Remember: You're never alone, and it's okay to ask for help</em>
</div>
""", unsafe_allow_html=True)