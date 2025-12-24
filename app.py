import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import plotly.express as px
import torch
import numpy as np
import lime
import lime.lime_text
import streamlit.components.v1 as components
from datetime import datetime
import csv
import os

# setting a confidence limit so we know when the model is unsure
CONFIDENCE_THRESHOLD = 0.70

# loading the model + tokenizer once so the app doesn’t reload it every time
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "./my_finbert_classifier"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error("couldn’t load the model… maybe training wasn’t done yet?")
        st.error(e)
        return None, None

tokenizer, model = load_model_and_tokenizer()

# this function sends text to the model and gets back probabilities
def predictor(texts):
    if not tokenizer or not model:
        return np.array([])

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
    return probs

# creating the LIME explainer so we can show “why” the model predicted something
explainer = lime.lime_text.LimeTextExplainer(
    class_names=model.config.id2label.values()
)

# writing user feedback into a csv file so we can check it later
def log_feedback(text, prediction, feedback):
    file_exists = os.path.exists("feedback.csv")
    with open("feedback.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Transaction Text", "Model Prediction", "User Feedback"])
        writer.writerow([datetime.now().isoformat(), text, prediction, feedback])


# adding some custom styling just to make the UI look nicer :)
st.markdown("""
    <style>
        .main { background-color:#f7f9fc; padding:20px; border-radius:15px; }
        h1, h2, h3 { text-align:center; color:#003366; }
        .footer { text-align:center; margin-top:40px; color:gray; }
        .prediction-box { background:#eaf4ff; border-radius:10px; padding:15px; }
        .university-logo { display:flex; justify-content:center; }
    </style>
""", unsafe_allow_html=True)

# header / titles
st.markdown("<h3>PAMUKKALE UNIVERSITY</h3>", unsafe_allow_html=True)
st.markdown("<h2>FINBERT TRANSACTION CLASSIFIER</h2>", unsafe_allow_html=True)
st.markdown("<h3>AI-Based Financial Transaction Categorization System</h3>", unsafe_allow_html=True)
st.markdown("---")

# sidebar info about the project
st.sidebar.header("About the Project")
st.sidebar.write("""
This project shows how AI can classify financial transactions:
- uses a FinBERT model trained on finance text
- explains predictions using LIME
- lets users give feedback to improve things
- warns when the model is not confident
- adds a small feedback score for fun :)
""")

# keeping state so the app remembers stuff between runs
if "history" not in st.session_state:
    st.session_state.history = []
if "feedback_score" not in st.session_state:
    st.session_state.feedback_score = 0
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_predictions" not in st.session_state:
    st.session_state.last_predictions = None

st.sidebar.metric("Your Feedback Score", st.session_state.feedback_score)
st.sidebar.info("Tip: you can enter multiple transactions separated by commas")

# button to show feedback log
if st.sidebar.button("View Feedback Log", use_container_width=True):
    try:
        feedback_df = pd.read_csv("feedback.csv")
        st.sidebar.dataframe(feedback_df)
    except FileNotFoundError:
        st.sidebar.warning("no feedback yet, you’re the first tester 😅")

# main instructions
st.write("""
Type a transaction (like "Amazon" or "Starbucks") and the model will guess the category.
""")

if model and tokenizer:
    user_input = st.text_area(
        "Enter transaction descriptions (comma separated):",
        "Spotify Subscription"
    )

    show_explanations = st.checkbox(
        "Generate Explanations & Give Feedback (a bit slower)"
    )

    if st.button("Classify Transactions"):
        texts = [x.strip() for x in user_input.split(",") if x.strip()]
        if texts:
            with st.spinner("model is thinking…"):
                # clearing old feedback for new results
                for key in list(st.session_state.keys()):
                    if key.startswith("feedback_for_"):
                        del st.session_state[key]

                st.session_state.last_predictions = predictor(texts)

                results = []
                for i, txt in enumerate(texts):
                    probs = st.session_state.last_predictions[i]
                    top_idx = np.argmax(probs)

                    results.append({
                        "Text": txt,
                        "Category": model.config.id2label[top_idx],
                        "Confidence": probs[top_idx]
                    })

                st.session_state.last_results = results
                st.session_state.history.extend(results)
        else:
            st.warning("write at least one transaction please")
            st.session_state.last_results = None

    # showing predictions
    if st.session_state.last_results:
        df = pd.DataFrame(st.session_state.last_results)

        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        st.success("Here are the results:")
        st.dataframe(df.style.format({"Confidence": "{:.2%}"}), use_container_width=True)

        # warning when model confidence is low
        for i, result in enumerate(st.session_state.last_results):
            if result["Confidence"] < CONFIDENCE_THRESHOLD:
                probs = st.session_state.last_predictions[i]
                top2 = np.argsort(probs)[-2:][::-1]
                st.warning(
                    f"Low confidence for `{result['Text']}` — maybe {model.config.id2label[top2[0]]} or {model.config.id2label[top2[1]]}",
                    icon="⚠️"
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # charts for this batch
        st.header("Batch Analysis")
        fig_conf = px.bar(df, x="Text", y="Confidence", color="Category",
                          title="Confidence per Transaction", text_auto=".2%")
        st.plotly_chart(fig_conf, use_container_width=True)

        fig_pie = px.pie(df, names="Category", title="Category Breakdown")
        st.plotly_chart(fig_pie, use_container_width=True)

        # optional explanations + feedback
        if show_explanations:
            st.header("Model Explanations & Feedback")

            for i, result in enumerate(st.session_state.last_results):
                explanation = explainer.explain_instance(
                    result["Text"], predictor, num_features=6, num_samples=1000
                )

                prob_df = pd.DataFrame({
                    "Category": model.config.id2label.values(),
                    "Probability": st.session_state.last_predictions[i]
                }).sort_values("Probability")

                st.markdown(f"#### `{result['Text']}`")
                st.bar_chart(prob_df.set_index("Category"))

                # feedback buttons
                feedback_key = f"feedback_for_{result['Text']}_{i}"
                if not st.session_state.get(feedback_key):
                    if st.button("👍 Correct", key=f"yes_{i}"):
                        log_feedback(result["Text"], result["Category"], "Correct")
                        st.session_state.feedback_score += 1
                        st.session_state[feedback_key] = True
                        st.rerun()
                    if st.button("👎 Incorrect", key=f"no_{i}"):
                        log_feedback(result["Text"], result["Category"], "Incorrect")
                        st.session_state.feedback_score += 1
                        st.session_state[feedback_key] = True
                        st.rerun()

# showing last few predictions
if st.session_state.history:
    st.header("Recent History")
    hist = pd.DataFrame(st.session_state.history)
    st.dataframe(hist.tail(10).style.format({"Confidence": "{:.2%}"}))

# footer credits
st.markdown("""
<div class="footer">
    <strong>Developed by:</strong>
    Ahmad Khaled Samim & Abdul Hakim Nazari<br><br>
    <strong>Mentor:</strong> Arş. Gör. MERVE ÖZDEŞ DEMİR<br><br>
    © 2025 Pamukkale University | FinBERT Transaction Classifier Project
</div>
""", unsafe_allow_html=True)
