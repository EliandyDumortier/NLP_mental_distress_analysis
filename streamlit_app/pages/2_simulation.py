import streamlit as st

# Import des librairies pour le modèle
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import create_optimizer
import pandas as pd

# Configuration
MODEL_NAME = 'bert-base-uncased'

# Chargement du tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# chemin vers le modele
from pathlib import Path
modele_path = (Path(__file__).parent.parent.parent / "modele").resolve()

# Chargement du modèle
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
model = TFAutoModelForSequenceClassification.from_pretrained(modele_path)


# Tokenization
def encode(texts):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

# fonction de prédiction
def predict_statement(statement):
    inputs = encode([statement])
    outputs = model(inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    return predicted_class


# Test prédiction
# example = "It's enough. I don't want to live now"
# status = predict_statement(example)
# print(f"Predicted status: {status}")





# Front-end

st.title("Simulation Sentences")

# simulation's sentence form
with st.form("my_form"):
    sentence_input = st.text_input("Write a sentence on the field :")
    submitted = st.form_submit_button("Submit")

if submitted:
    result = predict_statement(sentence_input)
    st.write(result)
